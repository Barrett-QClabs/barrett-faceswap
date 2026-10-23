[English](ARCHITECTURE.md) | [中文](ARCHITECTURE.zh-CN.md)

# 架构

公开调用是 `FaceSwapPipeline.run`。第三阶段加上可选换脸图、像素提升、多人脸选择、泊松融合、可续跑的任务队列，以及可选的无磁盘视频管道。视频目标会交给 `VideoPipeline`。两条路径都会在处理器前后调用 `CompliancePolicy`，续跑的任务和管道模式也一样。处理器只返回帧，不写输出文件，因此不能靠某个处理器提前存盘来绕过水印。

`configs/default.yaml` 里的 `compliance.enabled` 保持 `false`。策略对象仍然会执行。在这个默认值下，它会记录强制已关闭，并写出不带标记的文件。

第一阶段的背景阅读在 [RESEARCH.zh-CN.md](RESEARCH.zh-CN.md)。

## 静态图像

```text
SwapRequest
  -> CompliancePolicy.before_process
       同意、过滤权限、源/目标过滤        （仅在强制打开时）
  -> 按顺序执行处理器
       face_swapper：检测 -> 对齐 -> 向量 -> 替换 -> 蒙版 -> 贴回
       face_enhancer：GFPGAN 1.4 或 CodeFormer，再按混合系数贴回（可选）
  -> CompliancePolicy.after_process
       输出过滤、水印、审计、写文件
```

`--dry-run` 会跳过 ONNX，但仍然经过策略。对静态图像传入 `--trim-start`、`--trim-end` 或 `--video-encoder` 是配置错误。静态图像上使用 `video_mode: pipe` 也是配置错误。静态图像带任务编号时，会记一帧并把任务标成完成。

## 视频

后缀为 `.mp4`、`.mkv`、`.mov`、`.webm`、`.avi` 或 `.m4v` 的目标走视频路径。源文件必须是静态图像。输出后缀必须是 `.mp4`、`.mkv` 或 `.mov`。

```text
ffprobe
  -> ffmpeg 抽出 PNG 帧（可选 -ss / -to 裁剪，放在 -i 之前）
  -> CompliancePolicy.before_process 只调用一次
       强制打开时抽检目标帧
  -> ThreadPoolExecutor
       与静态图像相同的处理器链，逐帧执行
  -> 抽检输出帧
  -> 强制和水印都打开时，给每一帧输出打水印
  -> ffmpeg 编码（默认 libx264，或 h264_nvenc）
  -> 恢复原始音轨
  -> 容器注释 + .meta.json 旁路文件，一条审计
```

磁盘模式（`video.mode: disk`，默认）把帧写在临时目录里，编码结束后删除。带任务编号时，这些帧留在 `jobs/<id>/{raw,processed}`，续跑可以复用。`video.threads` 默认是 4。工作线程共享 ONNX 会话。逐帧分数笔记被关掉（`log_every_frame` 为 false），长视频不会每帧追加一行。一条汇总笔记记录处理了多少帧。多人脸、参考脸、映射，以及任何任务都会强制单线程，这样跟踪器和 `next_frame` 保持顺序。

驱动无法编码时，`h264_nvenc` 不会退回 `libx264`。`video.keep_audio` 为 true 时保留音轨。完整文件先尝试流复制，失败再试 AAC。裁剪过的片段总会把音频重编码为 AAC。`--no-audio` 丢掉音轨。

过滤抽检使用 `video.filter_every_n_frames`（默认 30），并总会包含最后一帧。强制打开时，水印打在每一帧输出上，而不只是抽检帧。`apply_video_metadata` 写容器注释。那次 ffmpeg 调用失败时，运行仍然成功：帧上已经有标记，旁路文件也会写上。强制关闭时跳过过滤、水印和审计。这种情况下 `before_process` 立即返回，不要求抽检帧。

管道模式（`--video-mode pipe`）在 ffmpeg 的标准输出上解码原始 BGR24，走同一套处理器，再在标准输入上编码原始 BGR24。它不为每一帧写 PNG。`h264_nvenc` 仍然不会退回。尺寸会向下取偶数，以满足 yuv420p。合规仍然只调用一次 `before_process`：抽检集合按时长乘以帧率估计，并加上最后一帧解码结果。这次抽检解码不会启动编码器。输出抽检在水印之前。强制打开时每一帧输出都会打水印。`complete_output` 写一行审计。失败时会中止管道并删除不完整的输出。续跑把 `next_frame / fps` 加到裁剪起点上。这个定位是时间戳，不是精确到帧的序号。

## 换脸器

`swap_catalog.py` 为每个可选模型写出 ONNX 约定。`--swapper-model` 和界面下拉框设置 `swap.model`。默认仍是 `inswapper_128`，角色仍是 `swapper`，因此已有的 `ensure_present()` 调用继续有效。其他图是可选角色。一次运行会打开检测器、识别器、选中的换脸器，以及该换脸器声明的 crossface 文件。`CORE_ROLES` 没有变，所以缺少可选权重不会挡住 inswapper 换脸。

| 名称 | 原生尺寸 | 模板 | 归一化 | 隐向量 | Crossface |
| --- | --- | --- | --- | --- | --- |
| `inswapper_128` | 128 | inswapper（ArcFace，128 时再平移 8 像素，并随尺寸缩放） | 0..1 RGB | ONNX 初始化器里的 `embedding @ emap` | 无 |
| `hyperswap_1a/1b/1c_256` | 256 | ArcFace 按 size/112 缩放 | [-1, 1] | crossface | `crossface_hyperswap` |
| `simswap_256` | 256 | ArcFace 按 size/112 缩放 | [-1, 1] | crossface | `crossface_simswap` |
| `simswap_unofficial_512` | 512 | ArcFace 按 size/112 缩放 | [-1, 1] | crossface | `crossface_simswap` |
| `ghost_1/2/3_256` | 256 | ArcFace 按 size/112 缩放 | [-1, 1] | crossface | `crossface_ghost` |

图像和隐向量按秩绑定（4 维图像、2 维隐向量），不写死输入名。输出解码与增强器相同：有值低于 `-0.01` 时按 [-1, 1]，最大值高于 `1.5` 时按 0..255，其余按 0..1。crossface 文件是操作者提供的 512 维到 512 维 ONNX。本仓库不附带这个文件，也不复制 FaceFusion、SimSwap 或 GHOST 的源码。新记录的 `commercial_use` 都是 false。来源和上面的约定写在 [scripts/download_models.zh-CN.md](../scripts/download_models.zh-CN.md)。

## 像素提升

`pixel_boost` 先按提升后的边长对齐裁剪，再切成连续的原生尺寸方块，用同一个隐向量逐块推理，然后拼回去。允许的边长是 128、256、512、768、1024，且边长不小于原生尺寸并能整除。`0` 表示原生尺寸。inswapper 在 boost 为 `0` 时仍遵守 `swap.resolution`，因此测试可以把模拟图指到一个不是整块倍数的尺寸。边长不能整除原生尺寸时，只做一次推理。

## 选脸

`swap.selector` 为 `index`（默认，先按面积从大到小，`--target-face-index`）、`many`（每一张检测到的脸，一个源）、`reference`（对参考图做余弦相似度；没有参考图时对源脸），或 `map`。映射文件是 JSON 列表，每项为 `{source, target_index, reference}`。锁定的 `target_index` 直接分配，顺序是面积从大到小。其余脸在 `swap.similarity`（默认 0.45）以上一对一匹配。没有匹配时抛出 `swap.no_match`。

`FaceTracker` 是贪心 IoU（`swap.track_iou`，默认 0.3，连续错过 8 帧后丢掉）。一条轨迹一旦分到源槽，后面的帧就保持这个槽。参考和映射默认打开跟踪。

## 融合

`blend.mode` 是 `alpha`（默认的逆变换合成）或 `poisson`（`cv2.seamlessClone`，normal clone）。泊松会清掉一像素的边，避免蒙版碰到画面边缘；蒙版太小或 OpenCV 拒绝克隆时退回 alpha。`blend.color` 是 `mean_std`（默认）、`lab`、`histogram` 或 `none`。

## 任务

`jobs/<id>/job.json` 保存请求、状态（`pending`、`running`、`interrupted`、`done`）和 `next_frame`（第一帧尚未完成的序号）。`barrett-faceswap jobs submit` 只排队，不运行。`jobs resume ID`、`jobs run` 和 `swap --job-id ID --resume` 会继续。硬停止可能把状态留在 `running`；续跑接受 `pending`、`running` 和 `interrupted`。已完成的编号会被拒绝。磁盘续跑跳过序号低于 `next_frame` 且已经写出的帧，并在 `raw/frame_*.png` 已存在时复用它们。磁盘运行结束时的水印步骤仍会走过每一帧已处理的图，包括中断前写好的帧。管道续跑是上面说的时间戳定位。在标成 `done` 之前抛出的异常会把任务标成 `interrupted`。图像任务把 `next_frame` 设为 1。

## 人脸增强

`face_enhancer` 已注册，但不在默认处理器列表里，因此换脸不强制要求这些权重。用 `--processors face_swapper,face_enhancer` 或界面上的复选框打开它。`--enhancer-model` 是 `gfpgan_1.4`（默认）或 `codeformer`。`--enhancer-blend` 把修复结果和原始裁剪混合（`1` 保留修复结果，默认 `0.8`）。`--enhancer-weight` 是 CodeFormer 的保真系数（默认 `0.5`）。GFPGAN 图没有旁路输入，因此那里用不到这个系数。

裁剪按 FFHQ 五点模板对齐，模板按 `size / 512` 缩放，再用椭圆区域蒙版贴回。没有检测到脸时，帧保持原样，并记一条笔记。

ONNX 约定是本地代码，不是 GFPGAN 或 CodeFormer 训练项目的副本：

- 输入是 NCHW RGB float32，用 `(pixel / 255 - 0.5) / 0.5` 归一化（范围 [-1, 1]）。
- 图里若有非 4 维输入，则送入 CodeFormer 权重。
- 输出里若有值低于 `-0.01`，按 [-1, 1] 解码；最大值高于 `1.5` 时按 0..255 解码；其余按 0..1。
- 空间尺寸跟随会话。注册表的后备尺寸是 512。

两条记录的 `commercial_use` 都是 `False`。GFPGAN 公开的代码和 v1.4 权重是 Apache-2.0，网络里包含 NVIDIA Source Code License 下的 StyleGAN2，因此注册表不把该文件标成已获商业许可。CodeFormer 权重是 S-Lab License 1.0（非商业研究）。本仓库不下载、也不附带这些文件。来源写在 [scripts/download_models.zh-CN.md](../scripts/download_models.zh-CN.md)。

## 蒙版

`masks.py` 仍然生成内缩方框、椭圆占位和全 1 的遮挡占位，再逐元素取最小值（`merge_min`）。当对应的 ONNX 文件在磁盘上且开关打开时（`masks.occlusion`、`masks.regions_enabled`），`masking.py` 会替换占位。文件缺失时保留占位，并把路径写进日志。

遮挡（`models/occlusion_xseg.onnx`，注册表尺寸 256）接收 0..1 的 RGB；静态输入是单通道时则接收灰度。数值越高表示该像素保持可见。最大值高于 `1.5` 时按 0..255 解释。`masks.occlusion_invert` 用来翻转那些预测遮挡物而不是人脸的图。xseg 这个想法来自 DeepFaceLab。那个项目的代码是 GPL-3.0，不在本仓库里。`commercial_use` 为 false，因为权重由操作者自行提供，本仓库不授予该文件的权利。

解析（`models/bisenet_face_parsing.onnx`）使用 ImageNet 的均值和标准差。标签是 `parsing.py` 里的 19 个 CelebAMask-HQ 序号（0 背景到 18 帽子）。默认区域是皮肤、眉毛、眼睛、眼镜、鼻子、嘴和嘴唇。头发、衣服、帽子、脖子和耳朵留在外面。图可以返回类别分数或整数标签图。常见检查点在 CelebAMask-HQ 上训练，该数据集是非商业研究用途，因此 `commercial_use` 为 false。

## 界面和基准

`barrett-faceswap ui` 构建一个 Gradio Blocks 应用。Gradio 只在 `build_demo` / `launch_ui` 里导入，属于 `[ui]` 额外依赖。语言下拉框调用 `t()`，并重写每一条标题，包括预览和运行按钮，以及第三阶段的控件（换脸模型、像素提升、多人脸、参考脸、脸映射、融合、颜色、视频模式）。对视频点预览时抽出第一帧，再走静态图像流水线。运行会写到输出路径。只有目标是视频并且点击的是运行时，才会送出编码器和视频模式下拉框的值。合规复选框不能把已经打开的配置关掉。随仓库提供的复选框跟随配置，配置是关闭的。

`barrett-faceswap benchmark` 为处理器链计时。`--dry-run` 不加载 ONNX。会打印每秒帧数。`nvidia-smi` 可用时读取显存，否则报告不可用。

## 采纳的想法

**处理器协议。** `Processor` 有 `pre_check`、`process_frame` 和 `post_process`。`processors/registry.py` 把名称映射到工厂，工厂接收 pydantic 配置和提供者列表。`face_swapper` 和 `face_enhancer` 已注册。这是 FaceFusion 那类接缝。方法和注册表是本地代码。

**会话缓存。** `ModelStore.session` 用解析后的路径、sha256、请求的提供者元组以及设备号作为键。创建会话时加锁。重复调用不会重建会话。`prefer_cuda` 在请求包含 `cuda` 时把它放到最前，`cpu` 留在最后。如果安装的 ONNX Runtime 没有 CUDA 提供者，这个名称会被跳过，CPU 仍然保留。可以写出 `tensorrt`，日志会说明它延后：没有引擎缓存，也没有构建参数。RTX 5090（Blackwell）需要链接 CUDA 12.8 或更新版本的 `onnxruntime-gpu`。CI 跑的是 CPU 轮子。

**对齐和贴回。** `align.py` 放着 ArcFace 五点模板和 FFHQ 512 模板。112 是识别裁剪。128 在此基础上向右平移 8 像素，这是 inswapper 导出所期望的。增强器按 `size / 512` 缩放 FFHQ 模板。`blend.py` 做逆变换并按蒙版合成。

**模型注册表。** `model_registry.py` 记录检测器、识别器、inswapper、可选的 HyperSwap / SimSwap / Ghost / crossface 角色、GFPGAN、CodeFormer、遮挡和解析：文件名、许可键、`commercial_use`、对齐名称和输入尺寸。sha256 锁定值在配置里，默认是 null，因为本仓库不重新分发这些文件。`ensure_present()` 默认只检查换脸用的三个角色，因此增强、蒙版和额外换脸文件保持可选。`inspect-models` 每个角色打印一行。

**配置。** `config.py` 是 pydantic 模型。CLI 和界面从 YAML 加载同一个模型。相对路径相对工作目录解析。

**国际化。** `i18n/locales/en.yaml` 和 `zh_CN.yaml` 是仅有的用户可见文案目录。CLI 帮助、日志、错误、合规文本和每一条 Gradio 标题都走 `t(key)`。语言顺序是 `--lang`，然后 `BARRETT_FACESWAP_LANG`，然后 `ui.language`，最后是 `en`。PyYAML 1.1 会把 `yes`、`no` 读成布尔值，这些键要加引号。

**合规。** 见 [COMPLIANCE.zh-CN.md](COMPLIANCE.zh-CN.md)。总开关默认关闭。策略对象仍然会执行，视频、续跑的任务和管道模式也不例外。

## 这个仓库的差别

- 代码是 MIT，没有打包 FaceFusion、Deep-Live-Cam、roop 或 SimSwap 的源码
- 安装器、CLI、界面和 CI 都不会下载模型
- 合规是完整的包，操作者不打开就保持关闭
- 文案走英文和中文目录
- 哈希锁定在操作者填写之前是可选的
- 内容过滤的默认提供者是 stub：放行该帧，并写明没有加载检测器
- 蒙版模型在文件缺失时退回占位，因此测试和 CI 不需要权重

## 仍然延后

实时摄像头、CUDA 图、TensorRT 引擎缓存、本阶段这组之外的换脸器（blendswap、uniface、hififace 等）、自动下载权重、精确到帧的管道续跑，以及能在像素上画出中文水印的捆绑字体。OpenCV 的 Hershey 字体写不出 `人工智能生成`。可见色带仍走现有的拉丁文路径。容器注释和旁路文件携带 Unicode 字符串。管道续跑按时间戳定位。磁盘续跑会给已经写好的帧再打一次水印。
