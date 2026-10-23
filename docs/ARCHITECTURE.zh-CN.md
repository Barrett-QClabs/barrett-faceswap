[English](ARCHITECTURE.md) | [中文](ARCHITECTURE.zh-CN.md)

# 架构

第二阶段保留第一阶段的入口，并加上视频路径、真正的人脸增强、遮挡与解析蒙版、Gradio 界面和基准测试。公开调用仍然是 `FaceSwapPipeline.run`。视频目标会交给 `VideoPipeline`。两条路径都会在处理器前后调用 `CompliancePolicy`。处理器只返回帧，不写输出文件，因此不能靠某个处理器提前存盘来绕过水印。

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

`--dry-run` 会跳过 ONNX，但仍然经过策略。对静态图像传入 `--trim-start`、`--trim-end` 或 `--video-encoder` 是配置错误。

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

帧写在临时目录里，编码结束后删除。没有实现无磁盘的管道。`video.threads` 默认是 4。工作线程共享 ONNX 会话。逐帧分数笔记被关掉（`log_every_frame` 为 false），长视频不会每帧追加一行。一条汇总笔记记录处理了多少帧。

驱动无法编码时，`h264_nvenc` 不会退回 `libx264`。`video.keep_audio` 为 true 时保留音轨。完整文件先尝试流复制，失败再试 AAC。裁剪过的片段总会把音频重编码为 AAC。`--no-audio` 丢掉音轨。

过滤抽检使用 `video.filter_every_n_frames`（默认 30），并总会包含最后一帧。强制打开时，水印打在每一帧输出上，而不只是抽检帧。`apply_video_metadata` 写容器注释。那次 ffmpeg 调用失败时，运行仍然成功：帧上已经有标记，旁路文件也会写上。强制关闭时跳过过滤、水印和审计。这种情况下 `before_process` 立即返回，不要求抽检帧。

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

`barrett-faceswap ui` 构建一个 Gradio Blocks 应用。Gradio 只在 `build_demo` / `launch_ui` 里导入，属于 `[ui]` 额外依赖。语言下拉框调用 `t()`，并重写每一条标题，包括预览和运行按钮。对视频点预览时抽出第一帧，再走静态图像流水线。运行会写到输出路径。只有目标是视频并且点击的是运行时，才会送出编码器下拉框的值。合规复选框不能把已经打开的配置关掉。随仓库提供的复选框跟随配置，配置是关闭的。

`barrett-faceswap benchmark` 为处理器链计时。`--dry-run` 不加载 ONNX。会打印每秒帧数。`nvidia-smi` 可用时读取显存，否则报告不可用。

## 采纳的想法

**处理器协议。** `Processor` 有 `pre_check`、`process_frame` 和 `post_process`。`processors/registry.py` 把名称映射到工厂，工厂接收 pydantic 配置和提供者列表。`face_swapper` 和 `face_enhancer` 已注册。这是 FaceFusion 那类接缝。方法和注册表是本地代码。

**会话缓存。** `ModelStore.session` 用解析后的路径、sha256、请求的提供者元组以及设备号作为键。创建会话时加锁。重复调用不会重建会话。`prefer_cuda` 在请求包含 `cuda` 时把它放到最前，`cpu` 留在最后。如果安装的 ONNX Runtime 没有 CUDA 提供者，这个名称会被跳过，CPU 仍然保留。可以写出 `tensorrt`，日志会说明它延后：没有引擎缓存，也没有构建参数。RTX 5090（Blackwell）需要链接 CUDA 12.8 或更新版本的 `onnxruntime-gpu`。CI 跑的是 CPU 轮子。

**对齐和贴回。** `align.py` 放着 ArcFace 五点模板和 FFHQ 512 模板。112 是识别裁剪。128 在此基础上向右平移 8 像素，这是 inswapper 导出所期望的。增强器按 `size / 512` 缩放 FFHQ 模板。`blend.py` 做逆变换并按蒙版合成。

**模型注册表。** `model_registry.py` 为检测器、识别器、换脸器、GFPGAN、CodeFormer、遮挡和解析记录：文件名、许可键、`commercial_use`、对齐名称和输入尺寸。sha256 锁定值在配置里，默认是 null，因为本仓库不重新分发这些文件。`ensure_present()` 默认只检查换脸用的三个角色，因此增强和蒙版文件保持可选。`inspect-models` 打印合并后的一行。

**配置。** `config.py` 是 pydantic 模型。CLI 和界面从 YAML 加载同一个模型。相对路径相对工作目录解析。

**国际化。** `i18n/locales/en.yaml` 和 `zh_CN.yaml` 是仅有的用户可见文案目录。CLI 帮助、日志、错误、合规文本和每一条 Gradio 标题都走 `t(key)`。语言顺序是 `--lang`，然后 `BARRETT_FACESWAP_LANG`，然后 `ui.language`，最后是 `en`。PyYAML 1.1 会把 `yes`、`no` 读成布尔值，这些键要加引号。

**合规。** 见 [COMPLIANCE.zh-CN.md](COMPLIANCE.zh-CN.md)。总开关默认关闭。策略对象仍然会执行，视频也不例外。

## 这个仓库的差别

- 代码是 MIT，没有打包 FaceFusion、Deep-Live-Cam、roop 或 SimSwap 的源码
- 安装器、CLI、界面和 CI 都不会下载模型
- 合规是完整的包，操作者不打开就保持关闭
- 文案走英文和中文目录
- 哈希锁定在操作者填写之前是可选的
- 内容过滤的默认提供者是 stub：放行该帧，并写明没有加载检测器
- 蒙版模型在文件缺失时退回占位，因此测试和 CI 不需要权重

## 仍然延后

实时摄像头、CUDA 图、TensorRT 引擎缓存、像素提升、更多换脸器（hyperswap、SimSwap、ghost）、多人脸映射、泊松融合、任务续跑、无磁盘视频管道、自动下载权重，以及能在像素上画出中文水印的捆绑字体。OpenCV 的 Hershey 字体写不出 `人工智能生成`。可见色带仍走现有的拉丁文路径。容器注释和旁路文件携带 Unicode 字符串。
