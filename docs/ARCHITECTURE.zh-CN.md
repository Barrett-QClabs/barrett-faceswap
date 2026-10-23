[English](ARCHITECTURE.md) | [中文](ARCHITECTURE.zh-CN.md)

# 架构

对文件的公开调用是 `FaceSwapPipeline.run`。第四阶段加上实时摄像头循环、在 CPU 上保持空闲的 TensorRT 与 CUDA 图选项、fp16 旁路文件查找、商业使用门禁，以及一个不实际训练的训练脚手架。第三阶段的可选换脸图、像素提升、多人脸选择、泊松融合、可续跑任务和可选的无磁盘管道仍然保留。视频目标会交给 `VideoPipeline`。文件路径和实时路径都会在处理器前后调用 `CompliancePolicy`，续跑的任务和管道模式也一样。处理器只返回帧，不写输出文件，因此不能靠某个处理器提前存盘来绕过水印。

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

磁盘模式（`video.mode: disk`，默认）把帧写在临时目录里，编码结束后删除。带任务编号时，这些帧留在 `jobs/<id>/{raw,processed}`，续跑可以复用。`video.threads` 默认是 4。干跑使用这个线程池。真实运行把人脸补全留在一条线程上，并让 PNG 的读取和写入重叠。逐帧分数笔记被关掉（`log_every_frame` 为 false），长视频不会每帧追加一行。一条汇总笔记记录处理了多少帧，以及检测到、重试后找到、补上和原样通过的数量。多人脸、参考脸、映射，以及任何任务都会强制单线程，这样跟踪器和 `next_frame` 保持顺序。

真实视频在检测失败时不会中止。第一次没找到时，会降低阈值，再试旋转（含 ±90°），然后把输入放大 1.5 倍。`video.max_face_gap`（默认 10）以内的空洞会用相邻帧插值，或者沿用较近一侧。超出这个间隔的帧保持原样。`video.smooth` 是 `one_euro`（默认）、`ema` 或 `none`，原样通过后会重置。静态图像仍然抛出 `NoFaceError`。干跑不做检测。管道模式用队列让 ffmpeg 的解码和编码重叠（解码端带 `-thread_queue_size 32`）。换脸器在一次任务里保持同一个检测器和同一个换脸后端，因此 inswapper 矩阵只读一次。贴回只变换裁剪的包围盒，而不是整帧。Lab 和 mean/std 色彩匹配使用带蒙版的均值和标准差。

驱动无法编码时，`h264_nvenc` 不会退回 `libx264`。`video.keep_audio` 为 true 时保留音轨。完整文件先尝试流复制，失败再试 AAC。裁剪过的片段总会把音频重编码为 AAC。`--no-audio` 丢掉音轨。

过滤抽检使用 `video.filter_every_n_frames`（默认 30），并总会包含最后一帧。强制打开时，水印打在每一帧输出上，而不只是抽检帧。`apply_video_metadata` 写容器注释。那次 ffmpeg 调用失败时，运行仍然成功：帧上已经有标记，旁路文件也会写上。强制关闭时跳过过滤、水印和审计。这种情况下 `before_process` 立即返回，不要求抽检帧。

管道模式（`--video-mode pipe`）在 ffmpeg 的标准输出上解码原始 BGR24，走同一套处理器，再在标准输入上编码原始 BGR24。它不为每一帧写 PNG。`h264_nvenc` 仍然不会退回。尺寸会向下取偶数，以满足 yuv420p。合规仍然只调用一次 `before_process`：抽检集合按时长乘以帧率估计，并加上最后一帧解码结果。这次抽检解码不会启动编码器。输出抽检在水印之前。强制打开时每一帧输出都会打水印。`complete_output` 写一行审计。失败时会中止管道并删除不完整的输出。续跑把 `next_frame / fps` 加到裁剪起点上。这个定位是时间戳，不是精确到帧的序号。

## 换脸器

`swap_catalog.py` 为每个可选模型写出 ONNX 约定。`--swapper-model` 和界面下拉框设置 `swap.model`。默认是 `hyperswap_1a_256`。`inswapper_128` 仍然可用，角色仍是 `swapper`，因此 inswapper 运行里已有的 `ensure_present()` 调用继续有效。其他图是可选角色。一次运行会打开检测器、识别器、选中的换脸器，以及该换脸器声明的 crossface 文件。`CORE_ROLES` 没有变，所以缺少可选权重不会挡住 inswapper 换脸。hyperswap 运行需要 `models/hyperswap_1a_256.onnx`，它不在默认的 buffalo 下载里。

| 名称 | 原生尺寸 | 模板 | 归一化 | 隐向量 | Crossface |
| --- | --- | --- | --- | --- | --- |
| `inswapper_128` | 128 | inswapper（ArcFace，128 时再平移 8 像素，并随尺寸缩放） | 0..1 RGB | `embedding @` 512×512 矩阵。优先使用名为 `emap` 的初始值；否则用最后一个初始值（官方文件里叫 `buff2fs`） | 无 |
| `hyperswap_1a/1b/1c_256` | 256 | `arcface_128`（按裁剪尺寸的 inswapper 布局） | 均值 0.5、标准差 0.5，再对输出反归一化 | L2 归一化的 ArcFace | 无 |
| `simswap_256` | 256 | ArcFace 112 v1 按裁剪边长缩放 | ImageNet 均值和标准差 | crossface，再 L2 | `crossface_simswap` |
| `simswap_unofficial_512` | 512 | ArcFace 112 v1 按裁剪边长缩放 | 0..1 RGB | crossface，再 L2 | `crossface_simswap` |
| `blendswap_256` | 256 | 目标用 FFHQ；源是 112 的 ArcFace 112 v2 | 两张图都是 0..1 RGB | 源裁剪，不是向量 | 无 |
| `ghost_1/2/3_256` | 256 | ArcFace 112 v1 按裁剪边长缩放 | 均值 0.5、标准差 0.5，再对输出反归一化 | crossface，原始向量（不再 L2） | `crossface_ghost` |

图像和隐向量按秩绑定（4 维图像、2 维隐向量），不写死输入名。BlendSwap 有两个 4 维输入。名为 `source` 和 `target` 的输入优先；否则空间更小的那个是源。可选的反归一化之后，输出解码与增强器相同：有值低于 `-0.01` 时按 [-1, 1]，最大值高于 `1.5` 时按 0..255，其余按 0..1。crossface 文件是操作者提供的 512 维到 512 维 ONNX。本仓库不附带这个文件，也不复制 FaceFusion、SimSwap 或 GHOST 的源码。新记录的 `commercial_use` 都是 false。来源和上面的约定写在 [scripts/download_models.zh-CN.md](../scripts/download_models.zh-CN.md)。`barrett-faceswap compare` 写出带标签的联系表和 `.timing.txt`。`--dry-run` 不打开权重。

## 像素提升

`pixel_boost` 先按提升后的边长对齐裁剪，再取交错的整脸子采样 `crop[row::count, col::count]`。每一块用同一个隐向量推理，并写回同一组格子。这些块不是四个象限。允许的边长是 128、256、512、768、1024，且边长不小于原生尺寸并能整除。`0` 表示原生尺寸。inswapper 在 boost 为 `0` 时仍遵守 `swap.resolution`，因此测试可以把模拟图指到一个不是整块倍数的尺寸。边长不能整除原生尺寸时，只做一次推理。视频使用同一个换脸器，因此 `--pixel-boost 256` 或 `512` 同样切块。256 网络上的 512 裁剪是整张脸的四路交错视图。inswapper（原生 128）上的 `256` 仍然是四块 128 像素。

`--eye-mode` 决定 inswapper 之后眼睛怎么处理。128 的裁剪经常带上源脸的眼睑，直接替换会看起来窄、或者半闭。

| `--eye-mode` | 保留什么 |
| --- | --- |
| `swap` | 只保留换过的裁剪。 |
| `blend`（默认） | 在眼部蒙版里混入目标裁剪。眼睛张开时，这个权重从 0.65 提高到大约 0.91。106 点用每一簇眼睛的垂直跨度除以宽度来衡量张开程度。五点没有眼睑，因此按张开处理。 |
| `target` | 原来的目标眼睛。 |

五点关键点变成两个椭圆。106 点集合使用上半脸左右两簇的凸包。加载了 `models/bisenet_face_parsing.onnx` 时，左右眼和左右眉的类别会在换脸上替换这个几何蒙版。`blend.eye_sharpen`（默认 0.6）只在这个区域内做反锐化。`--with-enhancer` 在列表里还没有 `face_enhancer` 时把它追加进去。眼睛张开时，`enhancer.eye_blend`（默认 0.35）会再降低，避免修复重画眼睑。`enhancer.blend`（默认 0.6）覆盖脸的其余部分。

## 关键点与表情

`landmarks.refine` 是 `auto`（默认）、`always` 或 `off`。`auto` 和 `always` 在 `models/2dfan4.onnx` 存在时打开它。裁剪把人脸框的长边缩放到 256 画布里的 195 像素。网络返回 68 个点。五个对齐点是 36:42 与 42:48 的均值、第 30 点，以及第 48 和第 54 点。`auto` 只在侧脸（`profile_ratio`，默认 0.22）或检测分数低于 `min_score`（默认 0.65）时替换检测器的五个点。正面脸保留检测器的点。`always` 替换每一张脸。`off` 不打开文件。文件缺失，或热图分数低于 0.05，都保留检测器关键点。68 个点放在 `Face.landmarks_68`，不会追加到五点数组上。`commercial_use` 保持 false。`--with-landmarks` 下载这个文件。哈希写在 `scripts/download_models.zh-CN.md`。

`expression.enabled`（默认 true）用这 68 个点把目标眼睛和抿起的嘴贴回去。眼睛的量是 `0.05 + 0.85 * closedness`，嘴的量是 `0.55 * pursedness`，使用 `ear_open` 0.24、`ear_closed` 0.15、`mar_open` 0.045、`mar_closed` 0.012。凸包是眼与眉、以及嘴，画在对齐后的裁剪里。这条路径运行时会跳过 `blend.eye_mode`，因为那个混合会在张开的五点脸上把大部分目标眼睛贴回去。没有 68 个点时，`blend.eye_mode` 仍然运行。增强器把 `1 - closedness` 当作 `eye_openness`。视频默认仍然不跑 `face_enhancer`，除非 `--with-enhancer` 或处理器列表把它加进去。

## 选脸

`swap.selector` 为 `index`（默认，先按面积从大到小，`--target-face-index`）、`many`（每一张检测到的脸，一个源）、`reference`（对参考图做余弦相似度；没有参考图时对源脸），或 `map`。映射文件是 JSON 列表，每项为 `{source, target_index, reference}`。锁定的 `target_index` 直接分配，顺序是面积从大到小。其余脸在 `swap.similarity`（默认 0.45）以上一对一匹配。没有匹配时抛出 `swap.no_match`。

`FaceTracker` 是贪心 IoU（`swap.track_iou`，默认 0.3，连续错过 8 帧后丢掉）。一条轨迹一旦分到源槽，后面的帧就保持这个槽。参考和映射默认打开跟踪。

## 融合

`blend.mode` 是 `alpha`（默认的逆变换合成）或 `poisson`（`cv2.seamlessClone`，normal clone）。泊松会清掉一像素的边，避免蒙版碰到画面边缘；蒙版太小或 OpenCV 拒绝克隆时退回 alpha。`blend.color` 默认是 `lab`（在换上的区域里按 Lab 的均值和标准差匹配）。`mean_std`、`histogram` 和 `none` 仍然可用。`--color-transfer none` 会关掉匹配。

## 任务

`jobs/<id>/job.json` 保存请求、状态（`pending`、`running`、`interrupted`、`done`）和 `next_frame`（第一帧尚未完成的序号）。`barrett-faceswap jobs submit` 只排队，不运行。`jobs resume ID`、`jobs run` 和 `swap --job-id ID --resume` 会继续。硬停止可能把状态留在 `running`；续跑接受 `pending`、`running` 和 `interrupted`。已完成的编号会被拒绝。磁盘续跑跳过序号低于 `next_frame` 且已经写出的帧，并在 `raw/frame_*.png` 已存在时复用它们。水印步骤会走过每一帧已处理的图，并跳过已经有同名 `.wm` 标记的帧，因此续跑不会把色带画两次。已经写出但还没有标记的帧仍会打一次水印。管道续跑是上面说的时间戳定位。在标成 `done` 之前抛出的异常会把任务标成 `interrupted`。图像任务把 `next_frame` 设为 1。

## 人脸增强

`face_enhancer` 已注册，但不在默认处理器列表里，因此换脸不强制要求这些权重。用 `--processors face_swapper,face_enhancer`、`--with-enhancer` 或界面上的复选框打开它。`--face-enhancer-model`（别名 `--enhancer-model`）是 `gfpgan_1.4`（默认）、`codeformer`、`gpen_bfr_512`、`gpen_bfr_1024`、`gpen_bfr_2048` 或 `restoreformer_plus_plus`。`--enhancer-blend` 把修复结果和原始裁剪混合（`1` 保留修复结果，默认 `0.6`）。`--enhancer-weight` 是 CodeFormer 的保真系数（默认 `0.7`）。GFPGAN 图没有旁路输入，因此那里用不到这个系数。加载了 `models/bisenet_face_parsing.onnx` 时，混合只用皮肤类别，眼睛和眉毛像素留在未增强的换脸上。没有解析文件时仍走眼部椭圆混合。

裁剪按 FFHQ 五点模板对齐，模板按 `size / 512` 缩放，再用椭圆区域蒙版贴回。混合在脸上使用 `enhancer.blend`，在同一模板生成的眼部蒙版里使用 `enhancer.eye_blend`（默认 `0.35`）。更低的眼部系数会留下目标眼睛，包括眼睑，而不是让修复结果替换它们。`enhancer.temporal`（默认 `0.4`）在五点平均位移除以脸宽不超过 `enhancer.temporal_motion`（默认 `0.04`）时，把这一帧的混合裁剪和上一帧的增强裁剪再混一次。`temporal: 0` 关闭这次混合。位移更大时丢掉上一帧裁剪，一次运行结束时 `post_process` 也会清掉它。没有检测到脸时，帧保持原样，并记一条笔记。调用方已经提供 `known_faces` 时，增强器使用这些脸。

ONNX 约定是本地代码，不是 GFPGAN 或 CodeFormer 训练项目的副本：

- 输入是 NCHW RGB float32，用 `(pixel / 255 - 0.5) / 0.5` 归一化（范围 [-1, 1]）。
- 图里若有非 4 维输入，则送入 CodeFormer 权重。类型为 `tensor(double)` 的输入会转成 float64。浮点输入仍是 float32。
- 输出里若有值低于 `-0.01`，按 [-1, 1] 解码；最大值高于 `1.5` 时按 0..255 解码；其余按 0..1。
- 空间尺寸跟随会话。注册表的后备尺寸是 512。

两条记录的 `commercial_use` 都是 `False`。GFPGAN 公开的代码和 v1.4 权重是 Apache-2.0，网络里包含 NVIDIA Source Code License 下的 StyleGAN2，因此注册表不把该文件标成已获商业许可。CodeFormer 权重是 S-Lab License 1.0（非商业研究）。本仓库不附带这些文件。只有在你显式运行 `download-models --with-gfpgan` 时才会去取一份已发布的 GFPGAN ONNX；CI 不会。来源和从 `.pth` 导出的说明写在 [scripts/download_models.zh-CN.md](../scripts/download_models.zh-CN.md)。

## 蒙版

`masks.py` 生成内缩方框、区域和全 1 的遮挡占位，再逐元素取最小值（`merge_min`）。当对应的 ONNX 文件在磁盘上且开关打开时（`masks.occlusion`、`masks.parser`），`masking.py` 会替换占位。遮挡文件缺失时保持全 1。解析文件缺失时，如果换脸器传入了裁剪模板上的五个点，就用扩张后的关键点凸包并羽化；没有关键点时仍用椭圆。增强器自己的贴回仍用椭圆。解析文件已加载时，眼和眉的类别用作换脸的眼部蒙版。

视频和实时（`masks.stride`，默认 3）每隔两帧才跑遮挡和解析图，并在关键点位移除以脸宽不超过 `masks.stride_motion`（默认 0.04）时，把上一张裁剪蒙版按对齐变化贴到当前帧。位移更大就立刻重算。`stride: 1` 表示每一帧都跑。静态图像始终重算。两张图都在时用两个线程一起跑。加载时，填充不对称的 ConvTranspose 会被改成零填充再加一层切片，数值不变，并且可以被 CUDA 提供者执行。facefusion 的 xseg 图是通道在最后（`[N, 256, 256, 3]`）；这种布局会先转置，不会按 NCHW 送进去。`--occluder-model` 是 `xseg_1`（默认）、`xseg_2`、`xseg_3` 或 `sam2_refine`。最后一项是操作者放到 `models/sam2_refine.onnx` 的 ONNX。它接收 0..1 的 RGB 裁剪和粗糙的 xseg 蒙版，并返回一张蒙版。它不会被下载。

遮挡（`models/occlusion_xseg.onnx`，注册表尺寸 256）接收 0..1 的 RGB；静态输入是单通道时则接收灰度。通道在最后的静态形状也可以。数值越高表示该像素保持可见。最大值高于 `1.5` 时按 0..255 解释。`masks.occlusion_invert` 用来翻转那些预测遮挡物而不是人脸的图。`download-models --with-masks` 会下载 facefusion-assets 的 `xseg_1.onnx`（models-3.1.0）并存到这个路径。该文件的许可未经核实，很可能不能用于商业用途。xseg 这个想法来自 DeepFaceLab。那个项目的代码是 GPL-3.0，不在本仓库里。`commercial_use` 为 false。

解析（`models/bisenet_face_parsing.onnx`）使用 ImageNet 的均值和标准差。标签是 `parsing.py` 里的 19 个 CelebAMask-HQ 序号（0 背景到 18 帽子）。默认区域是皮肤、眉毛、眼睛、眼镜、鼻子、嘴和嘴唇。头发、衣服、帽子、脖子和耳朵留在外面。图可以返回类别分数或整数标签图。`--with-masks` 也会下载 facefusion-assets 的 `bisenet_resnet_34.onnx`（models-3.0.0）。该文件的许可未经核实，很可能不能用于商业用途。常见检查点在 CelebAMask-HQ 上训练，该数据集是非商业研究用途，因此 `commercial_use` 为 false。

## 界面和基准

`barrett-faceswap ui` 构建一个 Gradio Blocks 应用。Gradio 只在 `build_demo` / `launch_ui` 里导入，属于 `[ui]` 额外依赖。页面是深色（`#0B0D10` / `#14181F` / `#2563EB`），有两个页签：图片·视频，以及摄像头。随仓库提供的语言是 zh-CN。页头开关调用 `t()`，并重写每一条标题，包括两个页签按钮。主路径是源脸、目标、换脸器和质量预设（快速、均衡、高清、自定义）。均衡是 HyperSwap，关闭增强，遮挡模型 `xseg_1`，像素提升为 0。高级设置默认收起，里面仍有选脸、增强、蒙版、视频、执行、合规和输出路径。长视频会报告帧进度。摄像头页读取同一套模型设置，并与命令行共用 `LiveSession`。对视频点预览时抽出第一帧，再走静态图像流水线。开始换脸会写到输出路径。只有目标是视频并且点击的是开始换脸时，才会送出编码器和视频模式。合规复选框不能把已经打开的配置关掉。随仓库提供的复选框跟随配置，配置是关闭的。开始只是待命，第一帧摄像头画面做同意检查，停止时若强制打开会写一行审计。三条线程在 `barrett-faceswap live` 里，不在浏览器回调里。

`barrett-faceswap benchmark` 为处理器链计时，并打印各阶段毫秒数和 `realtime_fps`。`--source` 和 `--target` 一起传入时会在这两张静态图上跑检测。报告里的尺寸是目标图的真实尺寸。`--width` 和 `--height` 只用于绘制的后备人脸。不传静态图时，用一张绘制的人脸提供关键点，即使检测器不会给这张图打分，替换仍然会计时。`--live` 给一次会话计时（采集、检测、替换、输出），不会打开摄像头。`--dry-run` 不加载 ONNX。`nvidia-smi` 可用时读取显存，否则报告不可用。

## 实时

`barrett-faceswap live` 用三条线程做采集、处理和输出。每一跳是一个 `LatestQueue`。队列满时 `put_latest` 丢掉最旧的帧（`live.queue_size`，默认 1）。检测每隔 `live.detect_every` 帧做一次（默认 3）。中间帧由 `DetectionCache` 按上次匹配到的速度移动框和关键点。缓存更新之后，One-Euro 滤波（`video.smooth`，与文件路径相同）会在处理器对齐之前平滑关键点。第一个样本保持原样。视频已经在 `FaceStream` 里、对齐之前做过平滑，因此实时滤波不会在那条路径上再做一次。`prepare_source` 只对源脸做一次向量。设置了 `DISPLAY` 时打开预览窗口。`--no-preview` 跳过预览。`--virtual-camera /dev/video2` 经 ffmpeg 把原始 BGR 写到 v4l2loopback 设备。设备不存在时记一条日志，预览继续。

合规包住这条路径。`before_process` 检查源图和第一帧摄像头画面。强制打开时，每一帧输出在内存里加水印。`complete_output` 追加一行审计，不给容器文件打标签。

## 执行

`execution.precision` 是 `fp32`（默认）或 `fp16`。fp16 会在配置文件旁边查找 `{stem}_fp16.onnx`，找不到就退回 fp32。`execution.cuda_graph` 给 CUDA 提供者加上 `enable_cuda_graph`。只要会话报告 `CUDAExecutionProvider`，就会复用一个 IO binding；失败时退回 `session.run`。CPU 会话（包括 CI）仍走 `session.run`。TensorRT 选项（引擎缓存、计时缓存、构建优化级别 0–5）只在安装了 `TensorrtExecutionProvider` 时挂上。`tensorrt.fp16` 默认是 false，因为 fp16 会改变身份。`--execution-provider tensorrt` 会选中该提供者，然后再退回 CPU。引擎缓存的前缀是模型角色，因此 inswapper（`swapper`）和 xseg（`occluder`）不会共用同一个引擎名。同时列出 CUDA 和 TensorRT 时，`prefer_cuda` 仍然把 CUDA 放在前面。CPU 的 CI 会跳过两者，也不会创建缓存目录。基础包不依赖某个 ONNX Runtime 轮子。安装 `[cpu]` 或 `[gpu]` 附加项，不要两个都装。`doctor` 在两个发行名都存在时会警告。

## 商业路线

`--commercial` 把 `execution.commercial` 设为真。`guard_commercial` 会拒绝这次运行将打开、且注册表行缺失或 `commercial_use` 为 false 的角色。内置行全部为 false，因此该开关会拒绝默认图，蒙版仍开启时也包括遮挡和解析。没有任何 YAML 键能把内置行改成 true。你自己训练的模型，是法律复核之后新增的一行注册表。

`src/barrett_faceswap/train/` 检查一份本地清单，许可字符串必须正好是 `commercial-ok`，然后打印训练步骤，给出 ONNX 约定（NCHW RGB，范围 [-1, 1]，512 维隐向量，ArcFace 模板按 size/112 缩放），并可以计算全局 SSIM 和 ArcFace 余弦。它不下载数据，不打开训练循环，也不复制 GHOST 或 AEI-Net 的源码。那些项目是这条 Apache-2.0 路线所指的方向。这里的代码是原创的。入口是 `scripts/train_swapper.py --check`、`scripts/export_onnx.py --check` 和 `scripts/eval_swapper.py --self-test`。没有 `--check` 时，训练和导出脚本以状态 2 退出，不写权重。

## 文档站

`mkdocs.yml` 用 Material 和 `mkdocs-static-i18n`（`docs_structure: folder`）从 `docs/en` 与 `docs/zh-CN` 构建站点。`scripts/build_cli_reference.py` 根据 `build_parser()` 用两种语言写出 CLI 页。CI 以 `--check` 运行该脚本，并执行 `mkdocs build --strict`。不会部署。`docs/*.md` 里的叙述文件留在站点旁边，并且被排除在构建之外。

## 采纳的想法

**处理器协议。** `Processor` 有 `pre_check`、`process_frame` 和 `post_process`。`processors/registry.py` 把名称映射到工厂，工厂接收 pydantic 配置和提供者列表。`face_swapper` 和 `face_enhancer` 已注册。这是 FaceFusion 那类接缝。方法和注册表是本地代码。

**会话缓存。** `ModelStore.session` 用解析后的路径、大小、mtime、请求的提供者元组、设备号、CUDA 图开关和精度作为键。命中缓存时直接返回会话，不再计算文件哈希。哈希只在会话第一次打开时计算。`file_sha256` 也按路径、大小和 mtime 缓存，因此之后的 doctor 或 inspect 不会重读一个没有变化的权重。文件被替换后大小或 mtime 会变，这时会重新哈希并加载。创建会话时加锁。重复调用不会重建会话。`prefer_cuda` 在请求包含 `cuda` 时把它放到最前，`cpu` 留在最后。如果安装的 ONNX Runtime 没有 CUDA 提供者，这个名称会被跳过，CPU 仍然保留。写出 `tensorrt` 时，如果该提供者已安装，会挂上引擎缓存、计时缓存、fp16 和构建优化级别；CPU 轮子上则跳过。驱动 580 上的 RTX 5090（Blackwell，`sm_120`）使用 `onnxruntime-gpu[cuda,cudnn]`（CUDA 13 和 cuDNN 9）。创建提供者时，如果存在 `onnxruntime.preload_dlls()` 就会调用，并把 NVIDIA pip 包里的 `lib` 目录加到前面。找不到库也不会抛错。CI 安装 `[cpu]` 附加项。`barrett-faceswap doctor` 打印提供者、预加载状态和模型哈希，并在同时安装了 `onnxruntime` 和 `onnxruntime-gpu` 时警告。

**对齐和贴回。** `align.py` 放着 ArcFace 五点模板和 FFHQ 512 模板。112 是识别裁剪。128 在此基础上向右平移 8 像素，这是 inswapper 导出所期望的。增强器按 `size / 512` 缩放 FFHQ 模板。`blend.py` 把裁剪逆变换到它的包围盒上，再按蒙版合成。

**模型注册表。** `model_registry.py` 记录检测器、识别器、inswapper、可选的 HyperSwap / SimSwap / BlendSwap / Ghost / crossface 角色、GFPGAN、CodeFormer、GPEN、RestoreFormer++、xseg 遮挡、`sam2_refine` 和解析：文件名、许可键、`commercial_use`、对齐名称和输入尺寸。sha256 锁定值在配置里，默认是 null，因为本仓库不重新分发这些文件。可选包的下载哈希写在 `fetch.py`。`ensure_present()` 默认只检查换脸用的三个角色，因此增强、蒙版和额外换脸文件保持可选。`inspect-models` 每个角色打印一行。每一条内置行的 `commercial_use` 都保持 false。

**配置。** `config.py` 是 pydantic 模型。CLI 和界面从 YAML 加载同一个模型。相对路径相对工作目录解析。

**国际化。** `i18n/locales/en.yaml` 和 `zh_CN.yaml` 是仅有的用户可见文案目录。CLI 帮助、日志、错误、合规文本和每一条 Gradio 标题都走 `t(key)`。语言顺序是 `--lang`，然后 `BARRETT_FACESWAP_LANG`，然后 `ui.language`，最后是 `en`。PyYAML 1.1 会把 `yes`、`no` 读成布尔值，这些键要加引号。

**合规。** 见 [COMPLIANCE.zh-CN.md](COMPLIANCE.zh-CN.md)。总开关默认关闭。策略对象仍然会执行，视频、续跑的任务和管道模式也不例外。

## 这个仓库的差别

- 代码是 MIT，没有打包 FaceFusion、Deep-Live-Cam、roop 或 SimSwap 的源码
- `download-models` 是显式命令。安装器、界面和 CI 都不会下载权重
- 合规是完整的包，操作者不打开就保持关闭
- 文案走英文和中文目录
- 哈希锁定在操作者填写之前是可选的
- 内容过滤的默认提供者是 stub：放行该帧，并写明没有加载检测器
- 蒙版模型在文件缺失时退回占位，因此测试和 CI 不需要权重

## 仍然延后

这组之外的换脸器（uniface、hififace 等），以及精确到帧的管道续跑。管道续跑仍按时间戳定位。中文水印用原创位图覆盖 `人工智能生成`；其他字符需要 `compliance.watermark.font_path`。训练脚手架不训练、也不导出权重。TensorRT 和 CUDA 图只在安装了对应提供者时运行。`compliance.enabled` 在正式发布前保持 false。
