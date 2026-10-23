[English](README.md) | [中文](README.zh-CN.md)

# Barrett FaceSwap

Barrett FaceSwap 可以对静态图像或视频换脸。它把常见的工程路径接起来：检测人脸、对齐、用 ONNX 做身份替换、可选地修复裁剪，再把脸贴回去。它不训练基础模型，也不附带模型权重。

本仓库的代码使用 MIT 许可。它能够调用的 InsightFace 权重（包括 `inswapper_128.onnx`）不是 MIT。这些权重按非商业研究发布，商业使用需要 InsightFace 另行授权。把工具指向真实的人之前，请先阅读 [docs/COMPLIANCE.zh-CN.md](docs/COMPLIANCE.zh-CN.md)。

## 合规默认值

强制执行默认**关闭**。

`configs/default.yaml` 里 `compliance.enabled` 为 `false`。本地干跑不会要求同意文件，不会打水印，也不会因为内容过滤而拦截。这是开发默认值。任何使用真实肖像的运行都应该打开强制：

- 把 `compliance.enabled` 设为 `true`，或
- 传入 `--enable-compliance`

强制打开后，流水线会要求同意记录，写入 `AI-generated` / `人工智能生成` 标识和文件元数据，运行内容过滤，并追加一行审计。模块本身始终是完整实现。测试会强制打开它。生产环境也应该打开。

## 可接受的使用

只有在对方授权了这次具体用途时，才使用其人脸。不要制作未经同意的换脸、色情深度伪造、涉及未成年人的性内容，或用于欺诈、冒充的材料。对外分享合成结果时要标明它是合成的。合规门是为了让这些规则能被执行。它不是法律意见，占位过滤器也不能代替同意。

## 功能

- SCRFD 风格的 ONNX 检测（`det_10g`）、ArcFace 向量（`w600k_r50`），以及可选换脸器：`inswapper_128`（默认）、`hyperswap_1a/1b/1c_256`、`simswap_256`、`simswap_unofficial_512`、`ghost_1/2/3_256`。注册表每一行都有许可键，并且 `commercial_use` 为 false。HyperSwap、SimSwap 和 Ghost 需要操作者提供的 crossface ONNX，把 ArcFace 映射到该图的隐向量
- 像素提升：把更大的对齐裁剪切成原生尺寸的小块，逐块推理后再拼回去（128 / 256 / 512 / 768 / 1024，且该尺寸必须能被所选模型整除）
- 选脸：按序号选一张、`--many-faces` 替换每一张、用 ArcFace 余弦相似度对参考脸，或用 JSON 做源脸到目标脸的映射。视频用一个小的 IoU 跟踪器保持这次选择
- 贴回默认是 alpha。`--blend-mode poisson` 使用 `cv2.seamlessClone`。颜色迁移默认是 `lab`（Lab 空间的均值和标准差）。`mean_std`、`histogram` 和 `none` 仍然可用。`--color-transfer none` 会关掉匹配
- 任务把规格和下一帧未完成的序号存在 `jobs/<id>/`，中断的视频可以续跑
- 视频默认把帧写到磁盘。`--video-mode pipe` 经 ffmpeg 管道解码和编码，不为每一帧写 PNG
- 蒙版按像素取最小值合并：内缩方框、xseg 风格的遮挡 ONNX、BiSeNet 人脸解析区域。遮挡文件缺失时仍是全 1 蒙版。解析文件缺失时，换脸裁剪用带羽化的关键点凸包；没有关键点时用椭圆
- `face_enhancer`：GFPGAN 1.4 或 CodeFormer ONNX，带混合系数。它已注册，但默认不在处理器列表里，因此换脸不强制要求这些文件
- 视频：ffprobe、抽帧、线程池走同一套处理器、`libx264` 或 `h264_nvenc`，再把原音轨混回去。可以裁剪时间段
- ONNX Runtime 会话按模型哈希、提供者列表、设备、精度和 CUDA 图标记缓存。请求里包含 CUDA 时优先 CUDA。安装了 TensorRT 提供者时会挂上引擎缓存、计时缓存、fp16 和构建优化级别。`execution.precision: fp16` 选用同目录的 `*_fp16.onnx`，文件不存在时退回 fp32。CUDA 会话会复用 IO binding。只有 `execution.cuda_graph` 打开时才请求 CUDA 图捕获
- 模型注册表包含许可、商业标记、对齐名称、输入尺寸，以及加载时可选的 sha256 校验
- CLI、视频流水线和界面共用的 Pydantic 配置
- 中英文目录（`--lang`、`ui.language`、`BARRETT_FACESWAP_LANG`）。Gradio 的语言下拉会切换全部标签
- 可选合规包：同意、水印、可插拔内容过滤、只追加的审计日志。视频只在开始时走一次前置检查，按间隔抽样过滤，强制打开时给每一帧加水印，并写一行审计。实时会话使用同一组钩子：第一帧做同意检查，每一帧输出加水印，每个会话一行审计
- 实时摄像头：`barrett-faceswap live` 用三条线程做采集、检测、替换和输出。有界队列丢掉过期帧。检测隔几帧做一次，其余帧由缓存插值。源脸向量只算一次。输出可以打开预览，在 Linux 上也可以写到 v4l2loopback。Gradio 页面有一个摄像头页，使用同一个会话对象
- `--commercial` 拒绝注册表里 `commercial_use` 为 false 的每一行。内置行保持 false。`scripts/train_swapper.py` 只检查本地的 `commercial-ok` 清单并打印计划。它不训练，也不下载
- `barrett-faceswap download-models` 下载 `buffalo_l` 和 `inswapper_128`，校验 sha256，支持断点续传，并在镜像之间重试。每个镜像的连接最多等 15 秒。`--from-dir` 会在下载前复制匹配的 ONNX 文件。`--with-gfpgan` 是可选项。CI 不会运行它。`barrett-faceswap doctor` 打印提供者、CUDA/cuDNN 预加载状态和本地模型哈希，并在同时安装两个 ONNX Runtime 轮子时发出警告
- 视频不会因为某一帧没有脸而中止。它会降低阈值、旋转并放大后重试，然后在 `video.max_face_gap`（默认 10）以内用相邻帧补上关键点。`--eye-mode blend`（默认）尽量保留目标眼睛。`--with-enhancer` 加上 GFPGAN 或 CodeFormer，眼部混合更强。`--pixel-boost 256` 或 `512` 会把 inswapper 切块，静态图和视频都一样
- `docs/en` 与 `docs/zh-CN` 下的双语文档站（MkDocs Material）。CI 会构建，不会部署

以后再做：这组之外的换脸器，以及精确到帧的管道续跑。`compliance.enabled` 在正式发布前保持 false。

## 安装

需要 Python 3.10 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[cpu,dev]"
```

依赖使用无界面的 OpenCV 轮子（`opencv-python-headless`，导入名仍是 `cv2`），这样服务器和 CI 不需要显示器。基础安装不固定 ONNX Runtime 轮子。`[cpu]` 附加项安装 `onnxruntime`。CI 使用这个附加项。

在 NVIDIA 机器上，包括 Ubuntu 24.04、驱动 580 的 RTX 5090（Blackwell，`sm_120`），用 GPU 额外依赖**替换** CPU 轮子：

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install -e ".[gpu]"
```

`[gpu]` 是 `onnxruntime-gpu[cuda,cudnn]`。onnxruntime-gpu 1.31 需要 CUDA 13 和 cuDNN 9（`libcublasLt.so.13`、`libcudnn.so.9`）。这些库来自 NVIDIA 的 pip 包。创建提供者时，如果存在 `onnxruntime.preload_dlls()` 就会调用它，并把 `site-packages/nvidia/cu13/lib` 和 `site-packages/nvidia/cudnn/lib` 加到 `LD_LIBRARY_PATH` 前面。不需要自己导出这个路径。不要同时留下 CPU 轮子和 GPU 轮子。两者都在时，`barrett-faceswap doctor` 会警告。CI 安装 `[cpu]` 附加项。没有安装对应提供者时，TensorRT 和 CUDA 图选项会被跳过。

```bash
barrett-faceswap doctor
```

Gradio 界面是可选项：

```bash
pip install -e ".[ui]"
```

## 快速开始

在仓库根目录，保持合规关闭：

```bash
python examples/make_synthetic.py
barrett-faceswap swap \
  --source examples/synthetic/source.png \
  --target examples/synthetic/target.png \
  --output outputs/dry-run.png \
  --dry-run
```

`python -m barrett_faceswap swap ...` 是同一个入口。合成图是色块，不是人脸。`--dry-run` 会跳过 ONNX。真正的替换需要 `scripts/download_models.zh-CN.md` 里的权重，以及你有权使用的照片。

视频用同一条 `swap` 命令。目标文件的后缀决定走帧流水线。默认编码器是 `libx264`。`--video-encoder h264_nvenc` 选择 NVENC，驱动不能编码时不会悄悄退回。

```bash
barrett-faceswap swap \
  --source examples/synthetic/source.png \
  --target path/to/authorized-clip.mp4 \
  --output outputs/clip.mp4 \
  --dry-run \
  --trim-start 1.0 \
  --trim-end 4.0
```

磁盘上有 `models/gfpgan_1.4.onnx` 时可以加上修复：

```bash
barrett-faceswap swap \
  --source path/to/authorized-source.png \
  --target path/to/authorized-target.png \
  --output outputs/swap.png \
  --processors face_swapper,face_enhancer \
  --enhancer-model gfpgan_1.4 \
  --enhancer-blend 0.8
```

`--enhancer-model codeformer` 选择另一张图。`--enhancer-weight` 是 CodeFormer 的保真输入。

Gradio：

```bash
barrett-faceswap ui
```

页面上有源、目标、输出、处理器、执行提供者、增强器、换脸模型、像素提升、多人脸、参考脸、脸映射、融合、颜色迁移、视频模式、视频编码器、商业门禁、预览，以及摄像头页。语言下拉（`zh-CN` / `en`）会按目录重写全部标签，包括按钮和摄像头页。视频预览只处理第一帧。目标是视频并且点击运行时，才会采用编码器和视频模式下拉框的值。静态图像的运行会忽略它们。除非勾选合规或配置里已经打开，否则合规保持关闭。摄像头页在浏览器允许摄像头之前不会打开设备。开始只是待命。第一帧做同意检查。合规打开时，停止会写审计行。

```bash
barrett-faceswap swap \
  --source path/to/authorized-source.png \
  --target path/to/authorized-clip.mp4 \
  --output outputs/clip.mp4 \
  --swapper-model simswap_256 \
  --pixel-boost 512 \
  --many-faces \
  --blend-mode poisson \
  --color-transfer lab \
  --video-mode pipe \
  --job-id clip-a \
  --dry-run
```

`--swapper-model` 选择图。权重不会被下载，见 `scripts/download_models.zh-CN.md`。不传 `--pixel-boost`（或传 0）表示用模型的原生尺寸（inswapper 仍遵守 `swap.resolution`）。在 `inswapper_128` 上使用 `--pixel-boost 256` 或 `512` 会按更大的裁剪对齐，再跑 128 像素的切块，视频同样生效。`--eye-mode blend`（默认）在羽化蒙版下把目标眼睛混回去。`swap` 保留换过的眼睛。`target` 保留原来的眼睛。`--with-enhancer` 会追加 `face_enhancer`；眼部使用 `enhancer.eye_blend`（默认 1）。某一帧没有检测到脸不会中止视频。静态图像仍然会失败。脸映射是 JSON 列表，每项为 `{source, target_index?, reference?}`，优先于 `--many-faces` 和 `--reference`。

```bash
barrett-faceswap jobs submit \
  --source path/to/authorized-source.png \
  --target path/to/authorized-clip.mp4 \
  --output outputs/clip.mp4 \
  --dry-run \
  --job-id clip-a
barrett-faceswap jobs list
barrett-faceswap jobs resume clip-a
```

`jobs submit` 只写入规格。`jobs resume` 和 `swap --job-id ID --resume` 从 `next_frame` 继续。磁盘模式复用已经抽出的帧。管道模式按时间戳定位，不是精确到帧。`jobs run` 启动最早的一条 pending、running 或 interrupted 任务。

不需要权重的基准测试：

```bash
barrett-faceswap benchmark --dry-run --frames 32
barrett-faceswap benchmark --live --dry-run --frames 8
```

这一行会给出每秒帧数、`realtime_fps` 和各阶段毫秒数。PATH 里没有 `nvidia-smi` 时，`vram_mb` 是 `unavailable`。只有配置的权重都在时才去掉 `--dry-run`；没有人脸的合成帧仍然会让检测失败。`--live` 使用合成帧，不会打开摄像头。

不需要显示器、也不需要权重的实时试跑：

```bash
barrett-faceswap live \
  --source examples/synthetic/source.png \
  --frames 8 \
  --no-preview \
  --dry-run
```

`--virtual-camera /dev/video2` 在该 v4l2loopback 设备存在时写出原始帧。设备缺失时记一条日志，会话继续。`swap` 或 `live` 上的 `--commercial` 会拒绝内置权重：

```bash
barrett-faceswap swap \
  --source examples/synthetic/source.png \
  --target examples/synthetic/target.png \
  --output outputs/dry-run.png \
  --dry-run \
  --commercial
```

这条命令会以错误退出。默认配置里 `execution.commercial` 为 false。

训练脚手架（不下载，也不进入训练循环）：

```bash
python scripts/train_swapper.py --dataset configs/dataset.example.yaml --check
python scripts/export_onnx.py --check
python scripts/eval_swapper.py --self-test
```

文档站，不会部署：

```bash
pip install -e ".[docs]"
python scripts/build_cli_reference.py --check
mkdocs build --strict
```

接近生产的命令：

```bash
barrett-faceswap swap \
  --source path/to/authorized-source.png \
  --target path/to/authorized-target.png \
  --output outputs/swap.png \
  --enable-compliance \
  --consent-file path/to/consent.json
```

`--lang zh-CN` 或 `BARRETT_FACESWAP_LANG=zh-CN` 会切换命令行、日志和错误文本。配置里的 `ui.language` 是兜底。优先级是 `--lang`，然后是环境变量，然后是 `ui.language`，最后是英语。

其他命令：`barrett-faceswap inspect-models`、`barrett-faceswap list-processors`、`barrett-faceswap benchmark`、`barrett-faceswap ui`。

## 目录

```text
configs/default.yaml          由 pydantic 加载的默认配置
docs/ARCHITECTURE.zh-CN.md    设计和与参考项目的差别
docs/COMPLIANCE.zh-CN.md      同意、水印、过滤、审计、使用边界
docs/RESEARCH.zh-CN.md        对 FaceFusion、Deep-Live-Cam、roop、InsightFace、SimSwap 的简短记录
docs/en/ docs/zh-CN/          MkDocs Material 站点，两种语言
scripts/download_models.zh-CN.md
scripts/train_swapper.py      清单检查和训练计划；不训练
mkdocs.yml                    文档构建，CI 检查，不部署
src/barrett_faceswap/         见英文 README 的结构树
```

## 参考项目

流水线的形状参考了公开项目。没有复制它们的代码。许可证各不相同，其中几个不能把源文件放进 MIT 仓库。链接和说明见 [docs/RESEARCH.zh-CN.md](docs/RESEARCH.zh-CN.md)。

- [FaceFusion](https://github.com/facefusion/facefusion)（OpenRAIL-AS）：处理器链、会话缓存、执行提供者
- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)（AGPL-3.0）：一张源脸覆盖一串帧
- [roop](https://github.com/s0md3v/roop)（AGPL-3.0）：很短的 source / target / output 命令
- [InsightFace](https://github.com/deepinsight/insightface)（库代码 MIT，权重非商业）：SCRFD、ArcFace、inswapper
- [SimSwap](https://github.com/neuralchen/SimSwap)（CC BY-NC 4.0）：把换脸网络看成可替换后端

## 许可证

本仓库代码为 [MIT](LICENSE)。

InsightFace 预训练模型，包括 buffalo 包和 `inswapper_128.onnx`，仅限非商业研究。InsightFace 在 2025-11-24 的说明中要求，inswapper 系列的商业使用先联系 `contact@insightface.ai`。SimSwap 的代码和权重是 CC BY-NC 4.0，这里没有收录。FaceFusion 是 OpenRAIL-AS。Deep-Live-Cam 和 roop 是 AGPL-3.0。这些代码库都没有被打包进本仓库。
