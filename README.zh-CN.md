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

- SCRFD 风格的 ONNX 检测（`det_10g`）、ArcFace 向量（`w600k_r50`）、inswapper 风格替换（`inswapper_128`）
- 五点对齐模板和贴回
- 蒙版按像素取最小值合并：内缩方框、xseg 风格的遮挡 ONNX、BiSeNet 人脸解析区域。蒙版文件缺失时仍使用椭圆或全 1 占位
- `face_enhancer`：GFPGAN 1.4 或 CodeFormer ONNX，带混合系数。它已注册，但默认不在处理器列表里，因此换脸不强制要求这些文件
- 视频：ffprobe、抽帧、线程池走同一套处理器、`libx264` 或 `h264_nvenc`，再把原音轨混回去。可以裁剪时间段
- ONNX Runtime 会话按模型哈希、提供者列表和设备缓存。请求里包含 CUDA 时优先 CUDA。可以写出 TensorRT，引擎缓存以后再做
- 模型注册表包含许可、商业标记、对齐名称、输入尺寸，以及加载时可选的 sha256 校验
- CLI、视频流水线和界面共用的 Pydantic 配置
- 中英文目录（`--lang`、`ui.language`、`BARRETT_FACESWAP_LANG`）。Gradio 的语言下拉会切换全部标签
- 可选合规包：同意、水印、可插拔内容过滤、只追加的审计日志。视频只在开始时走一次前置检查，按间隔抽样过滤，强制打开时给每一帧加水印，并写一行审计

以后再做、不承诺日期：摄像头循环、TensorRT 引擎缓存、pixel boost，以及更多换脸后端。

## 安装

需要 Python 3.10 或更新版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

依赖使用无界面的 OpenCV 轮子（`opencv-python-headless`，导入名仍是 `cv2`），这样服务器和 CI 不需要显示器。默认安装 CPU 版 `onnxruntime`。

在 NVIDIA 机器上，用 GPU 轮子**替换** CPU 轮子：

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

不要两个同时留着。RTX 5090 是 Blackwell（`sm_120`），需要链接 CUDA 12.8 或更新版本的 `onnxruntime-gpu`。CI 走 CPU 轮子。TensorRT 引擎缓存仍留到以后。

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

页面上有源、目标、输出、处理器、执行提供者、增强器、视频编码器和预览。语言下拉（`zh-CN` / `en`）会按目录重写全部标签，包括按钮。视频预览只处理第一帧。目标是视频时，运行会采用编码器下拉框的值。静态图像的运行会忽略它。除非勾选合规或配置里已经打开，否则合规保持关闭。

不需要权重的基准测试：

```bash
barrett-faceswap benchmark --dry-run --frames 32
```

这一行会给出每秒帧数。PATH 里没有 `nvidia-smi` 时，`vram_mb` 是 `unavailable`。只有配置的权重都在时才去掉 `--dry-run`；没有人脸的合成帧仍然会让检测失败。

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
scripts/download_models.zh-CN.md
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
