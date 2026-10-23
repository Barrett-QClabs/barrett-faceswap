[English](RESEARCH.md) | [中文](RESEARCH.zh-CN.md)

# 调研记录

这份记录说明第一阶段设计时读过哪些项目，以及许可证边界。当前设计（含视频、增强器、蒙版和界面）写在 [ARCHITECTURE.zh-CN.md](ARCHITECTURE.zh-CN.md)。下面这些项目的源代码都没有被复制进本仓库。

## FaceFusion

仓库：<https://github.com/facefusion/facefusion>。许可证：OpenRAIL-AS。

从外部看，值得借鉴的形状是：

- 处理器链。每个阶段有预检查、逐帧处理和收尾。`face_swapper` 与 `face_enhancer` 等阶段并列。
- ONNX 会话放在一个池里，键包含模块、模型名和执行提供者。
- 把 `cuda`、`tensorrt`、`cpu` 等名称映射到 ONNX Runtime 提供者，并保留 CPU。
- Gradio 界面分成源、目标、处理器、执行设备和预览。我们没有做这个界面。文案键已经放在 `ui.*`，以后的界面不必把字符串写死。

OpenRAIL-AS 是带使用限制的许可证，不是 MIT。该仓库没有任何文件被收进这里。

## Deep-Live-Cam

仓库：<https://github.com/hacksider/Deep-Live-Cam>。许可证：AGPL-3.0。

有用的想法是实时循环：一张源脸，然后采集一帧、替换、显示，增强器是可选的第二阶段。如果复制他们的代码，AGPL-3.0 会覆盖衍生源码。第一阶段只保留处理器钩子，让以后的摄像头调用 `process_frame`。现在没有摄像头代码。

## roop

仓库：<https://github.com/s0md3v/roop>。许可证：AGPL-3.0。项目已归档。

有用的想法是很短的命令：一张源图、一个目标、一个输出，外加可选的修复步骤。我们的命令行保留这个形状，并加上显式的合规开关。实现是自己写的。

## InsightFace

仓库：<https://github.com/deepinsight/insightface>。库代码是 MIT。预训练模型不是。

第一阶段直接使用三种权重的公开张量约定，不依赖 `insightface` 包，也不下载任何文件：

- `buffalo_l` 包中的 `det_10g.onnx`：SCRFD-10GF，输入 640，五个人脸点
- `w600k_r50.onnx`：在 WebFace600K 上训练的 ArcFace ResNet50，输入 112，512 维向量
- `inswapper_128.onnx`：128 的裁剪图加上该向量，图内带有 `emap` 矩阵

模型库说明：无论手动下载还是通过他们的库下载，预训练模型都只供非商业研究。2025-11-24 的说明要求，inswapper 系列的授权联系 `contact@insightface.ai`，开放识别包联系 `recognition-oss-pack@insightface.ai`。见上游许可段落：<https://github.com/deepinsight/insightface#license>。

换脸示例仍记录了文件地址 <https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx>，并写明向量必须来自 `buffalo_l`。该示例也建议更高分辨率的替换使用 InsightFace 自己的产品，并说明 128 的演示不再维护。

同一模型库里还有 RetinaFace 权重。本阶段解码的是 `det_10g` 使用的 SCRFD 距离头。RetinaFace 需要单独的适配器。

## SimSwap

仓库：<https://github.com/neuralchen/SimSwap>。许可证：CC BY-NC 4.0。作者同时说明项目用于技术和学术用途。

SimSwap 表明一个训练好的网络可以把身份换到许多目标上，蒙版是让贴回可信的一部分。他们的代码和检查点是非商业的，这里没有收录。本地的 `face_swapper` 处理器是以后接入其他后端的接缝，届时必须遵守那个后端自己的许可证。

## 我们没有带走的东西

- 训练脚本、数据集和检查点
- 把某个 NSFW 分类器静默设为默认（stub 过滤器会放行并说明自己不是检测器）
- Gradio、摄像头采集或视频解码
- 任何 AGPL、OpenRAIL-AS 或 CC BY-NC 源文件
