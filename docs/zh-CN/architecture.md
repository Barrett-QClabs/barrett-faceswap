# 架构

文件换脸走 `FaceSwapPipeline`。视频走 `VideoPipeline`。摄像头走 `LivePipeline`：采集、处理和输出各一条线程，并与 Gradio 摄像头页共用 `LiveSession`。

处理器只返回帧，不写输出文件。`CompliancePolicy` 同时包住文件路径和实时路径。`compliance.enabled` 默认为 false，因此干跑不会要求同意、不会加水印，也不会写审计。

执行选项会退回 CPU：

- `execution.precision: fp16` 选用同目录的 `*_fp16.onnx`，没有该文件时用 fp32。
- `execution.cuda_graph` 请求 CUDA 图捕获。带有 `CUDAExecutionProvider` 的会话都会复用 IO binding。CPU 仍走 `session.run`。
- TensorRT 的缓存目录、fp16 和构建优化级别只在安装了该提供者时挂上。
- RTX 50 系列使用 `[gpu]` 附加项 `onnxruntime-gpu[cuda,cudnn]`（CUDA 13 和 cuDNN 9）。如果存在 `onnxruntime.preload_dlls()` 就会调用。安装 `[cpu]` 或 `[gpu]`，不要两个都装。`barrett-faceswap doctor` 报告提供者，并在两个轮子都在时警告。CPU 轮子不会因此报错。

`benchmark` 和 `benchmark --live` 打印各阶段延迟和 `realtime_fps`。实时基准使用合成帧。

换脸器表、像素提升、选脸、任务和蒙版见 `docs/ARCHITECTURE.zh-CN.md`。
