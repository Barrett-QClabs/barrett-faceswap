# 架构

文件换脸走 `FaceSwapPipeline`。视频走 `VideoPipeline`。摄像头走 `LivePipeline`：采集、处理和输出各一条线程，并与 Gradio 摄像头页共用 `LiveSession`。Gradio 页面默认是简体中文，深色主题，长视频有进度条。打开时选中均衡。快速、均衡、高质都保持 HyperSwap、关闭增强、遮挡模型 xseg_1、像素提升 0。摄像头页使用同一套换脸器、像素提升、遮挡和增强设置。`compliance.enabled` 保持 false。

处理器只返回帧，不写输出文件。`CompliancePolicy` 同时包住文件路径和实时路径。`compliance.enabled` 默认为 false，因此干跑不会要求同意、不会加水印，也不会写审计。

执行选项会退回 CPU：

- `execution.precision: fp16` 选用同目录的 `*_fp16.onnx`，没有该文件时用 fp32。
- `execution.cuda_graph` 请求 CUDA 图捕获。带有 `CUDAExecutionProvider` 的会话都会复用 IO binding。CPU 仍走 `session.run`。
- TensorRT 的缓存目录和构建优化级别只在安装了该提供者时挂上。`tensorrt.fp16` 默认是 false，因为 fp16 会改变身份。
- RTX 50 系列使用 `[gpu]` 附加项 `onnxruntime-gpu[cuda,cudnn]`（CUDA 13 和 cuDNN 9）。如果存在 `onnxruntime.preload_dlls()` 就会调用。安装 `[cpu]` 或 `[gpu]`，不要两个都装。`barrett-faceswap doctor` 报告提供者，并在两个轮子都在时警告。CPU 轮子不会因此报错。

`benchmark` 和 `benchmark --live` 打印各阶段延迟和 `realtime_fps`。`--source` 和 `--target` 会在这两张静态图上计时检测，报告使用目标图的真实尺寸。不传时，用绘制的人脸提供关键点。实时基准不会打开摄像头。缓存的 ONNX 会话每次加载只哈希一次。xseg 可以是通道在最后。不对称的 ConvTranspose 填充会在加载时改写。人脸几乎不动时，视频蒙版按 `masks.stride`（默认 3）隔帧运行。`--face-swapper-model` 选择 inswapper、HyperSwap、SimSwap、BlendSwap 或 Ghost。默认是 `hyperswap_1a_256`。增强器不在视频的处理器列表里。静态图需要修复时用 `--with-enhancer` 打开。`--face-enhancer-model` 选择 GFPGAN、CodeFormer、GPEN 或 RestoreFormer++。有解析文件时，增强只混合皮肤类别。CodeFormer 权重 0.7 在输入类型为 `tensor(double)` 时以 float64 送入。`--occluder-model` 选择 xseg_1（默认）、xseg_2、xseg_3，或本地的 `sam2_refine` ONNX。CUDA 是第一个提供者，CPU 是退路。`--execution-provider tensorrt` 会建立引擎缓存，`tensorrt.fp16` 默认是 false。两条视频编码路径都带 `-movflags +faststart`，因此 MP4 的 moov 原子留在文件开头，浏览器不用等下载完成就能开始播放。检测按比例把画面放进方形 SCRFD 输入，再用这个缩放把框和关键点映回原图。存在 `2dfan4.onnx` 时会细化 68 个关键点。侧脸和低分人脸会替换五个对齐点。表情贴回保留闭上的目标眼睛和抿起的嘴。像素提升使用交错的整脸子采样。`compare` 写出一张联系表。权重由操作者下载。`commercial_use` 保持 false。`compliance.enabled` 保持 false。没有 68 个点时，`--eye-mode blend` 仍然运行。

换脸器表、像素提升、选脸、任务和蒙版见 `docs/ARCHITECTURE.zh-CN.md`。
