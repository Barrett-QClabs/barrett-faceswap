# Architecture

File swaps go through `FaceSwapPipeline`. Video goes through `VideoPipeline`. The webcam goes through `LivePipeline`, which runs capture, process, and output on three threads and shares `LiveSession` with the Gradio webcam tab.

Processors return frames. They do not write the output file. `CompliancePolicy` wraps the file path and the live path. `compliance.enabled` defaults to false, so a dry-run does not ask for consent, does not watermark, and does not write an audit line.

Execution options degrade to CPU:

- `execution.precision: fp16` selects a sibling `*_fp16.onnx` and falls back to the fp32 file.
- `execution.cuda_graph` uses IO binding only when the session has `CUDAExecutionProvider`.
- TensorRT cache directories, fp16, and the builder optimization level are attached only when that provider is installed.
- RTX 50-series uses `onnxruntime-gpu[cuda,cudnn]` (CUDA 13 and cuDNN 9). Setup calls `onnxruntime.preload_dlls()` when it exists. `barrett-faceswap doctor` reports the result. A CPU wheel does not raise.

`benchmark` and `benchmark --live` print per-stage latency and `realtime_fps`. The live benchmark uses synthetic frames.

See `docs/ARCHITECTURE.md` for the swapper table, pixel boost, face selection, jobs, and masks.
