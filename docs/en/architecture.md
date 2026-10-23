# Architecture

File swaps go through `FaceSwapPipeline`. Video goes through `VideoPipeline`. The webcam goes through `LivePipeline`, which runs capture, process, and output on three threads and shares `LiveSession` with the Gradio webcam tab.

Processors return frames. They do not write the output file. `CompliancePolicy` wraps the file path and the live path. `compliance.enabled` defaults to false, so a dry-run does not ask for consent, does not watermark, and does not write an audit line.

Execution options degrade to CPU:

- `execution.precision: fp16` selects a sibling `*_fp16.onnx` and falls back to the fp32 file.
- `execution.cuda_graph` requests CUDA graph capture. A session with `CUDAExecutionProvider` reuses IO binding either way. CPU stays on `session.run`.
- TensorRT cache directories, fp16, and the builder optimization level are attached only when that provider is installed.
- RTX 50-series uses the `[gpu]` extra, `onnxruntime-gpu[cuda,cudnn]` (CUDA 13 and cuDNN 9). Setup calls `onnxruntime.preload_dlls()` when it exists. Install `[cpu]` or `[gpu]`, not both. `barrett-faceswap doctor` reports providers and warns when both wheels are installed. A CPU wheel does not raise.

`benchmark` and `benchmark --live` print per-stage latency and `realtime_fps`. `--source` and `--target` time detection on those stills, and the report uses the target's real size. Without them, a drawn face supplies landmarks. The live benchmark does not open a webcam. A cached ONNX session is hashed once per load. xseg may be channels-last. Asymmetric ConvTranspose padding is rewritten at load. Video masks run every `masks.stride` frames (default 3) while the face barely moves. `--face-swapper-model` selects inswapper, HyperSwap, SimSwap, BlendSwap, or Ghost. `--face-enhancer-model` selects GFPGAN, CodeFormer, GPEN, or RestoreFormer++. A parser file limits enhancement to the skin class. `--occluder-model` selects xseg_1, xseg_2, xseg_3, or a local `sam2_refine` ONNX. `--execution-provider tensorrt` builds an fp16 engine cache. `compare` writes a contact sheet. Weights stay downloaded by the operator. `commercial_use` stays false. `compliance.enabled` stays false. `--eye-mode blend` keeps an open target eye more strongly than a closed one.

See `docs/ARCHITECTURE.md` for the swapper table, pixel boost, face selection, jobs, and masks.
