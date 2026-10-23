[English](ARCHITECTURE.md) | [中文](ARCHITECTURE.zh-CN.md)

# Architecture

Phase 2 keeps the Phase 1 entry point and adds a video path, a real face enhancer, occlusion and parsing masks, a Gradio UI, and a benchmark. `FaceSwapPipeline.run` is still the public call. A video target is handed to `VideoPipeline`. Both paths call `CompliancePolicy` before and after the processors. Processors return frames. They do not write the output file, so a watermark cannot be skipped by a processor that saves early.

`compliance.enabled` stays `false` in `configs/default.yaml`. The policy object still runs. With that default it logs that enforcement is off and writes an unmarked file.

Background reading for the first cut is in [RESEARCH.md](RESEARCH.md).

## Still images

```text
SwapRequest
  -> CompliancePolicy.before_process
       consent, filter permission, source/target filter   (only if enforcement is on)
  -> processors in order
       face_swapper: detect -> align -> embed -> swap -> masks -> paste
       face_enhancer: GFPGAN 1.4 or CodeFormer, then blend (optional)
  -> CompliancePolicy.after_process
       output filter, watermark, audit, write the file
```

`--dry-run` skips ONNX and still goes through the policy. Passing `--trim-start`, `--trim-end`, or `--video-encoder` on a still image is a config error.

## Video

A target whose suffix is `.mp4`, `.mkv`, `.mov`, `.webm`, `.avi`, or `.m4v` takes the video path. The source must be a still. The output suffix must be `.mp4`, `.mkv`, or `.mov`.

```text
ffprobe
  -> ffmpeg extract PNG frames (optional -ss / -to trim, applied before -i)
  -> CompliancePolicy.before_process once
       sampled target frames when enforcement is on
  -> ThreadPoolExecutor
       same processor chain as a still, one frame at a time
  -> screen sampled output frames
  -> watermark every output frame when enforcement and the watermark are on
  -> ffmpeg encode (libx264 default, or h264_nvenc)
  -> restore the original audio
  -> container comment + .meta.json sidecar, one audit line
```

Frames are written to a temporary directory and deleted after encode. A diskless pipe is not implemented. `video.threads` defaults to 4. Workers share ONNX sessions. Per-frame score notes are suppressed (`log_every_frame` is false) so a long clip does not append one line per frame. A summary note records how many frames were processed.

`h264_nvenc` does not fall back to `libx264` when the driver cannot encode. Audio is kept when `video.keep_audio` is true. A full file tries stream copy and retries AAC. A trim always re-encodes audio as AAC. `--no-audio` drops the track.

Filter sampling uses `video.filter_every_n_frames` (default 30) and always includes the last frame. The watermark, when enforcement is on, is applied to every output frame, not only the sample. `apply_video_metadata` sets a container comment. If that ffmpeg call fails, the run still succeeds: the frames are already marked and the sidecar is written. Enforcement off skips the filter, the watermark, and the audit line. `before_process` returns immediately in that case and does not require sample frames.

## Face enhancer

`face_enhancer` is registered and left out of the default processor list, so a swap does not require those weights. Select it with `--processors face_swapper,face_enhancer` or the UI checkbox. `--enhancer-model` is `gfpgan_1.4` (default) or `codeformer`. `--enhancer-blend` mixes the restoration with the original crop (`1` keeps the restoration, default `0.8`). `--enhancer-weight` is the CodeFormer fidelity scalar (default `0.5`). GFPGAN graphs have no side input, so the weight is unused there.

The crop is aligned with the FFHQ 5-point template scaled by `size / 512`, then pasted back with the ellipse region mask. No detected face leaves the frame unchanged and records one note.

The ONNX contract is local code, not a copy of the GFPGAN or CodeFormer training projects:

- Input is NCHW RGB float32, normalized with `(pixel / 255 - 0.5) / 0.5` (range [-1, 1]).
- A non-4-D input, when the graph has one, receives the CodeFormer weight.
- Output is decoded as [-1, 1] when any value is below `-0.01`, as 0..255 when the max is above `1.5`, and as 0..1 otherwise.
- Spatial size follows the session. The registry fallback is 512.

Both records set `commercial_use=False`. GFPGAN's published code and v1.4 weights are Apache-2.0, and the network incorporates StyleGAN2 under the NVIDIA Source Code License, so the registry does not mark the file as commercially cleared. CodeFormer weights are S-Lab License 1.0 (non-commercial research). Nothing in this tree downloads or vendors those files. Sources are in [scripts/download_models.md](../scripts/download_models.md).

## Masks

`masks.py` still builds an inset box, an ellipse stub, and an occlusion stub of ones, then takes the element-wise minimum (`merge_min`). `masking.py` replaces a stub when the matching ONNX file is on disk and the flag is on (`masks.occlusion`, `masks.regions_enabled`). A missing file keeps the stub and logs the path.

Occlusion (`models/occlusion_xseg.onnx`, registry size 256) takes 0..1 RGB, or one gray channel when that is the static input. Higher values mean the pixel stays visible. A max above `1.5` is treated as 0..255. `masks.occlusion_invert` flips graphs that predict the occluder instead of the face. The historical xseg idea comes from DeepFaceLab. That project's code is GPL-3.0 and is not in this tree. `commercial_use` is false because the weight is operator-supplied and this repository grants no rights to it.

Parsing (`models/bisenet_face_parsing.onnx`) uses ImageNet mean and standard deviation. Labels follow the 19 CelebAMask-HQ indexes in `parsing.py` (0 background through 18 hat). The default region set is skin, brows, eyes, eyeglasses, nose, mouth, and lips. Hair, cloth, hat, neck, and ears stay out. The graph may return class scores or an integer label map. Common checkpoints are trained on CelebAMask-HQ, which is non-commercial research, so `commercial_use` is false.

## UI and benchmark

`barrett-faceswap ui` builds a Gradio Blocks app. Gradio is imported only inside `build_demo` / `launch_ui` and is the `[ui]` extra. The language dropdown calls `t()` and rewrites every caption, including the Preview and Run buttons. Preview on a video extracts the first frame and runs the still pipeline. Run writes the output path. The encoder dropdown is sent only when the target is a video and the click is Run. The compliance checkbox cannot turn a config that is already on back off. The shipped checkbox follows the config, which is false.

`barrett-faceswap benchmark` times the processor chain. `--dry-run` does not load ONNX. Frames-per-second is printed. VRAM is read from `nvidia-smi` when that command works, and reported as unavailable otherwise.

## What was adopted

**Processor protocol.** `Processor` has `pre_check`, `process_frame`, and `post_process`. `processors/registry.py` maps a name to a factory that receives the pydantic config and the provider list. `face_swapper` and `face_enhancer` are registered. This is the same kind of seam FaceFusion uses. The methods and the registry are local code.

**Session cache.** `ModelStore.session` keys a session by resolved path, sha256, the requested provider tuple, and the device id. Creation is locked. A repeated call does not rebuild the session. `prefer_cuda` moves `cuda` to the front when it is in the request and leaves `cpu` last. If the installed ONNX Runtime has no CUDA provider, that name is skipped and CPU remains. Naming `tensorrt` is allowed and logged as deferred: there is no engine cache or builder setup. An RTX 5090 (Blackwell) needs an `onnxruntime-gpu` build linked against CUDA 12.8 or newer. CI runs the CPU wheel.

**Alignment and paste-back.** `align.py` holds the five-point ArcFace template and an FFHQ 512 template. Size 112 is the recognizer crop. Size 128 adds the 8 pixel horizontal shift the inswapper export expects. The enhancer scales the FFHQ template by `size / 512`. `blend.py` inverse-warps the crop and composites with a mask.

**Model registry.** `model_registry.py` records detector, recognizer, swapper, GFPGAN, CodeFormer, occluder, and parser: filename, license key, `commercial_use`, alignment name, and input size. sha256 pins live in the config and default to null because this repo does not redistribute the files. `ensure_present()` defaults to the three swap roles, so enhancer and mask files stay optional. `inspect-models` prints the merged row.

**Config.** `config.py` is a pydantic model. The CLI and the UI load that same model from YAML. Relative paths resolve from the working directory.

**i18n.** `i18n/locales/en.yaml` and `zh_CN.yaml` are the only user-facing catalogs. `t(key)` is used for CLI help, logs, errors, compliance text, and every Gradio caption. Language order is `--lang`, then `BARRETT_FACESWAP_LANG`, then `ui.language`, then `en`. Quote YAML keys that PyYAML 1.1 would read as booleans (`yes`, `no`).

**Compliance.** See [COMPLIANCE.md](COMPLIANCE.md). The master switch defaults to off. The policy object still runs, including around a video.

## How this tree differs

- MIT code, no vendored FaceFusion, Deep-Live-Cam, roop, or SimSwap sources
- No model download in the installer, the CLI, the UI, or CI
- Compliance is a real package and is off unless the operator opts in
- Messages go through catalogs in English and Chinese
- Hash pins are optional until the operator sets them
- The content filter default is a stub that allows the frame and logs that no detector is loaded
- Mask models fall back to stubs when the file is absent, so tests and CI run without weights

## Still deferred

Realtime webcam capture, CUDA graphs, a TensorRT engine cache, pixel boost, extra swappers (hyperswap, SimSwap, ghost), multi-face mapping, Poisson blending, job resume, a diskless video pipe, automatic weight download, and a bundled font that can draw the Chinese watermark on pixels. OpenCV's Hershey font cannot draw `人工智能生成`. The visible bar uses the existing Latin label path. The container comment and the sidecar carry the Unicode string.
