[English](ARCHITECTURE.md) | [中文](ARCHITECTURE.zh-CN.md)

# Architecture

`FaceSwapPipeline.run` is the public call for a file. Phase 4 adds a realtime webcam loop, TensorRT and CUDA-graph options that stay idle on CPU, an fp16 sibling lookup, a commercial-use gate, and a training scaffold that does not train. Phase 3's selectable swappers, pixel boost, multi-face selection, Poisson blending, resumable jobs, and optional diskless pipe stay in place. A video target is handed to `VideoPipeline`. File and live paths call `CompliancePolicy` before and after the processors, including a resumed job and pipe mode. Processors return frames. They do not write the output file, so a watermark cannot be skipped by a processor that saves early.

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

`--dry-run` skips ONNX and still goes through the policy. Passing `--trim-start`, `--trim-end`, or `--video-encoder` on a still image is a config error. `video_mode: pipe` on a still image is also a config error. A job id on a still image records one frame and marks the job done.

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

Disk mode (`video.mode: disk`, the default) writes frames to a temporary directory and deletes them after encode. When a job id is set, those frames stay under `jobs/<id>/{raw,processed}` so a resume can reuse them. `video.threads` defaults to 4. Workers share ONNX sessions. Per-frame score notes are suppressed (`log_every_frame` is false) so a long clip does not append one line per frame. A summary note records how many frames were processed. Many-faces, reference, map, and any job force a single worker so the tracker and `next_frame` stay in order.

`h264_nvenc` does not fall back to `libx264` when the driver cannot encode. Audio is kept when `video.keep_audio` is true. A full file tries stream copy and retries AAC. A trim always re-encodes audio as AAC. `--no-audio` drops the track.

Filter sampling uses `video.filter_every_n_frames` (default 30) and always includes the last frame. The watermark, when enforcement is on, is applied to every output frame, not only the sample. `apply_video_metadata` sets a container comment. If that ffmpeg call fails, the run still succeeds: the frames are already marked and the sidecar is written. Enforcement off skips the filter, the watermark, and the audit line. `before_process` returns immediately in that case and does not require sample frames.

Pipe mode (`--video-mode pipe`) decodes raw BGR24 on ffmpeg stdout, runs the same processors, and encodes raw BGR24 on ffmpeg stdin. It does not write a PNG per frame. `h264_nvenc` still does not fall back. Dimensions are rounded down to even values for yuv420p. Compliance still calls `before_process` once: the sample set is estimated from duration times fps, plus the last decoded frame, collected by a decode pass that does not start an encoder. Output samples are screened before the watermark. Every output frame is watermarked when enforcement is on. One audit line is written by `complete_output`. A failure aborts the pipes and deletes the partial output. Resume seeks by adding `next_frame / fps` to the trim start. That seek is a timestamp, not a frame-exact index.

## Swappers

`swap_catalog.py` names the ONNX contract for each selectable model. `--swapper-model` and the UI dropdown set `swap.model`. The default remains `inswapper_128`, whose role is still `swapper` so existing `ensure_present()` calls keep working. The other graphs are optional roles. A run opens the detector, the recognizer, the selected swapper, and that swapper's crossface file when it has one. `CORE_ROLES` is unchanged, so missing optional weights do not block an inswapper swap.

| Name | Native | Template | Normalize | Latent | Crossface |
| --- | --- | --- | --- | --- | --- |
| `inswapper_128` | 128 | inswapper (ArcFace plus the 8 px shift at 128, scaled with size) | 0..1 RGB | `embedding @` the 512×512 matrix. Prefer an initializer named `emap`; otherwise the last initializer (`buff2fs` in the official file) | none |
| `hyperswap_1a/1b/1c_256` | 256 | ArcFace scaled by size/112 | [-1, 1] | crossface | `crossface_hyperswap` |
| `simswap_256` | 256 | ArcFace scaled by size/112 | [-1, 1] | crossface | `crossface_simswap` |
| `simswap_unofficial_512` | 512 | ArcFace scaled by size/112 | [-1, 1] | crossface | `crossface_simswap` |
| `ghost_1/2/3_256` | 256 | ArcFace scaled by size/112 | [-1, 1] | crossface | `crossface_ghost` |

Image and latent inputs are bound by rank (4-D image, 2-D latent), not by a hard-coded name. Output decoding matches the enhancer: [-1, 1] when any value is below `-0.01`, 0..255 when the max is above `1.5`, otherwise 0..1. A crossface file is a 512-d to 512-d ONNX supplied by the operator. This tree does not vendor one and does not copy FaceFusion, SimSwap, or GHOST source. Every new record sets `commercial_use` to false. Sources and the contracts above are in [scripts/download_models.md](../scripts/download_models.md).

## Pixel boost

`pixel_boost` aligns the crop at the boosted side, splits it into contiguous native-size squares, runs the same latent on each tile, and stitches the tiles. Allowed sides are 128, 256, 512, 768, and 1024 when the side is at least the native size and divides evenly. `0` means native size. Inswapper with boost `0` still honors `swap.resolution`, so a test can point a mock graph at a size that is not a tile multiple. A side that is not a multiple of native is one inference.

## Face selection

`swap.selector` is `index` (default, largest face first, `--target-face-index`), `many` (every detected face, one source), `reference` (cosine similarity against a reference image, or against the source face when no reference is passed), or `map`. The map file is a JSON list of `{source, target_index, reference}`. A locked `target_index` is assigned directly, in largest-first order. The remaining faces are matched one-to-one above `swap.similarity` (default 0.45). No match raises `swap.no_match`.

`FaceTracker` is greedy IoU (`swap.track_iou`, default 0.3, drop after 8 missed frames). Once a track receives a source slot, later frames keep that slot. Tracking is on by default for reference and map.

## Blend

`blend.mode` is `alpha` (default inverse-warp composite) or `poisson` (`cv2.seamlessClone`, normal clone). Poisson clears a one-pixel border so the mask does not touch the frame edge, and falls back to alpha when the mask is tiny or OpenCV refuses the clone. `blend.color` defaults to `lab` (mean and standard deviation in Lab on the swapped region). `mean_std`, `histogram`, and `none` remain available. `--color-transfer none` turns the match off.

## Jobs

`jobs/<id>/job.json` stores the request, the state (`pending`, `running`, `interrupted`, `done`), and `next_frame` (the first unfinished index). `barrett-faceswap jobs submit` queues a pending job and does not run it. `jobs resume ID`, `jobs run`, and `swap --job-id ID --resume` continue. A hard stop can leave the state at `running`; resume accepts `pending`, `running`, and `interrupted`. A finished id is refused. Disk resume skips processed frames whose index is below `next_frame` and reuses `raw/frame_*.png` when those files are already there. The watermark pass visits every processed frame and skips a frame that already has a sibling `.wm` stamp, so a resume does not draw the bar twice. A frame that was written and not yet stamped is marked once. Pipe resume is the timestamp seek described above. An exception before `done` marks the job `interrupted`. Image jobs set `next_frame` to 1.

## Face enhancer

`face_enhancer` is registered and left out of the default processor list, so a swap does not require those weights. Select it with `--processors face_swapper,face_enhancer` or the UI checkbox. `--enhancer-model` is `gfpgan_1.4` (default) or `codeformer`. `--enhancer-blend` mixes the restoration with the original crop (`1` keeps the restoration, default `0.8`). `--enhancer-weight` is the CodeFormer fidelity scalar (default `0.5`). GFPGAN graphs have no side input, so the weight is unused there.

The crop is aligned with the FFHQ 5-point template scaled by `size / 512`, then pasted back with the ellipse region mask. No detected face leaves the frame unchanged and records one note.

The ONNX contract is local code, not a copy of the GFPGAN or CodeFormer training projects:

- Input is NCHW RGB float32, normalized with `(pixel / 255 - 0.5) / 0.5` (range [-1, 1]).
- A non-4-D input, when the graph has one, receives the CodeFormer weight.
- Output is decoded as [-1, 1] when any value is below `-0.01`, as 0..255 when the max is above `1.5`, and as 0..1 otherwise.
- Spatial size follows the session. The registry fallback is 512.

Both records set `commercial_use=False`. GFPGAN's published code and v1.4 weights are Apache-2.0, and the network incorporates StyleGAN2 under the NVIDIA Source Code License, so the registry does not mark the file as commercially cleared. CodeFormer weights are S-Lab License 1.0 (non-commercial research). This tree does not vendor those files. `download-models --with-gfpgan` can fetch a published GFPGAN ONNX when you ask; CI does not. Sources and the `.pth` export notes are in [scripts/download_models.md](../scripts/download_models.md).

## Masks

`masks.py` builds an inset box, a region, and an occlusion stub of ones, then takes the element-wise minimum (`merge_min`). `masking.py` replaces a stub when the matching ONNX file is on disk and the flag is on (`masks.occlusion`, `masks.regions_enabled`). A missing occlusion file stays all ones. A missing parser uses a feathered convex hull of the expanded five-point landmarks when the swapper passes the crop template, and the ellipse when no landmarks are passed. The enhancer still uses the ellipse.

Occlusion (`models/occlusion_xseg.onnx`, registry size 256) takes 0..1 RGB, or one gray channel when that is the static input. Higher values mean the pixel stays visible. A max above `1.5` is treated as 0..255. `masks.occlusion_invert` flips graphs that predict the occluder instead of the face. The historical xseg idea comes from DeepFaceLab. That project's code is GPL-3.0 and is not in this tree. `commercial_use` is false because the weight is operator-supplied and this repository grants no rights to it.

Parsing (`models/bisenet_face_parsing.onnx`) uses ImageNet mean and standard deviation. Labels follow the 19 CelebAMask-HQ indexes in `parsing.py` (0 background through 18 hat). The default region set is skin, brows, eyes, eyeglasses, nose, mouth, and lips. Hair, cloth, hat, neck, and ears stay out. The graph may return class scores or an integer label map. Common checkpoints are trained on CelebAMask-HQ, which is non-commercial research, so `commercial_use` is false.

## UI and benchmark

`barrett-faceswap ui` builds a Gradio Blocks app. Gradio is imported only inside `build_demo` / `launch_ui` and is the `[ui]` extra. The language dropdown calls `t()` and rewrites every caption, including the Preview and Run buttons, the Phase 3 controls, the commercial checkbox, and the webcam tab. Preview on a video extracts the first frame and runs the still pipeline. Run writes the output path. The encoder and video-mode dropdowns are sent only when the target is a video and the click is Run. The compliance checkbox cannot turn a config that is already on back off. The shipped checkbox follows the config, which is false. The webcam tab shares `LiveSession` with the CLI. Start arms the session, the first camera frame runs consent, and Stop writes one audit line when enforcement is on. The three threads live in `barrett-faceswap live`, not in the browser callback.

`barrett-faceswap benchmark` times the processor chain and prints per-stage milliseconds plus `realtime_fps`. `--live` times a synthetic webcam session (capture, detect, swap, output) and does not open a camera. `--dry-run` does not load ONNX. VRAM is read from `nvidia-smi` when that command works, and reported as unavailable otherwise.

## Live

`barrett-faceswap live` runs capture, process, and output on three threads. Each hop is a `LatestQueue`. `put_latest` drops the oldest frame when the queue is full (`live.queue_size`, default 1). Detection runs every `live.detect_every` frames (default 3). Between detections `DetectionCache` steps the box and the landmarks by the last matched velocity. `prepare_source` embeds the source once. A preview window opens when `DISPLAY` is set. `--no-preview` skips it. `--virtual-camera /dev/video2` writes raw BGR to a v4l2loopback device through ffmpeg. A missing device is logged and the preview continues.

Compliance wraps this path. `before_process` screens the source and the first camera frame. Each output frame is watermarked in memory when enforcement is on. `complete_output` appends one audit line and does not tag a container file.

## Execution

`execution.precision` is `fp32` (the default) or `fp16`. fp16 looks beside the configured file for `{stem}_fp16.onnx` and falls back to the fp32 path when that sibling is absent. `execution.cuda_graph` sets `enable_cuda_graph` on the CUDA provider and feeds the swapper through IO binding only when the live session reports `CUDAExecutionProvider`. Otherwise `session.run` is used. TensorRT options (engine cache, timing cache, fp16, builder optimization level 0–5) are attached only when `TensorrtExecutionProvider` is installed. CPU CI skips both and does not create the cache directories.

## Commercial route

`--commercial` sets `execution.commercial`. `guard_commercial` refuses any role that run would open when the registry row is missing or `commercial_use` is false. Every built-in row is false, so the flag rejects the default graphs, including the occluder and parser while those masks stay enabled. No YAML key flips a built-in row to true. A model you train is a new registry row you add after legal review.

`src/barrett_faceswap/train/` checks a local manifest whose license string is exactly `commercial-ok`, prints the training steps, states the ONNX contract (NCHW RGB in [-1, 1], a 512-d latent, ArcFace template scaled by size/112), and can score global SSIM plus ArcFace cosine. It does not download data, does not open a training loop, and does not copy GHOST or AEI-Net source. Those projects are the Apache-2.0 route this scaffold points at. The code here is original. `scripts/train_swapper.py --check`, `scripts/export_onnx.py --check`, and `scripts/eval_swapper.py --self-test` are the entry points. Without `--check`, the training and export scripts exit 2 and do not write weights.

## Docs site

`mkdocs.yml` builds a Material site from `docs/en` and `docs/zh-CN` with `mkdocs-static-i18n` (`docs_structure: folder`). `scripts/build_cli_reference.py` writes the CLI page in both languages from `build_parser()`. CI runs that script with `--check` and `mkdocs build --strict`. Nothing is deployed. The narrative files in `docs/*.md` stay next to the site and are excluded from the build.

## What was adopted

**Processor protocol.** `Processor` has `pre_check`, `process_frame`, and `post_process`. `processors/registry.py` maps a name to a factory that receives the pydantic config and the provider list. `face_swapper` and `face_enhancer` are registered. This is the same kind of seam FaceFusion uses. The methods and the registry are local code.

**Session cache.** `ModelStore.session` keys a session by resolved path, sha256, the requested provider tuple, and the device id. Creation is locked. A repeated call does not rebuild the session. `prefer_cuda` moves `cuda` to the front when it is in the request and leaves `cpu` last. If the installed ONNX Runtime has no CUDA provider, that name is skipped and CPU remains. Naming `tensorrt` attaches the engine cache, the timing cache, fp16, and the builder optimization level when that provider is installed, and is skipped on a CPU wheel. An RTX 5090 (Blackwell, `sm_120`) with driver 580 uses `onnxruntime-gpu[cuda,cudnn]` (CUDA 13 and cuDNN 9). Provider setup calls `onnxruntime.preload_dlls()` when it exists and prepends the NVIDIA pip `lib` directories. A missing library does not raise. CI runs the CPU wheel. `barrett-faceswap doctor` prints providers, preload status, and model hashes.

**Alignment and paste-back.** `align.py` holds the five-point ArcFace template and an FFHQ 512 template. Size 112 is the recognizer crop. Size 128 adds the 8 pixel horizontal shift the inswapper export expects. The enhancer scales the FFHQ template by `size / 512`. `blend.py` inverse-warps the crop and composites with a mask.

**Model registry.** `model_registry.py` records the detector, recognizer, inswapper, the optional HyperSwap / SimSwap / Ghost / crossface roles, GFPGAN, CodeFormer, occluder, and parser: filename, license key, `commercial_use`, alignment name, and input size. sha256 pins live in the config and default to null because this repo does not redistribute the files. `ensure_present()` defaults to the three swap roles, so enhancer, mask, and extra swapper files stay optional. `inspect-models` prints one row per role.

**Config.** `config.py` is a pydantic model. The CLI and the UI load that same model from YAML. Relative paths resolve from the working directory.

**i18n.** `i18n/locales/en.yaml` and `zh_CN.yaml` are the only user-facing catalogs. `t(key)` is used for CLI help, logs, errors, compliance text, and every Gradio caption. Language order is `--lang`, then `BARRETT_FACESWAP_LANG`, then `ui.language`, then `en`. Quote YAML keys that PyYAML 1.1 would read as booleans (`yes`, `no`).

**Compliance.** See [COMPLIANCE.md](COMPLIANCE.md). The master switch defaults to off. The policy object still runs, including around a video, a resumed job, and pipe mode.

## How this tree differs

- MIT code, no vendored FaceFusion, Deep-Live-Cam, roop, or SimSwap sources
- `download-models` is an explicit command. The installer, the UI, and CI do not download weights
- Compliance is a real package and is off unless the operator opts in
- Messages go through catalogs in English and Chinese
- Hash pins are optional until the operator sets them
- The content filter default is a stub that allows the frame and logs that no detector is loaded
- Mask models fall back to stubs when the file is absent, so tests and CI run without weights

## Still deferred

Swappers beyond this set (blendswap, uniface, hififace, and others), and frame-exact pipe resume. Pipe resume still seeks by timestamp. The Chinese watermark covers `人工智能生成` with original bitmaps; any other character needs `compliance.watermark.font_path`. The training scaffold does not train or export a weight. TensorRT and CUDA graph capture run only when those providers are installed. `compliance.enabled` stays false until an official launch.
