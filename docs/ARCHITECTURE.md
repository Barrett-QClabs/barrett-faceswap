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

Disk mode (`video.mode: disk`, the default) writes frames to a temporary directory and deletes them after encode. When a job id is set, those frames stay under `jobs/<id>/{raw,processed}` so a resume can reuse them. `video.threads` defaults to 4. Dry-run workers share that pool. A real run keeps face recovery on one thread and overlaps PNG reads and writes. Per-frame score notes are suppressed (`log_every_frame` is false) so a long clip does not append one line per frame. A summary note records how many frames were processed, plus detected, retried, filled, and passed-through counts. Many-faces, reference, map, and any job force a single worker so the tracker and `next_frame` stay in order.

Detection letterboxes the frame into the square network input. The aspect ratio stays, the spare side is black, and the box and five landmarks map back by that scale and top-left pad. The frame is not stretched. A real video does not abort when detection misses. The first miss retries a lower threshold, then rotations (including ±90°), then a 1.5× upscale. Neighbors inside `video.max_face_gap` (default 10) are interpolated, or the nearer side is carried forward. Beyond that gap the frame is left unchanged. `video.smooth` is `one_euro` (default), `ema`, or `none`, and it resets after a pass-through. A still image still raises `NoFaceError`. Dry-run does not detect. Pipe mode overlaps ffmpeg decode and encode on queues (`-thread_queue_size 32` on the decoder). The swapper keeps one detector and one swap backend for the job, so the inswapper matrix is read once. Paste warps the crop's bounding box rather than the whole frame. Lab and mean/std color match use a masked mean and deviation.

`h264_nvenc` does not fall back to `libx264` when the driver cannot encode. Audio is kept when `video.keep_audio` is true. A full file tries stream copy and retries AAC. A trim always re-encodes audio as AAC. `--no-audio` drops the track. Both encode paths pass `-movflags +faststart` before the output path, so the moov atom stays at the front of an MP4 and a browser can start playback before the file finishes downloading.

Filter sampling uses `video.filter_every_n_frames` (default 30) and always includes the last frame. The watermark, when enforcement is on, is applied to every output frame, not only the sample. `apply_video_metadata` sets a container comment and copies the file with `-movflags +faststart`, so that rewrite does not push the moov atom back to the end. If that ffmpeg call fails, the run still succeeds: the frames are already marked and the sidecar is written. Enforcement off skips the filter, the watermark, and the audit line. `before_process` returns immediately in that case and does not require sample frames.

Pipe mode (`--video-mode pipe`) decodes raw BGR24 on ffmpeg stdout, runs the same processors, and encodes raw BGR24 on ffmpeg stdin. It does not write a PNG per frame. `h264_nvenc` still does not fall back. Dimensions are rounded down to even values for yuv420p. Compliance still calls `before_process` once: the sample set is estimated from duration times fps, plus the last decoded frame, collected by a decode pass that does not start an encoder. Output samples are screened before the watermark. Every output frame is watermarked when enforcement is on. One audit line is written by `complete_output`. A failure aborts the pipes and deletes the partial output. Resume seeks by adding `next_frame / fps` to the trim start. That seek is a timestamp, not a frame-exact index.

## Swappers

`swap_catalog.py` names the ONNX contract for each selectable model. `--swapper-model` and the UI dropdown set `swap.model`. The default is `hyperswap_1a_256`. `inswapper_128` remains available and its role is still `swapper`, so an inswapper run's `ensure_present()` calls keep working. The other graphs are optional roles. A run opens the detector, the recognizer, the selected swapper, and that swapper's crossface file when it has one. `CORE_ROLES` is unchanged, so missing optional weights do not block an inswapper swap. A hyperswap run needs `models/hyperswap_1a_256.onnx`, which is not part of the default buffalo download.

| Name | Native | Template | Normalize | Latent | Crossface |
| --- | --- | --- | --- | --- | --- |
| `inswapper_128` | 128 | inswapper (ArcFace plus the 8 px shift at 128, scaled with size) | 0..1 RGB | `embedding @` the 512×512 matrix. Prefer an initializer named `emap`; otherwise the last initializer (`buff2fs` in the official file) | none |
| `hyperswap_1a/1b/1c_256` | 256 | `arcface_128` (inswapper layout at the crop size) | mean 0.5, std 0.5, then denormalize the output | L2-normalized ArcFace | none |
| `simswap_256` | 256 | ArcFace 112 v1 scaled by the crop side | ImageNet mean and std | crossface, then L2 | `crossface_simswap` |
| `simswap_unofficial_512` | 512 | ArcFace 112 v1 scaled by the crop side | 0..1 RGB | crossface, then L2 | `crossface_simswap` |
| `blendswap_256` | 256 | FFHQ target; ArcFace 112 v2 source at 112 | 0..1 RGB on both images | the source crop, not an embedding | none |
| `ghost_1/2/3_256` | 256 | ArcFace 112 v1 scaled by the crop side | mean 0.5, std 0.5, then denormalize the output | crossface, raw (not L2 again) | `crossface_ghost` |

Image and latent inputs are bound by rank (4-D image, 2-D latent), not by a hard-coded name. BlendSwap has two 4-D inputs. Named `source` and `target` win; otherwise the smaller spatial input is the source. Output decoding matches the enhancer after an optional denormalize: [-1, 1] when any value is below `-0.01`, 0..255 when the max is above `1.5`, otherwise 0..1. A crossface file is a 512-d to 512-d ONNX supplied by the operator. This tree does not vendor one and does not copy FaceFusion, SimSwap, or GHOST source. Every new record sets `commercial_use` to false. Sources and the contracts above are in [scripts/download_models.md](../scripts/download_models.md). `barrett-faceswap compare` writes a labeled contact sheet and a `.timing.txt` table. `--dry-run` does not open weights.

## Pixel boost

`pixel_boost` aligns the crop at the boosted side, then takes interleaved full-face subsamples `crop[row::count, col::count]`. Each tile is inferred with the same latent and written back on that lattice. The tiles are not quadrants. Allowed sides are 128, 256, 512, 768, and 1024 when the side is at least the native size and divides evenly. `0` means native size. Inswapper with boost `0` still honors `swap.resolution`, so a test can point a mock graph at a size that is not a tile multiple. A side that is not a multiple of native is one inference. Video uses the same swapper, so `--pixel-boost 256` or `512` tiles there too. A 512 crop on a 256 network is four interleaved views of the whole face. `256` on inswapper (native 128) is still four 128-pixel tiles.

`--eye-mode` chooses the eyes after inswapper. The 128 crop often draws the source eyelids, so a plain swap looks narrow or half-closed.

| `--eye-mode` | What is kept |
| --- | --- |
| `swap` | The swapped crop only. |
| `blend` (default) | The target crop inside the eye mask. Open eyes raise that weight from 0.65 to about 0.91. A 106-point set measures openness from the vertical spread of each eye cluster. Five-point landmarks have no lids, so they are treated as open. |
| `target` | The original target eyes. |

Five-point landmarks become two ellipses. A 106-point set uses the convex hull of the upper-face clusters, split left and right. When `models/bisenet_face_parsing.onnx` is loaded, the left and right eye and brow classes replace that geometric mask on the swap. `blend.eye_sharpen` (default 0.6) is an unsharp mask limited to that region. `--with-enhancer` appends `face_enhancer` when it is not already listed. `enhancer.eye_blend` (default 0.35) is scaled down again while the eyes are open, so the restoration does not redraw the lids. `enhancer.blend` (default 0.6) covers the rest of the face.

## Landmarks and expression

`landmarks.refine` is `auto` (default), `always`, or `off`. `auto` and `always` open `models/2dfan4.onnx` when that file exists. The crop scales the face box so its longer side is 195 pixels inside a 256 canvas. The network returns 68 points. Five alignment points are the means of indices 36:42 and 42:48, point 30, and points 48 and 54. `auto` replaces the detector's five points only when the face is a profile (`profile_ratio`, default 0.22) or the detector score is below `min_score` (default 0.65). Frontal faces keep the detector points. `always` replaces every face. `off` does not open the file. A missing file, or a heatmap score below 0.05, keeps the detector landmarks. The 68 points stay on `Face.landmarks_68` and are not appended to the five-point array. `commercial_use` stays false. `--with-landmarks` downloads the file. The pin is in `scripts/download_models.md`.

`expression.enabled` (default true) pastes target eyes and a pursed mouth from those 68 points. The eye amount is `0.05 + 0.85 * closedness` and the mouth amount is `0.55 * pursedness`, using `ear_open` 0.24, `ear_closed` 0.15, `mar_open` 0.045, and `mar_closed` 0.012. Hulls are the eye-and-brow rings and the mouth ring, drawn in the aligned crop. When this path runs, `blend.eye_mode` is skipped, because that blend would put most of the target eye back on an open five-point face. Without 68 points, `blend.eye_mode` still runs. The enhancer sees `1 - closedness` as `eye_openness`. Video still leaves `face_enhancer` off unless `--with-enhancer` or the processor list adds it.

`sunglasses.enabled` (default true) is the path when the target crop is wearing sunglasses. A parser glasses bar, or a dark bar across both eyes that is clearly darker than the forehead and the cheeks, turns it on. A flat dark crop does not. Stage 1 widens the paste to a fuller head (`box_inset` 0.04) and leaves expression off, including `--eye-mode`, so landmarks under the lenses are not used. A mid-face hole in the mask is filled by that identity coverage. It is not inpainted. Stage 2, when `sunglasses.expression` is on and 68 points exist, mixes only the mouth, scaled by `mouth_scale` (default 0.45). The enhancer is told the eyes are open so it does not treat the lenses as closed lids. Stills, video, and live all go through this swapper path.

## Source library

`library.dir` (default `library/`, gitignored) stores named profiles. Each sample is a video, a photo, or an audio clip of the same person. Face vectors go through `select_identity`, the same held-out average used for a source video. A video sample stores that clip's fused vector. A photo stores one embedding. Audio does not change the face. Trimmed speech is concatenated in sample order and clipped to the newest `library.voice_seconds` (default 24) at 16 kHz. The preview still is the highest-quality sample. A swap uses `still.png`, the fused embedding, and `voice.wav` when it exists. One-shot upload stays available. `barrett-faceswap library` creates, extends, lists, and deletes profiles. `swap --library NAME` and `live --library NAME` replace `--source`. Pass exactly one of the two.

## Face selection

`swap.selector` is `index` (default, largest face first, `--target-face-index`), `many` (every detected face, one source), `reference` (cosine similarity against a reference image, or against the source face when no reference is passed), or `map`. The map file is a JSON list of `{source, target_index, reference}`. A locked `target_index` is assigned directly, in largest-first order. The remaining faces are matched one-to-one above `swap.similarity` (default 0.45). No match raises `swap.no_match`.

`FaceTracker` is greedy IoU (`swap.track_iou`, default 0.3, drop after 8 missed frames). Once a track receives a source slot, later frames keep that slot. Tracking is on by default for reference and map.

## Blend

`blend.mode` is `alpha` (default inverse-warp composite) or `poisson` (`cv2.seamlessClone`, normal clone). Poisson clears a one-pixel border so the mask does not touch the frame edge, and falls back to alpha when the mask is tiny or OpenCV refuses the clone. `blend.color` defaults to `lab` (mean and standard deviation in Lab on the swapped region). `mean_std`, `histogram`, and `none` remain available. `--color-transfer none` turns the match off.

## Jobs

`jobs/<id>/job.json` stores the request, the state (`pending`, `running`, `interrupted`, `done`), and `next_frame` (the first unfinished index). `barrett-faceswap jobs submit` queues a pending job and does not run it. `jobs resume ID`, `jobs run`, and `swap --job-id ID --resume` continue. A hard stop can leave the state at `running`; resume accepts `pending`, `running`, and `interrupted`. A finished id is refused. Disk resume skips processed frames whose index is below `next_frame` and reuses `raw/frame_*.png` when those files are already there. The watermark pass visits every processed frame and skips a frame that already has a sibling `.wm` stamp, so a resume does not draw the bar twice. A frame that was written and not yet stamped is marked once. Pipe resume is the timestamp seek described above. An exception before `done` marks the job `interrupted`. Image jobs set `next_frame` to 1.

## Face enhancer

`face_enhancer` is registered and left out of the default processor list, so a swap does not require those weights. Select it with `--processors face_swapper,face_enhancer`, `--with-enhancer`, or the UI checkbox. `--face-enhancer-model` (alias `--enhancer-model`) is `gfpgan_1.4` (default), `codeformer`, `gpen_bfr_512`, `gpen_bfr_1024`, `gpen_bfr_2048`, or `restoreformer_plus_plus`. `--enhancer-blend` mixes the restoration with the original crop (`1` keeps the restoration, default `0.6`). `--enhancer-weight` is the CodeFormer fidelity scalar (default `0.7`). GFPGAN graphs have no side input, so the weight is unused there. When `models/bisenet_face_parsing.onnx` is loaded, the blend uses the skin class only and leaves eye and brow pixels on the unenhanced swap. A missing parser keeps the eye-ellipse blend.

The crop is aligned with the FFHQ 5-point template scaled by `size / 512`, then pasted back with the ellipse region mask. The blend uses `enhancer.blend` on the face and `enhancer.eye_blend` (default `0.35`) inside the eye mask built from that same template. A lower eye amount keeps the target eyes, including the lid, instead of letting the restoration replace them. `enhancer.temporal` (default `0.4`) mixes that blended crop with the previous frame's enhanced crop when the mean five-point shift divided by face width is at most `enhancer.temporal_motion` (default `0.04`). `temporal: 0` turns the mix off. A larger motion drops the previous crop, and `post_process` clears it at the end of a run. No detected face leaves the frame unchanged and records one note. When the caller already supplied `known_faces`, the enhancer uses those faces.

The ONNX contract is local code, not a copy of the GFPGAN or CodeFormer training projects:

- Input is NCHW RGB float32, normalized with `(pixel / 255 - 0.5) / 0.5` (range [-1, 1]).
- A non-4-D input, when the graph has one, receives the CodeFormer weight. An input typed `tensor(double)` is cast to float64. Float inputs stay float32.
- Output is decoded as [-1, 1] when any value is below `-0.01`, as 0..255 when the max is above `1.5`, and as 0..1 otherwise.
- Spatial size follows the session. The registry fallback is 512.

Both records set `commercial_use=False`. GFPGAN's published code and v1.4 weights are Apache-2.0, and the network incorporates StyleGAN2 under the NVIDIA Source Code License, so the registry does not mark the file as commercially cleared. CodeFormer weights are S-Lab License 1.0 (non-commercial research). This tree does not vendor those files. `download-models --with-gfpgan` can fetch a published GFPGAN ONNX when you ask; CI does not. Sources and the `.pth` export notes are in [scripts/download_models.md](../scripts/download_models.md).

## Masks

`masks.py` builds an inset box, a region, and an occlusion stub of ones, then takes the element-wise minimum (`merge_min`). `masking.py` replaces a stub when the matching ONNX file is on disk and the flag is on (`masks.occlusion`, `masks.parser`). A missing occlusion file stays all ones. A missing parser uses a feathered convex hull of the expanded five-point landmarks when the swapper passes the crop template, and the ellipse when no landmarks are passed. The enhancer still uses the ellipse for its own paste. Eye and brow classes from the parser, when that file is loaded, are the swap eye mask.

Video and live (`masks.stride`, default 3) run the occlusion and parsing graphs every third frame and warp the last crop mask onto the current alignment while landmark motion over face width stays at or below `masks.stride_motion` (default 0.04). A larger move recomputes immediately. `stride: 1` runs the graphs on every frame. A still image always does. When both graphs are loaded they run on two threads. At load, ConvTranspose nodes whose padding is asymmetric are rewritten to zero padding plus a slice, which is the same result and is eligible for the CUDA provider. The facefusion xseg graph is channels-last (`[N, 256, 256, 3]`); that layout is transposed in and is not fed as NCHW. `--occluder-model` is `xseg_1` (default), `xseg_2`, `xseg_3`, or `sam2_refine`. The last one is an operator-supplied ONNX at `models/sam2_refine.onnx`. It takes the 0..1 RGB crop and the coarse xseg mask and returns a mask. It is not downloaded.

Occlusion (`models/occlusion_xseg.onnx`, registry size 256) takes 0..1 RGB, or one gray channel when that is the static input. A channels-last static shape is accepted. Higher values mean the pixel stays visible. A max above `1.5` is treated as 0..255. `masks.occlusion_invert` flips graphs that predict the occluder instead of the face. `download-models --with-masks` fetches facefusion-assets `xseg_1.onnx` (models-3.1.0) and stores it at this path. Its license is unverified and likely non-commercial. The historical xseg idea comes from DeepFaceLab. That project's code is GPL-3.0 and is not in this tree. `commercial_use` is false.

Parsing (`models/bisenet_face_parsing.onnx`) uses ImageNet mean and standard deviation. Labels follow the 19 CelebAMask-HQ indexes in `parsing.py` (0 background through 18 hat). The default region set is skin, brows, eyes, eyeglasses, nose, mouth, and lips. Hair, cloth, hat, neck, and ears stay out. The graph may return class scores or an integer label map. `--with-masks` also fetches facefusion-assets `bisenet_resnet_34.onnx` (models-3.0.0). That file's license is unverified and likely non-commercial. Common checkpoints are trained on CelebAMask-HQ, which is non-commercial research, so `commercial_use` is false.

## UI and benchmark

`barrett-faceswap ui` builds a Gradio Blocks app. Gradio is imported only inside `build_demo` / `launch_ui` and is the `[ui]` extra. The page is dark (`#0B0D10` / `#14181F` / `#2563EB`) with two tabs: image and video, and camera. The shipped language is zh-CN. The header toggle calls `t()` and rewrites every caption, including both tab buttons. The main path is source, target, swapper, and a quality preset (Fast, Balanced, High, Custom). Balanced is selected on open. Fast, Balanced, and High are HyperSwap, enhancer off, occluder `xseg_1`, and pixel boost 0. None of them turns GFPGAN or CodeFormer on. Advanced stays collapsed and still contains the detect score threshold, face selection, enhancer, mask, video, execution, compliance, and the output path. A long video reports frame progress. The camera tab reads the same model settings and shares `LiveSession` with the CLI. Preview on a video extracts the first frame and runs the still pipeline. Start swap writes the output path. The encoder and video-mode controls are sent only when the target is a video and the click is Start swap. The compliance checkbox cannot turn a config that is already on back off. The shipped checkbox follows the config, which is false. Start arms the session, the first camera frame runs consent, and Stop writes one audit line when enforcement is on. The three threads live in `barrett-faceswap live`, not in the browser callback.

`barrett-faceswap benchmark` times the processor chain and prints per-stage milliseconds plus `realtime_fps`. `--source` and `--target` together run detection on those stills. The reported size is the target image's real size. `--width` and `--height` apply only to the drawn fallback. Without the stills, a drawn face supplies landmarks so the swap is timed even when the detector would reject the drawing. `--live` times a session (capture, detect, swap, output) and does not open a camera. `--dry-run` does not load ONNX. VRAM is read from `nvidia-smi` when that command works, and reported as unavailable otherwise.

## Live

`barrett-faceswap live` runs capture, process, and output on three threads. Each hop is a `LatestQueue`. `put_latest` drops the oldest frame when the queue is full (`live.queue_size`, default 1). Detection runs every `live.detect_every` frames (default 3). Between detections `DetectionCache` steps the box and the landmarks by the last matched velocity. After that cache step, a One-Euro filter (`video.smooth`, the same setting as the file path) smooths landmarks before the processors align. The first sample is unchanged. Video already smooths inside `FaceStream` before alignment, so the live filter is not applied there a second time. `prepare_source` embeds the source once. A preview window opens when `DISPLAY` is set. `--no-preview` skips it. `--virtual-camera /dev/video2` writes raw BGR to a v4l2loopback device through ffmpeg. A missing device is logged and the preview continues.

Compliance wraps this path. `before_process` screens the source and the first camera frame. Each output frame is watermarked in memory when enforcement is on. `complete_output` appends one audit line and does not tag a container file.

## Execution

`execution.precision` is `fp32` (the default) or `fp16`. fp16 looks beside the configured file for `{stem}_fp16.onnx` and falls back to the fp32 path when that sibling is absent. `execution.cuda_graph` sets `enable_cuda_graph` on the CUDA provider. Any session that reports `CUDAExecutionProvider` reuses one IO binding; a failure falls back to `session.run`. CPU sessions, including CI, stay on `session.run`. TensorRT options (engine cache, timing cache, builder optimization level 0–5) are attached only when `TensorrtExecutionProvider` is installed. `tensorrt.fp16` defaults to false because fp16 changes identity. `--execution-provider tensorrt` selects that provider and then CPU. The engine-cache prefix is the model role, so inswapper (`swapper`) and xseg (`occluder`) do not share one engine name. `prefer_cuda` still puts CUDA first when both CUDA and TensorRT are listed. CPU CI skips both and does not create the cache directories. The base package does not depend on an ONNX Runtime wheel. Install the `[cpu]` extra or the `[gpu]` extra, not both. `doctor` warns when both distribution names are installed.

## Commercial route

`--commercial` sets `execution.commercial`. `guard_commercial` refuses any role that run would open when the registry row is missing or `commercial_use` is false. Every built-in row is false, so the flag rejects the default graphs, including the occluder and parser while those masks stay enabled. No YAML key flips a built-in row to true. A model you train is a new registry row you add after legal review.

`src/barrett_faceswap/train/` checks a local manifest whose license string is exactly `commercial-ok`, prints the training steps, states the ONNX contract (NCHW RGB in [-1, 1], a 512-d latent, ArcFace template scaled by size/112), and can score global SSIM plus ArcFace cosine. It does not download data, does not open a training loop, and does not copy GHOST or AEI-Net source. Those projects are the Apache-2.0 route this scaffold points at. The code here is original. `scripts/train_swapper.py --check`, `scripts/export_onnx.py --check`, and `scripts/eval_swapper.py --self-test` are the entry points. Without `--check`, the training and export scripts exit 2 and do not write weights.

## Docs site

`mkdocs.yml` builds a Material site from `docs/en` and `docs/zh-CN` with `mkdocs-static-i18n` (`docs_structure: folder`). `scripts/build_cli_reference.py` writes the CLI page in both languages from `build_parser()`. CI runs that script with `--check` and `mkdocs build --strict`. Nothing is deployed. The narrative files in `docs/*.md` stay next to the site and are excluded from the build.

## What was adopted

**Processor protocol.** `Processor` has `pre_check`, `process_frame`, and `post_process`. `processors/registry.py` maps a name to a factory that receives the pydantic config and the provider list. `face_swapper` and `face_enhancer` are registered. This is the same kind of seam FaceFusion uses. The methods and the registry are local code.

**Session cache.** `ModelStore.session` keys a session by resolved path, size, mtime, the requested provider tuple, device id, CUDA graph flag, and precision. A cache hit returns that session and does not hash the file. The hash runs once when the session is first opened. `file_sha256` caches by the same path, size, and mtime, so a later doctor or inspect call does not re-read a weight that has not changed. A replaced file has a new size or mtime and is hashed and loaded again. Creation is locked. A repeated call does not rebuild the session. `prefer_cuda` moves `cuda` to the front when it is in the request and leaves `cpu` last. If the installed ONNX Runtime has no CUDA provider, that name is skipped and CPU remains. Naming `tensorrt` attaches the engine cache, the timing cache, fp16, and the builder optimization level when that provider is installed, and is skipped on a CPU wheel. An RTX 5090 (Blackwell, `sm_120`) with driver 580 uses `onnxruntime-gpu[cuda,cudnn]` (CUDA 13 and cuDNN 9). Provider setup calls `onnxruntime.preload_dlls()` when it exists and prepends the NVIDIA pip `lib` directories. A missing library does not raise. CI installs the `[cpu]` extra. `barrett-faceswap doctor` prints providers, preload status, and model hashes, and warns when both `onnxruntime` and `onnxruntime-gpu` are installed.

**Alignment and paste-back.** `align.py` holds the five-point ArcFace template and an FFHQ 512 template. Size 112 is the recognizer crop. Size 128 adds the 8 pixel horizontal shift the inswapper export expects. The enhancer scales the FFHQ template by `size / 512`. `blend.py` inverse-warps the crop onto its bounding box and composites with a mask.

**Model registry.** `model_registry.py` records the detector, recognizer, inswapper, the optional HyperSwap / SimSwap / BlendSwap / Ghost / crossface roles, GFPGAN, CodeFormer, GPEN, RestoreFormer++, the xseg occluders, `sam2_refine`, and the parser: filename, license key, `commercial_use`, alignment name, and input size. sha256 pins live in the config and default to null because this repo does not redistribute the files. Download pins for the optional packs live in `fetch.py`. `ensure_present()` defaults to the three swap roles, so enhancer, mask, and extra swapper files stay optional. `inspect-models` prints one row per role. Every built-in row stays `commercial_use: false`.

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

Swappers beyond this set (uniface, hififace, and others), and frame-exact pipe resume. Pipe resume still seeks by timestamp. The Chinese watermark covers `人工智能生成` with original bitmaps; any other character needs `compliance.watermark.font_path`. The training scaffold does not train or export a weight. TensorRT and CUDA graph capture run only when those providers are installed. `compliance.enabled` stays false until an official launch.
