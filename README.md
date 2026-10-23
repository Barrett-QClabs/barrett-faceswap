[English](README.md) | [中文](README.zh-CN.md)

# Barrett FaceSwap

Barrett FaceSwap swaps a face in a still image or a video. It wires the usual engineering path: detect a face, align it, run an ONNX identity swap, optionally restore the crop, and paste it back. It does not train a foundation model, and it does not ship model weights.

The code in this repository is MIT. The InsightFace weights it knows how to run, including `inswapper_128.onnx`, are not. Those weights are published for non-commercial research, and commercial use needs a grant from InsightFace. Read [docs/COMPLIANCE.md](docs/COMPLIANCE.md) before you point the tool at a real person.

## Compliance default

Enforcement ships **off**.

`configs/default.yaml` sets `compliance.enabled: false`, so a local dry-run does not ask for a consent file, does not watermark the picture, and does not block on the content filter. That is a development default. Any run that uses a real likeness should turn enforcement on:

- set `compliance.enabled: true`, or
- pass `--enable-compliance`

With enforcement on, the pipeline requires a consent record, writes an `AI-generated` / `人工智能生成` label plus file metadata, runs the content filter, and appends an audit line. The module is fully implemented either way. Tests force it on. Production should too.

## Acceptable use

Use a face only when that person has authorized the specific use. Do not create non-consensual swaps, sexual deepfakes, sexual content involving minors, or material meant to defraud or impersonate someone. Label synthetic output when you share it. The compliance gate is there to make those rules executable. It is not legal advice, and a stub filter is not a substitute for consent.

## Features

- SCRFD-style ONNX detection (`det_10g`), ArcFace embedding (`w600k_r50`), and selectable swappers via `--face-swapper-model`: `hyperswap_1a_256` (default), `hyperswap_1b/1c_256`, `inswapper_128`, `simswap_256`, `simswap_unofficial_512`, `blendswap_256`, and `ghost_1/2/3_256`. Each graph uses its own template, mean, and standard deviation. SimSwap and Ghost still need a crossface ONNX. HyperSwap uses the normalized ArcFace vector directly. BlendSwap takes a 112 px source crop. Every registry row stays `commercial_use: false`. The project is for non-commercial use. Weights stay user-downloaded
- Pixel boost: a larger aligned crop is split into interleaved full-face subsamples of the native size, inferred, and stitched (128 / 256 / 512 / 768 / 1024 when that size divides cleanly for the chosen model). A 512 crop on a 256 network is four views of the whole face, not four quarters
- Face selection: one face by index, every face (`--many-faces`), a reference face by ArcFace cosine similarity, or a JSON source-to-target map. Video keeps that choice with a small IoU tracker
- Paste-back is alpha by default. `--blend-mode poisson` uses `cv2.seamlessClone`. Color transfer defaults to `lab` (mean and standard deviation in Lab). `mean_std`, `histogram`, and `none` remain available. `--color-transfer none` turns the match off
- Jobs persist a spec and the next unfinished frame under `jobs/<id>/` so an interrupted video can resume
- Video defaults to disk frames. `--video-mode pipe` decodes and encodes through ffmpeg pipes and does not write a PNG per frame
- Masks merged with a per-pixel minimum: inset box, xseg occlusion (`--occluder-model xseg_1`, `xseg_2`, `xseg_3`, or `sam2_refine`), and BiSeNet face-parsing regions. Video and live recompute the ONNX masks every `masks.stride` frames (default 3) and warp the previous mask while the face barely moves. A missing occlusion file stays an all-ones mask. A missing parser uses a feathered landmark convex hull on the swap crop, and an ellipse when no landmarks are passed. `sam2_refine` is an operator-supplied ONNX and is not downloaded
- `face_enhancer` via `--face-enhancer-model`: GFPGAN 1.4 (the model default), CodeFormer (fidelity default 0.7), GPEN BFR 512/1024/2048, or RestoreFormer++. When a parser file is present, the restoration is blended only on the skin class. Eyes and brows stay on the plain swap. Video leaves the enhancer off. Pass `--with-enhancer`, or add `face_enhancer` to `processors`, when a still needs restoration. CodeFormer weight 0.7 with a parser keeps eyes and brows. GFPGAN narrows eyes and oversmooths skin, so it is not the video default. A CodeFormer input typed `tensor(double)` is fed as float64
- `barrett-faceswap compare` writes a labeled contact sheet and a timing table for a grid of swapper and enhancer pairs. `--dry-run` does not load weights
- Live landmark smoothing uses the same One-Euro filter as video, and it runs before alignment. Video already smooths inside the face stream before the swapper aligns
- Video: ffprobe, frame extraction, a thread pool through the same processors, `libx264` or `h264_nvenc`, then the original audio muxed back. Optional trim
- ONNX Runtime sessions cached by model hash, provider list, device, precision, and the CUDA-graph flag. The default providers are CUDA, then CPU. CUDA stays first when both CUDA and TensorRT are listed. `--execution-provider tensorrt` builds an engine cache under `.cache/tensorrt/engines`, with a separate prefix per role. `tensorrt.fp16` defaults to false because fp16 changes identity. `execution.precision: fp16` selects a sibling `*_fp16.onnx` and falls back to fp32 when the file is absent. A CUDA session reuses IO binding. CUDA graph capture is requested only when `execution.cuda_graph` is on
- Model registry with license, commercial-use flag, alignment name, input size, and an optional sha256 pin checked on load
- Pydantic config shared by the CLI, the video pipeline, and the UI
- English and Chinese catalogs (`--lang`, `ui.language`, `BARRETT_FACESWAP_LANG`). The Gradio header toggle switches every label. The shipped UI language is zh-CN
- Opt-in compliance package: consent, watermark, pluggable content filter, append-only audit log. On a video it runs once up front, samples frames for the filter, watermarks every output frame when enforcement is on, and writes one audit line. A live session uses the same hooks: consent on the first frame, a watermark on each output frame, and one audit line per session
- Realtime webcam: `barrett-faceswap live` runs capture, detect, swap, and output on three threads. Bounded queues drop stale frames. Detection runs every few frames and the cache interpolates the rest. The source embedding is computed once. Output can open a preview and, on Linux, a v4l2loopback device. The Gradio page has a webcam tab that uses the same session object
- `--commercial` refuses every registry row with `commercial_use: false`. The built-in rows stay false. `scripts/train_swapper.py` only checks a local `commercial-ok` manifest and prints a plan. It does not train or download
- `barrett-faceswap download-models` fetches `buffalo_l` and `inswapper_128`, checks sha256, resumes, and tries the next mirror. Each mirror connect waits at most 15 seconds. `--from-dir` copies matching ONNX files before any download. `--with-gfpgan`, `--with-masks`, and `--with-landmarks` are optional. The quality set is `download-models --swapper hyperswap_1a_256 --with-masks --with-landmarks`. The mask files are facefusion-assets `xseg_1.onnx` and `bisenet_resnet_34.onnx`; their licenses are unverified and likely non-commercial. `2dfan4.onnx` is the MIT FAN landmark weight. CI does not run the download. `barrett-faceswap doctor` prints providers, CUDA/cuDNN preload, and local model hashes, and warns when both ONNX Runtime wheels are installed
- 2DFAN4 (`landmarks.refine: auto`) runs when `models/2dfan4.onnx` is present. Profile and low-score faces replace the five alignment points. Frontal faces keep the detector points. The 68-point set is stored for expression. A missing file keeps the detector landmarks. `commercial_use` stays false
- Expression paste (`expression.enabled`, default on) uses an EAR/MAR gate. Closed eyes keep more of the target. Open eyes stay on the swap (eye base 0.05). A pursed mouth is mixed in mildly (up to 0.55). An open mouth stays on the swap. When 68 points exist this replaces `--eye-mode`. Without them, `--eye-mode blend` stays
- Video does not stop on a frame with no face. A miss can rotate or upscale, still at `detect.score_threshold`, and the frame stays unchanged when nothing clears that score. The last box is not pasted onto empty frames. `video.max_face_gap` (default 10) is how long the identity tracker remembers a lost face. `--eye-mode blend` (the default) keeps most of the target eyes. `--with-enhancer` adds GFPGAN or CodeFormer with a stronger eye blend. `--pixel-boost 256` or `512` tiles inswapper for a still or a video
- Bilingual docs site under `docs/en` and `docs/zh-CN` (MkDocs Material). CI builds it and does not deploy

Still later: uniface, hififace, and frame-exact pipe resume. `compliance.enabled` stays false until an official launch.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[cpu,dev]"
```

The dependency is the headless OpenCV wheel (`opencv-python-headless`, imported as `cv2`) so a server or CI job does not need a display. The base install does not pin an ONNX Runtime wheel. The `[cpu]` extra installs `onnxruntime`. CI uses that extra.

On an NVIDIA machine, including an RTX 5090 (Blackwell, `sm_120`) on Ubuntu 24.04 with driver 580, install the GPU extra **instead of** the CPU wheel:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install -e ".[gpu]"
```

`[gpu]` is `onnxruntime-gpu[cuda,cudnn]`. onnxruntime-gpu 1.31 needs CUDA 13 and cuDNN 9 (`libcublasLt.so.13`, `libcudnn.so.9`). Those libraries come from the NVIDIA pip packages. At provider setup the process calls `onnxruntime.preload_dlls()` when that function exists and prepends `site-packages/nvidia/cu13/lib` and `site-packages/nvidia/cudnn/lib` onto `LD_LIBRARY_PATH`. You do not have to export that path yourself. Do not keep the CPU wheel and the GPU wheel installed together. `barrett-faceswap doctor` warns when both are present. CI installs the `[cpu]` extra. TensorRT and CUDA graph options are skipped when those providers are not installed.

```bash
barrett-faceswap doctor
```

The Gradio UI is optional:

```bash
pip install -e ".[ui]"
```

## Quickstart

From the repository root, with enforcement left off:

```bash
python examples/make_synthetic.py
barrett-faceswap swap \
  --source examples/synthetic/source.png \
  --target examples/synthetic/target.png \
  --output outputs/dry-run.png \
  --dry-run
```

`python -m barrett_faceswap swap ...` is the same entry point. The synthetic images are colored cards, not faces. `--dry-run` skips ONNX. A real swap needs the weights in `scripts/download_models.md` and a photograph you are allowed to use.

Video uses the same `swap` command. The target suffix selects the frame pipeline. `libx264` is the default encoder. `--video-encoder h264_nvenc` selects NVENC and does not fall back if the driver cannot encode.

```bash
barrett-faceswap swap \
  --source examples/synthetic/source.png \
  --target path/to/authorized-clip.mp4 \
  --output outputs/clip.mp4 \
  --dry-run \
  --trim-start 1.0 \
  --trim-end 4.0
```

Add face restoration when `models/gfpgan_1.4.onnx` is on disk:

```bash
barrett-faceswap swap \
  --source path/to/authorized-source.png \
  --target path/to/authorized-target.png \
  --output outputs/swap.png \
  --processors face_swapper,face_enhancer \
  --enhancer-model gfpgan_1.4 \
  --enhancer-blend 0.6
```

`--enhancer-model codeformer` selects the other graph. `--enhancer-weight` is the CodeFormer fidelity input.

Gradio:

```bash
barrett-faceswap ui
```

The page is a dark app (`#0B0D10` background, `#14181F` cards, `#2563EB` primary button). The shipped language is zh-CN. The header toggle switches to English and rewrites every label, including both tabs. The main path is source face, target, swapper, and a quality preset (Fast, Balanced, High, Custom), then Preview only and Start swap. Balanced is selected on open. Fast and High use that same stack and leave the enhancer off. The result preview is the large pane on the right, and it moves above the controls on a narrow screen. A sticky action row keeps the buttons reachable on a phone. Balanced uses `hyperswap_1a_256`, enhancer off, occluder `xseg_1`, and pixel boost 0. Advanced stays collapsed: detect score threshold, face selection, enhancer, mask and occluder, video, execution, compliance, and output path. A long video fills a progress bar. The camera tab uses the same swapper, pixel boost, occluder, enhancer, and execution provider. `barrett-faceswap ui --lang en` starts in English. A video preview processes the first frame only. The encoder and the video-mode controls apply on Start swap when the target is a video. A still-image run ignores them. Compliance stays off unless the checkbox or the config turns it on. The camera tab does not open a device until the browser grants the camera. Start arms the session. The first frame runs the consent check. Stop writes the audit line when compliance is on.

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

`--swapper-model` selects the graph. Weights are not downloaded; see `scripts/download_models.md`. `--pixel-boost 0` is omitted here and means the model's native size (inswapper still honors `swap.resolution`). `--pixel-boost 256` or `512` on `inswapper_128` aligns a larger crop and runs 128-pixel tiles, including on a video. `--with-enhancer` appends `face_enhancer`. The face blend defaults to `0.6`. `enhancer.eye_blend` defaults to `0.35`, and open eyes scale it down further so the restoration does not redraw the lids. `enhancer.temporal` (default `0.4`) mixes the previous enhanced crop when the face barely moves. `--eye-mode blend` (default) keeps most of an open target eye (about 0.91). `swap` keeps the inswapper eyes, which often look narrower. `target` keeps the original eyes. Five-point landmarks are treated as open. A 106-point set measures how open each eye cluster is. A loaded parser uses eye and brow classes instead of the ellipses. A frame with no detection does not abort a video. A still image still fails. A face map is a JSON list of `{source, target_index?, reference?}` and wins over `--many-faces` and `--reference`. A loaded ONNX session is hashed once; later frames reuse it and do not re-read the file.

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

`jobs submit` only writes the spec. `jobs resume` and `swap --job-id ID --resume` continue from `next_frame`. Disk mode reuses extracted frames. Pipe mode seeks by timestamp, which is not frame-exact. `jobs run` starts the oldest pending, running, or interrupted job.

Benchmark, without weights:

```bash
barrett-faceswap benchmark --dry-run --frames 32
barrett-faceswap benchmark --live --dry-run --frames 8
barrett-faceswap benchmark \
  --source path/to/authorized-source.png \
  --target path/to/authorized-target.png \
  --frames 32
```

The line reports frames per second, `realtime_fps`, and per-stage milliseconds. `vram_mb` is `unavailable` when `nvidia-smi` is not on PATH. Omit `--dry-run` only when the configured weights are present. Without `--source` and `--target`, a drawn face and its landmarks are used so the swap still runs. Pass both stills to time detection on a real face. The reported size is that target's real size, not `--width` and `--height`. `--live` does not open a webcam.

Realtime, without a display and without weights:

```bash
barrett-faceswap live \
  --source examples/synthetic/source.png \
  --frames 8 \
  --no-preview \
  --dry-run
```

`--virtual-camera /dev/video2` writes raw frames when that v4l2loopback device exists. A missing device is logged and the session continues. `--commercial` on `swap` or `live` refuses the built-in weights:

```bash
barrett-faceswap swap \
  --source examples/synthetic/source.png \
  --target examples/synthetic/target.png \
  --output outputs/dry-run.png \
  --dry-run \
  --commercial
```

That command exits with an error. The default config leaves `execution.commercial` false.

Training scaffold (no download, no training loop):

```bash
python scripts/train_swapper.py --dataset configs/dataset.example.yaml --check
python scripts/export_onnx.py --check
python scripts/eval_swapper.py --self-test
```

Docs site, not deployed:

```bash
pip install -e ".[docs]"
python scripts/build_cli_reference.py --check
mkdocs build --strict
```

Production-like command:

```bash
barrett-faceswap swap \
  --source path/to/authorized-source.png \
  --target path/to/authorized-target.png \
  --output outputs/swap.png \
  --enable-compliance \
  --consent-file path/to/consent.json
```

`--lang zh-CN` or `BARRETT_FACESWAP_LANG=zh-CN` switches CLI text, logs, and errors. `ui.language` in the shipped config is `zh-CN`. Precedence is `--lang`, then the environment variable, then `ui.language`, then English.

Other commands: `barrett-faceswap inspect-models`, `barrett-faceswap list-processors`, `barrett-faceswap benchmark`, `barrett-faceswap ui`.

## Project structure

```text
configs/default.yaml          pydantic-loaded defaults
docs/ARCHITECTURE.md          design, and how it differs from the projects we studied
docs/COMPLIANCE.md            consent, watermark, filter, audit, acceptable use
docs/RESEARCH.md              short notes on FaceFusion, Deep-Live-Cam, roop, InsightFace, SimSwap
docs/en/ docs/zh-CN/          MkDocs Material site, both languages
scripts/download_models.md    where to obtain weights; nothing here downloads them
scripts/train_swapper.py      manifest check and a training plan; does not train
mkdocs.yml                    docs build, checked in CI, not deployed
src/barrett_faceswap/
  cli.py pipeline.py live.py  CLI, file pipeline, and the webcam threads
  queues.py face_cache.py binding.py commercial.py glyphs.py
  detect.py align.py swap.py blend.py masks.py
  swap_catalog.py pixel_boost.py facesel.py jobs.py
  model_registry.py models.py execution.py
  config.py                   pydantic schema
  i18n/locales/               en.yaml and zh_CN.yaml
  processors/                 face_swapper and face_enhancer
  media/                      ffprobe, frame extract, encode, raw pipes
  compliance/                 policy, consent, watermark, filter, audit
  train/                      dataset check, export contract, SSIM and cosine
  ui/                         Gradio app; captions via t()
```

## Inspiration

The shape of this pipeline follows public projects. Their code was not copied. Licenses differ, and several of them are incompatible with dropping files into an MIT tree. Details and links: [docs/RESEARCH.md](docs/RESEARCH.md).

- [FaceFusion](https://github.com/facefusion/facefusion) (OpenRAIL-AS) for the processor chain, session cache, and execution providers
- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) (AGPL-3.0) for the idea of one source face over a stream of frames
- [roop](https://github.com/s0md3v/roop) (AGPL-3.0) for a short source / target / output command
- [InsightFace](https://github.com/deepinsight/insightface) (library MIT, weights non-commercial) for SCRFD, ArcFace, and inswapper
- [SimSwap](https://github.com/neuralchen/SimSwap) (CC BY-NC 4.0) for treating the swap network as a replaceable backend

## License

This repository's code is [MIT](LICENSE).

InsightFace pretrained models, including the buffalo packs and `inswapper_128.onnx`, are available for non-commercial research only. InsightFace's 2025-11-24 note asks operators to contact `contact@insightface.ai` before commercial use of the inswapper series. SimSwap's code and weights are CC BY-NC 4.0 and are not included here. FaceFusion is OpenRAIL-AS. Deep-Live-Cam and roop are AGPL-3.0. None of those codebases are vendored.
