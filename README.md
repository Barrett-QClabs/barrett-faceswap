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

- SCRFD-style ONNX detection (`det_10g`), ArcFace embedding (`w600k_r50`), and an inswapper-style swap (`inswapper_128`)
- Five-point alignment templates and paste-back
- Masks merged with a per-pixel minimum: inset box, xseg-style occlusion ONNX, and BiSeNet face-parsing regions. A missing mask file keeps the ellipse or all-ones stub
- `face_enhancer`: GFPGAN 1.4 or CodeFormer ONNX, with a blend amount. It is registered and left out of the default processor list so a swap does not require those files
- Video: ffprobe, frame extraction, a thread pool through the same processors, `libx264` or `h264_nvenc`, then the original audio muxed back. Optional trim
- ONNX Runtime sessions cached by model hash, provider list, and device. CUDA is preferred when you ask for it. TensorRT can be named; engine-cache tuning is later
- Model registry with license, commercial-use flag, alignment name, input size, and an optional sha256 pin checked on load
- Pydantic config shared by the CLI, the video pipeline, and the UI
- English and Chinese catalogs (`--lang`, `ui.language`, `BARRETT_FACESWAP_LANG`). The Gradio dropdown switches every label
- Opt-in compliance package: consent, watermark, pluggable content filter, append-only audit log. On a video it runs once up front, samples frames for the filter, watermarks every output frame when enforcement is on, and writes one audit line

Later, without promising a date: a webcam loop, TensorRT engine cache, pixel boost, and more swap backends.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The dependency is the headless OpenCV wheel (`opencv-python-headless`, imported as `cv2`) so a server or CI job does not need a display. `onnxruntime` is the default CPU build.

On an NVIDIA machine, install the GPU wheel **instead of** the CPU wheel:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

Do not keep both installed. An RTX 5090 is Blackwell (`sm_120`) and needs an `onnxruntime-gpu` build linked against CUDA 12.8 or newer. The CPU wheel is what CI runs. TensorRT engine-cache tuning is still later.

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
  --enhancer-blend 0.8
```

`--enhancer-model codeformer` selects the other graph. `--enhancer-weight` is the CodeFormer fidelity input.

Gradio:

```bash
barrett-faceswap ui
```

The page has source, target, output, processors, execution provider, enhancer, video encoder, and preview. The language dropdown (`zh-CN` / `en`) rewrites every label from the catalogs, including the buttons. A video preview processes the first frame only. The encoder dropdown is applied on Run when the target is a video. A still-image Run ignores it. Compliance stays off unless the checkbox or the config turns it on.

Benchmark, without weights:

```bash
barrett-faceswap benchmark --dry-run --frames 32
```

The line reports frames per second. `vram_mb` is `unavailable` when `nvidia-smi` is not on PATH. Omit `--dry-run` only when the configured weights are present; a synthetic frame with no face still fails the detector.

Production-like command:

```bash
barrett-faceswap swap \
  --source path/to/authorized-source.png \
  --target path/to/authorized-target.png \
  --output outputs/swap.png \
  --enable-compliance \
  --consent-file path/to/consent.json
```

`--lang zh-CN` or `BARRETT_FACESWAP_LANG=zh-CN` switches CLI text, logs, and errors. `ui.language` in the config is the fallback. Precedence is `--lang`, then the environment variable, then `ui.language`, then English.

Other commands: `barrett-faceswap inspect-models`, `barrett-faceswap list-processors`, `barrett-faceswap benchmark`, `barrett-faceswap ui`.

## Project structure

```text
configs/default.yaml          pydantic-loaded defaults
docs/ARCHITECTURE.md          design, and how it differs from the projects we studied
docs/COMPLIANCE.md            consent, watermark, filter, audit, acceptable use
docs/RESEARCH.md              short notes on FaceFusion, Deep-Live-Cam, roop, InsightFace, SimSwap
scripts/download_models.md    where to obtain weights; nothing here downloads them
src/barrett_faceswap/
  cli.py pipeline.py          CLI and the only supported run path
  detect.py align.py swap.py blend.py masks.py
  model_registry.py models.py execution.py
  config.py                   pydantic schema
  i18n/locales/               en.yaml and zh_CN.yaml
  processors/                 face_swapper and face_enhancer
  media/                      ffprobe, frame extract, encode
  compliance/                 policy, consent, watermark, filter, audit
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
