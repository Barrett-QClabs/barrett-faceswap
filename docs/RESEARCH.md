[English](RESEARCH.md) | [中文](RESEARCH.zh-CN.md)

# Research note

This file records what was read while shaping Phase 1, and the license boundary around that reading. The current design, including video, the enhancer, masks, and the UI, lives in [ARCHITECTURE.md](ARCHITECTURE.md). No source from the projects below was copied into this tree.

## FaceFusion

Repository: <https://github.com/facefusion/facefusion>. License: OpenRAIL-AS.

Useful shape, described from the outside:

- A processor chain. Each stage exposes a pre-check, a per-frame call, and a post-step. `face_swapper` sits next to other stages such as `face_enhancer`.
- ONNX sessions kept in a pool keyed by the module, the model names, and the execution providers.
- A small map from friendly device names (`cuda`, `tensorrt`, `cpu`, and others) onto ONNX Runtime provider ids, with CPU left in the list.
- A Gradio layout split into source, target, processors, execution, and preview. We did not build that UI. The caption keys are already in `ui.*` so a later shell does not hard-code strings.

OpenRAIL-AS is a use-restricted license. It is not MIT. Nothing from that repository is vendored here.

## Deep-Live-Cam

Repository: <https://github.com/hacksider/Deep-Live-Cam>. License: AGPL-3.0.

The useful idea is a live loop: one source face, then a capture frame, a swap, and a display, with an enhancer as an optional second stage. AGPL-3.0 would cover derivative source if their code were copied. Phase 1 only keeps the processor hook so a later webcam consumer can call `process_frame`. There is no webcam code yet.

## roop

Repository: <https://github.com/s0md3v/roop>. License: AGPL-3.0. The project is archived.

The useful idea is the short command: one source image, one target, one output, and an optional restore step. Our CLI keeps that shape and adds an explicit compliance switch. The implementation is original.

## InsightFace

Repository: <https://github.com/deepinsight/insightface>. The library code is MIT. The pretrained models are not.

Phase 1 speaks the public tensor contract of three weights, without depending on the `insightface` package and without downloading anything:

- `det_10g.onnx` from the `buffalo_l` pack: SCRFD-10GF, 640 input, five landmarks
- `w600k_r50.onnx`: ArcFace ResNet50 trained on WebFace600K, 112 input, 512-d embedding
- `inswapper_128.onnx`: 128 crop plus that embedding. The 512×512 matrix is the last initializer, named `buff2fs` in the official file. This repository prefers a tensor named `emap` and otherwise uses the last initializer.

The model zoo states that pretrained models are for non-commercial research only, whether you download them by hand or through their library. A note dated 2025-11-24 asks people to contact `contact@insightface.ai` for inswapper-series licensing, and `recognition-oss-pack@insightface.ai` for the open recognition packs. See the upstream license section: <https://github.com/deepinsight/insightface#license>.

The swapper example still documents the file URL <https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx> and says the embedding must come from `buffalo_l`. That example also points operators at InsightFace's own product for higher resolution swaps and says the 128 demo is no longer maintained.

RetinaFace weights exist in the same model zoo. This phase decodes the SCRFD distance heads used by `det_10g`. A RetinaFace graph needs a separate adapter.

## SimSwap

Repository: <https://github.com/neuralchen/SimSwap>. License: CC BY-NC 4.0. The authors also say the project is for technical and academic use.

SimSwap showed that one trained network can swap an identity onto many targets, and that a mask is part of making the paste believable. Their code and checkpoints are non-commercial and are not included. The local `face_swapper` processor is the seam where a different backend could be added later, under that backend's own license.

## What we did not take

- Training scripts, datasets, and checkpoints
- NSFW classifiers wired in as a silent default (the stub filter allows frames and says so)
- Gradio, webcam capture, or video decode
- Any AGPL, OpenRAIL-AS, or CC BY-NC source file
