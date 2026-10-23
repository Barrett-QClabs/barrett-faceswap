[English](ARCHITECTURE.md) | [中文](ARCHITECTURE.zh-CN.md)

# Architecture

Phase 1 is an integration skeleton. The public entry point is `FaceSwapPipeline.run`. It always calls `CompliancePolicy.before_process` and `after_process`. Processors return frames. They do not write the output file, so a watermark cannot be skipped by a processor that saves early.

Background reading is in [RESEARCH.md](RESEARCH.md).

## Pipeline

```text
SwapRequest
  -> CompliancePolicy.before_process
       consent, filter permission, source/target filter   (only if enforcement is on)
  -> processors in order
       face_swapper: detect -> align -> embed -> swap -> masks -> paste
       face_enhancer: hook, refused on a real run
  -> CompliancePolicy.after_process
       output filter, watermark, audit, write the file
```

`--dry-run` skips ONNX and still goes through the policy. With the shipped config, the policy logs that enforcement is off and writes the image unmarked.

## What was adopted

**Processor protocol.** `Processor` has `pre_check`, `process_frame`, and `post_process`. `processors/registry.py` maps a name to a factory that receives the pydantic config and the provider list. `face_swapper` is registered. `face_enhancer` is registered and raises on a real run so the chain can list it without pretending GFPGAN ran. This is the same kind of seam FaceFusion uses. The methods and the registry are local code.

**Session cache.** `ModelStore.session` keys a session by resolved path, sha256, the requested provider tuple, and the device id. A repeated call does not rebuild the session. `prefer_cuda` moves `cuda` to the front when it is in the request and leaves `cpu` last. If the installed ONNX Runtime has no CUDA provider, that name is skipped and CPU remains. Naming `tensorrt` is allowed and logged as deferred: Phase 1 does not set up an engine cache or builder flags.

**Alignment and paste-back.** `align.py` holds the five-point ArcFace template. Size 112 is the recognizer crop. Size 128 adds the 8 pixel horizontal shift the inswapper export expects. `blend.py` inverse-warps the crop and composites with a mask. `masks.py` builds an inset box, an ellipse, and an occlusion stub of ones, then takes the element-wise minimum. The occlusion stub is wired so a later model can return a real map without a new paste function.

**Model registry.** `model_registry.py` records, for detector, recognizer, and swapper: filename, license key, `commercial_use=False`, alignment name, and input size. sha256 pins live in the config and default to null because this repo does not redistribute the files. On load, the file is hashed. A pin must match. An empty pin is reported so you can copy the digest into `models.<role>.sha256`. `inspect-models` prints the merged row.

**Config.** `config.py` is a pydantic model. The CLI loads that same model from YAML. Relative paths resolve from the working directory.

**i18n.** `i18n/locales/en.yaml` and `zh_CN.yaml` are the only user-facing catalogs. `t(key)` is used for CLI help, logs, errors, compliance text, and `ui.gradio_labels`. Language order is `--lang`, then `BARRETT_FACESWAP_LANG`, then `ui.language`, then `en`.

**Compliance.** See [COMPLIANCE.md](COMPLIANCE.md). The master switch defaults to off. The policy object still runs.

## ONNX contract

The detector resizes to the configured square (default 640), converts to RGB, and normalizes with `(pixel - 127.5) / 128`. Heads are grouped by channel count: 1 for score, 4 for distances, 10 for keypoints. Anchors sit on the SCRFD cell origin. Distances are multiplied by the stride. This matches the published `det_10g` export. A RetinaFace graph fails with an explicit error.

The recognizer consumes a 112 crop, normalized the same way, and L2-normalizes the 512-d vector. The swapper multiplies that vector by the graph initializer named `emap`, renormalizes, and runs a 128 RGB crop scaled by `1/255`. Input names are taken from rank (4-D image, 2-D latent), not from a hard-coded string. Output is NCHW RGB in 0..1.

Color transfer is a per-channel mean and deviation match inside the mask, then the inverse warp.

## How this tree differs

- MIT code, no vendored FaceFusion, Deep-Live-Cam, roop, or SimSwap sources
- No model download in the installer, the CLI, or CI
- Compliance is a real package and is off unless the operator opts in
- Messages go through catalogs instead of one language
- The enhancer, webcam, video, and Gradio UI are named seams, not implementations
- Hash pins are optional until the operator sets them
- The content filter default is a stub that allows the frame and logs that no detector is loaded

## Non-goals for Phase 1

Video, realtime capture, GFPGAN/CodeFormer, TensorRT engine build, a RetinaFace decoder, and any redistribution of InsightFace or SimSwap weights.
