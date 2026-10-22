# Model weights

This directory is where ONNX weights belong on your machine. The repository
does not ship them.

Expected files, after you follow `scripts/download_models.md`:

- `det_10g.onnx` — SCRFD face detector (InsightFace `buffalo_l`)
- `w600k_r50.onnx` — ArcFace recognizer (InsightFace `buffalo_l`)
- `inswapper_128.onnx` — InsightFace identity swapper

Those weights have their own license. InsightFace pretrained models, including
the inswapper series, are not covered by this repository's MIT license. See
the README and `docs/COMPLIANCE.md`.
