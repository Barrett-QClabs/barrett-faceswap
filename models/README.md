# Model weights

This directory is where ONNX weights belong on your machine. The repository
does not ship them.

Expected files, after you follow `scripts/download_models.md`:

- `det_10g.onnx` — SCRFD face detector (InsightFace `buffalo_l`)
- `w600k_r50.onnx` — ArcFace recognizer (InsightFace `buffalo_l`)
- `inswapper_128.onnx` — InsightFace identity swapper
- `gfpgan_1.4.onnx` — optional GFPGAN 1.4 restorer
- `codeformer.onnx` — optional CodeFormer restorer
- `occlusion_xseg.onnx` — optional occlusion mask
- `bisenet_face_parsing.onnx` — optional face-parsing mask

Those weights have their own license. InsightFace pretrained models, including
the inswapper series, are not covered by this repository's MIT license. GFPGAN,
CodeFormer, and the mask files are optional and are not commercially cleared
by this registry. See the README, `docs/COMPLIANCE.md`, and
`scripts/download_models.md`.
