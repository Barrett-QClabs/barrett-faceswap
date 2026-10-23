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
- `hyperswap_1a_256.onnx`, `hyperswap_1b_256.onnx`, `hyperswap_1c_256.onnx`
- `simswap_256.onnx`, `simswap_unofficial_512.onnx`
- `ghost_1_256.onnx`, `ghost_2_256.onnx`, `ghost_3_256.onnx`
- `crossface_hyperswap.onnx`, `crossface_simswap.onnx`, `crossface_ghost.onnx`

The extra swapper and crossface files are optional. A default inswapper run
does not open them. Contracts and sources are in `scripts/download_models.md`
and `scripts/download_models.zh-CN.md`.

Those weights have their own license. InsightFace pretrained models, including
the inswapper series, are not covered by this repository's MIT license.
HyperSwap, SimSwap (CC BY-NC), Ghost ONNX packs, crossface converters, GFPGAN,
CodeFormer, and the mask files are optional and are not commercially cleared
by this registry. See the README, `docs/COMPLIANCE.md`, and
`scripts/download_models.md`.

中文：额外的换脸器和 crossface 文件都是可选的。默认的 inswapper 运行不会打开它们。约定和来源见 `scripts/download_models.zh-CN.md`。这些权重不在本仓库的 MIT 许可里，注册表里的 `commercial_use` 都是 false。
