[English](download_models.md) | [中文](download_models.zh-CN.md)

# Download model weights

Do not commit the files. Do not add a CI step that downloads them. Copy them into `models/` yourself, then optionally pin the sha256 with `barrett-faceswap inspect-models`.

Expected names:

| Role | Filename | Registry alignment | Input |
| --- | --- | --- | --- |
| detector | `models/det_10g.onnx` | `scrfd` | 640 |
| recognizer | `models/w600k_r50.onnx` | `arcface_112` | 112 |
| swapper | `models/inswapper_128.onnx` | `inswapper_128` | 128 |

## InsightFace buffalo_l

`det_10g.onnx` and `w600k_r50.onnx` are the detector and recognizer inside the `buffalo_l` pack (SCRFD-10GF and ResNet50@WebFace600K). The pack also contains landmark and attribute models this phase does not load.

Official index:

- Model zoo write-up: <https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md>
- Release tag: <https://github.com/deepinsight/insightface/releases/tag/model-zoo>
- The zoo page links a Google Drive archive for `buffalo_l`. Unzip it and copy the two filenames above. Leave `1k3d68.onnx`, `2d106det.onnx`, and `genderage.onnx` out unless you add a later stage that uses them.

The recognizer must be this WebFace600K ArcFace model. The inswapper graph was exported against that embedding. A different recognizer produces a bad swap.

## inswapper_128.onnx

InsightFace documents a direct file:

<https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx>

Place it at `models/inswapper_128.onnx`. Their example says the 128 demo is unmaintained and points at their own product for higher-resolution swaps. The graph's `emap` initializer is required. This repository reads that tensor from the ONNX file. It does not ship a copy.

## RetinaFace

RetinaFace training and export notes live at <https://github.com/deepinsight/insightface/tree/master/detection/retinaface>. Phase 1 does not decode that graph. If a file is not SCRFD-shaped, the detector raises and names `det_10g.onnx` as the expected export. A later adapter can register another detector without changing paste-back.

## License

These weights are not covered by this repository's MIT license.

InsightFace states that pretrained models are for non-commercial research only. The 2025-11-24 update asks you to contact:

- `contact@insightface.ai` for the inswapper series
- `recognition-oss-pack@insightface.ai` for open recognition packs such as `buffalo_l`

Confirm the current text at <https://github.com/deepinsight/insightface#license> before you rely on a file. Commercial use needs their grant. SimSwap checkpoints are CC BY-NC 4.0 and are not part of this layout.

## Pin the hash

After the files are in place:

```bash
barrett-faceswap inspect-models
```

Copy each printed sha256 into `configs/default.yaml` under `models.<role>.sha256`. The next load refuses a different file. An empty pin still hashes the file and logs the digest. It does not fail the run.
