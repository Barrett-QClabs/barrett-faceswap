[English](download_models.md) | [中文](download_models.zh-CN.md)

# Download model weights

Do not commit the files. Do not add a CI step that downloads them. Copy them into `models/` yourself, then optionally pin the sha256 with `barrett-faceswap inspect-models`.

Expected names:

| Role | Filename | Registry alignment | Input |
| --- | --- | --- | --- |
| detector | `models/det_10g.onnx` | `scrfd` | 640 |
| recognizer | `models/w600k_r50.onnx` | `arcface_112` | 112 |
| swapper | `models/inswapper_128.onnx` | `inswapper_128` | 128 |
| gfpgan | `models/gfpgan_1.4.onnx` | `ffhq_512` | 512 |
| codeformer | `models/codeformer.onnx` | `ffhq_512` | 512 |
| occluder | `models/occlusion_xseg.onnx` | `xseg` | 256 |
| parser | `models/bisenet_face_parsing.onnx` | `bisenet_19` | 512 |

The last four files are optional. A swap runs without them. `face_enhancer` needs the enhancer file you select. Occlusion and parsing masks fall back to stubs when their files are absent.

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

## GFPGAN 1.4

Official checkpoint (PyTorch, not ONNX):

<https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth>

Export it yourself to `models/gfpgan_1.4.onnx`. This repository does not ship a converter and does not download the file. The graph this pipeline runs is NCHW RGB float32, normalized with `(pixel / 255 - 0.5) / 0.5`, spatial size 512 unless the ONNX input says otherwise. One 4-D input is enough. GFPGAN's code and the published v1.4 weights are Apache-2.0. The network incorporates StyleGAN2, which is under the NVIDIA Source Code License. The model registry sets `commercial_use` to false and does not treat the file as commercially cleared.

## CodeFormer

Project and license:

<https://github.com/sczhou/CodeFormer>

The weights are S-Lab License 1.0 (non-commercial research). Export to `models/codeformer.onnx` with the same 512 NCHW RGB [-1, 1] contract. If the graph has a second, non-image input, this pipeline feeds `--enhancer-weight` (default 0.5) into it.

## Occlusion (xseg-style)

Place an operator-supplied file at `models/occlusion_xseg.onnx`. There is no blessed URL. The expected graph takes 0..1 RGB (or one gray channel) at 256 and returns one channel where higher means the pixel stays visible. A max above 1.5 is read as 0..255. Set `masks.occlusion_invert: true` when the graph predicts the occluder instead of the face. The xseg idea is associated with DeepFaceLab (<https://github.com/iperov/DeepFaceLab>). That code is GPL-3.0 and must not be copied into this tree. `commercial_use` is false.

## Face parsing (BiSeNet)

Training code often cited for this export:

<https://github.com/zllrunning/face-parsing.PyTorch>

That repository's code is MIT. Weights such as `79999_iter.pth` are commonly trained on CelebAMask-HQ, which is non-commercial research. Confirm the file you actually place at `models/bisenet_face_parsing.onnx`. Input is ImageNet-normalized RGB (mean 0.485, 0.456, 0.406; std 0.229, 0.224, 0.225). Output is either per-class scores or an integer label map. Class indexes are the 19 CelebAMask-HQ labels listed in `src/barrett_faceswap/parsing.py` (`background` is 0, `hat` is 18). `commercial_use` is false.

## RetinaFace

RetinaFace training and export notes live at <https://github.com/deepinsight/insightface/tree/master/detection/retinaface>. Phase 1 does not decode that graph. If a file is not SCRFD-shaped, the detector raises and names `det_10g.onnx` as the expected export. A later adapter can register another detector without changing paste-back.

## License

These weights are not covered by this repository's MIT license.

InsightFace states that pretrained models are for non-commercial research only. The 2025-11-24 update asks you to contact:

- `contact@insightface.ai` for the inswapper series
- `recognition-oss-pack@insightface.ai` for open recognition packs such as `buffalo_l`

Confirm the current text at <https://github.com/deepinsight/insightface#license> before you rely on a file. Commercial use needs their grant. SimSwap checkpoints are CC BY-NC 4.0 and are not part of this layout. GFPGAN, CodeFormer, the occlusion file, and the parsing file have the separate terms in the sections above. None of them are covered by this repository's MIT license.

## Pin the hash

After the files are in place:

```bash
barrett-faceswap inspect-models
```

Copy each printed sha256 into `configs/default.yaml` under `models.<role>.sha256`. The next load refuses a different file. An empty pin still hashes the file and logs the digest. It does not fail the run.
