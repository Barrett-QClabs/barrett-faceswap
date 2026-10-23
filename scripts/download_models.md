[English](download_models.md) | [中文](download_models.zh-CN.md)

# Download model weights

Do not commit the files. Do not add a CI step that downloads them. From the repository root:

```bash
barrett-faceswap download-models
barrett-faceswap download-models --with-gfpgan
barrett-faceswap download-models --with-masks
barrett-faceswap doctor
```

`download-models` fetches `buffalo_l.zip` and `inswapper_128.onnx`, checks sha256, resumes a partial file when the server answers HTTP 206, and tries the next mirror after three failures. Each attempt bounds the TCP connect at 15 seconds and the read at 30 seconds, so a hung mirror fails over instead of waiting on the socket. `--from-dir PATH` copies files by basename (including inside subdirectories) and checks sha256 before any download. A hash mismatch stops the command. A missing file is still downloaded. A `buffalo_l.zip` in that directory is extracted the same way as a downloaded archive. `--force` downloads again. `--with-gfpgan` and `--with-masks` are optional. `--with-masks` fetches `xseg_1.onnx` and `bisenet_resnet_34.onnx`. `--from-dir` also accepts those upstream names and the stored names `occlusion_xseg.onnx` and `bisenet_face_parsing.onnx`. The mask files' licenses are unverified and likely non-commercial. `doctor` prints execution providers, whether CUDA/cuDNN preload ran, the sha256 of the local files, and a warning when both `onnxruntime` and `onnxruntime-gpu` are installed. CI does not run either command.

Pins checked by the command:

| File | sha256 |
| --- | --- |
| `det_10g.onnx` | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| `w600k_r50.onnx` | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| `inswapper_128.onnx` | `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af` |
| `gfpgan_1.4.onnx` | `accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385` |
| `xseg_1.onnx` (stored as `occlusion_xseg.onnx`) | `c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3` |
| `bisenet_resnet_34.onnx` (stored as `bisenet_face_parsing.onnx`) | `4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96` |
| `2dfan4.onnx` | `678c6fa539d52335a31c980feefdf4a6e02d781d83dce00af8a894f114557285` |

The buffalo archive itself is not pinned. The command hashes the two ONNX members after extraction, so a valid re-pack is accepted. You can still copy files by hand and pin them with `barrett-faceswap inspect-models`.

`--with-landmarks` is not part of the default download. The quality set, matching the shipped config, is:

```bash
barrett-faceswap download-models --swapper hyperswap_1a_256 --with-masks --with-landmarks
```

The default command still fetches only `buffalo_l.zip` and `inswapper_128.onnx`. `swap.model` is `hyperswap_1a_256`, so that file has to be downloaded before a real run. Occlusion stays `xseg_1`. Providers stay CUDA, then CPU. `tensorrt.fp16` is false. Video does not run the enhancer unless `--with-enhancer` adds it.

## 2dfan4.onnx

Optional 68-point landmark refiner. `--with-landmarks` stores it at `models/2dfan4.onnx`. The only accepted sha256 is `678c6fa539d52335a31c980feefdf4a6e02d781d83dce00af8a894f114557285`.

- <https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.onnx>
- <https://huggingface.co/facefusion/models-3.0.0/resolve/main/2dfan4.onnx>

The weight is the 2018 FAN model by Adrian Bulat (breadbread1984), MIT, redistributed via facefusion-assets. `commercial_use` stays false. A missing file leaves the detector landmarks in place. `compliance.enabled` stays false.

Expected names:

| Role | Filename | Registry alignment | Input |
| --- | --- | --- | --- |
| detector | `models/det_10g.onnx` | `scrfd` | 640 |
| recognizer | `models/w600k_r50.onnx` | `arcface_112` | 112 |
| landmark | `models/2dfan4.onnx` | `fan_68` | 256 |
| swapper | `models/inswapper_128.onnx` | `inswapper_128` | 128 |
| gfpgan | `models/gfpgan_1.4.onnx` | `ffhq_512` | 512 |
| codeformer | `models/codeformer.onnx` | `ffhq_512` | 512 |
| occluder | `models/occlusion_xseg.onnx` | `xseg` | 256 |
| parser | `models/bisenet_face_parsing.onnx` | `bisenet_19` | 512 |
| hyperswap_1a | `models/hyperswap_1a_256.onnx` | `arcface_128` | 256 |
| hyperswap_1b | `models/hyperswap_1b_256.onnx` | `arcface_128` | 256 |
| hyperswap_1c | `models/hyperswap_1c_256.onnx` | `arcface_128` | 256 |
| simswap_256 | `models/simswap_256.onnx` | `arcface_112_v1` | 256 |
| simswap_512 | `models/simswap_unofficial_512.onnx` | `arcface_112_v1` | 512 |
| blendswap_256 | `models/blendswap_256.onnx` | `ffhq_256` | 256 |
| ghost_1 | `models/ghost_1_256.onnx` | `arcface_112_v1` | 256 |
| ghost_2 | `models/ghost_2_256.onnx` | `arcface_112_v1` | 256 |
| ghost_3 | `models/ghost_3_256.onnx` | `arcface_112_v1` | 256 |
| crossface_simswap | `models/crossface_simswap.onnx` | `embedding_512` | 512-d |
| crossface_ghost | `models/crossface_ghost.onnx` | `embedding_512` | 512-d |
| gpen_bfr_512 | `models/gpen_bfr_512.onnx` | `ffhq_512` | 512 |
| gpen_bfr_1024 | `models/gpen_bfr_1024.onnx` | `ffhq_1024` | 1024 |
| gpen_bfr_2048 | `models/gpen_bfr_2048.onnx` | `ffhq_2048` | 2048 |
| restoreformer_plus_plus | `models/restoreformer_plus_plus.onnx` | `ffhq_512` | 512 |
| xseg_2 | `models/xseg_2.onnx` | `xseg` | 256 |
| xseg_3 | `models/xseg_3.onnx` | `xseg` | 256 |
| sam2_refine | `models/sam2_refine.onnx` | `mask_refine` | 256 |

The enhancer, mask, extra swapper, and crossface files are optional. A default inswapper swap runs without them. `face_enhancer` needs the enhancer file you select. A missing occlusion file stays an all-ones mask. A missing parser uses a feathered landmark convex hull on the swap crop, and an ellipse when no landmarks are passed. Selecting SimSwap or Ghost also opens that model's crossface graph. HyperSwap and BlendSwap do not. Nothing in this document is fetched by CI. `download-models --swapper`, `--enhancer`, and `--occluder` fetch the extra rows. `sam2_refine` is not downloaded.

## InsightFace buffalo_l

`det_10g.onnx` and `w600k_r50.onnx` are the detector and recognizer inside the `buffalo_l` pack (SCRFD-10GF and ResNet50@WebFace600K). The pack also contains landmark and attribute models this phase does not load.

`download-models` tries these URLs in order and extracts members by filename (`buffalo_l/det_10g.onnx` is accepted):

- <https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip>
- <https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip>

Leave `1k3d68.onnx`, `2d106det.onnx`, and `genderage.onnx` out. The model zoo write-up is <https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md>. The release tag is <https://github.com/deepinsight/insightface/releases/tag/model-zoo>. A Google Drive copy of the same pack is fine if you check the two ONNX hashes above.

The recognizer must be this WebFace600K ArcFace model. The inswapper graph was exported against that embedding. A different recognizer produces a bad swap.

## inswapper_128.onnx

`download-models` tries these URLs in order:

- <https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx>
- <https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx>
- <https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128.onnx>

Place it at `models/inswapper_128.onnx`. Their example says the 128 demo is unmaintained and points at their own product for higher-resolution swaps. The 512×512 matrix is an initializer. The official file stores it last and names it `buff2fs` (InsightFace reads `graph.initializer[-1]`). This repository prefers an initializer named `emap` and otherwise uses the last initializer, then checks that the shape is 512×512. It does not ship a copy.

ONNX contract implemented here: a 4-D RGB image in 0..1 and a 2-D latent. The latent is the L2-normalized product of the 512-d ArcFace embedding and that matrix. The crop uses the inswapper template (ArcFace 112 scaled by size/128, then shifted by `8 * size / 128` pixels on x). Output is decoded as 0..1 RGB. `commercial_use` is false.

## HyperSwap 256

There is no weight file in this repository. The index operators use to find `hyperswap_1a_256`, `hyperswap_1b_256`, and `hyperswap_1c_256` packs is:

<https://github.com/facefusion/facefusion-assets>

FaceFusion itself is OpenRAIL-AS. Do not copy that project's source into this tree. `download-models --swapper hyperswap_1a_256` (and `hyperswap_1b_256`, `hyperswap_1c_256`) fetches that ONNX into `models/`.

ONNX contract implemented here, original to this repository:

- Swap image: NCHW RGB float32, `(pixel / 255 - 0.5) / 0.5`, spatial size 256 unless you tile with pixel boost.
- Swap latent: float32 `[1, 512]`, the L2-normalized ArcFace vector. There is no crossface converter.
- Inputs are bound by rank (4-D image, 2-D latent). Names are read from the session.
- Crop template: `arcface_128` (the inswapper 5-point layout scaled to the crop). The output is denormalized with mean 0.5 and standard deviation 0.5.
- `commercial_use` is false. The registry does not grant a commercial license.

## SimSwap

Project and license (CC BY-NC 4.0):

<https://github.com/neuralchen/SimSwap>

Do not copy SimSwap source. `download-models --swapper simswap_256` fetches `models/simswap_256.onnx` and `models/crossface_simswap.onnx`. `--swapper simswap_unofficial_512` fetches the 512 graph and the same converter. Both use the ArcFace 112 v1 template. The 256 graph uses ImageNet mean and standard deviation. The 512 graph uses mean 0 and standard deviation 1. The converter output is L2-normalized. Pixel boost must be a multiple of the native size. `commercial_use` is false.

## GHOST

Project:

<https://github.com/ai-forever/ghost>

The published training code is Apache-2.0. Circulating ONNX packs are not cleared for commercial use by this registry, so `commercial_use` stays false. `download-models --swapper ghost_1_256` (and `ghost_2_256`, `ghost_3_256`) fetches that graph plus `models/crossface_ghost.onnx`. The template is ArcFace 112 v1. Mean and standard deviation are 0.5, and the output is denormalized. The converter vector is not L2-normalized again. Do not copy GHOST source into this tree.

## BlendSwap 256

`download-models --swapper blendswap_256` fetches facefusion-assets `blendswap_256.onnx` (sha256 `20ac5e3c002288e7abd4ea6b1af74103892a6628382d1fa905d13c0270c2667b`) into `models/blendswap_256.onnx`. The target crop is FFHQ at 256 with mean 0 and standard deviation 1. The source is an ArcFace 112 v2 crop, also in 0..1, not an embedding. The upstream license is non-commercial. `commercial_use` is false.

## More enhancers

`--enhancer codeformer` fetches `codeformer.onnx` (sha256 `21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f`). `--enhancer gpen_bfr_512`, `gpen_bfr_1024`, and `gpen_bfr_2048` fetch the matching GPEN ONNX. `--enhancer restoreformer_plus_plus` fetches that file. All of them are FFHQ-aligned. GPEN 512/1024/2048 use 512, 1024, and 2048. CodeFormer fidelity defaults to 0.7. When `models/bisenet_face_parsing.onnx` is present, the enhancer blends only the skin class and leaves eyes and brows on the unenhanced swap. Licenses are non-commercial. `commercial_use` is false. This repository does not copy those projects.

## Occluders

`--occluder xseg_1` fetches `xseg_1.onnx` into `models/occlusion_xseg.onnx`. `--occluder xseg_2` and `--occluder xseg_3` fetch `models/xseg_2.onnx` and `models/xseg_3.onnx`. `--occluder sam2_refine` does not download a SAM 2 weight. Place `models/sam2_refine.onnx` yourself. That graph takes a 0..1 RGB crop and the coarse xseg mask and returns a mask. The coarse xseg file is still fetched. Their licenses are unverified or non-commercial. `commercial_use` is false.

## GFPGAN 1.4

`download-models --with-gfpgan` tries these ONNX mirrors and accepts only sha256 `accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385`:

- <https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx>
- <https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx>

Official checkpoint (PyTorch, not ONNX):

<https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth>

If every mirror times out, export ONNX outside this repository. Do not copy GFPGAN or FaceFusion source into this tree.

1. In a separate environment, install the Apache-2.0 GFPGAN project and load `GFPGANv1.4.pth` into its clean generator.
2. Export one float32 input of shape `1x3x512x512`.
3. This pipeline feeds NCHW RGB with `(pixel / 255 - 0.5) / 0.5`. One 4-D input is enough. Spatial size follows the ONNX input; the registry fallback is 512.
4. A file you export will not match the mirror hash. `download-models` rejects it. Copy it to `models/gfpgan_1.4.onnx` yourself and leave `models.gfpgan.sha256` null, or pin the hash `doctor` prints.

After you have constructed `net` from that upstream project:

```python
import torch

dummy = torch.zeros(1, 3, 512, 512)
torch.onnx.export(
    net,
    dummy,
    "gfpgan_1.4.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
)
```

GFPGAN's code and the published v1.4 weights are Apache-2.0. The network incorporates StyleGAN2, which is under the NVIDIA Source Code License. The model registry sets `commercial_use` to false and does not treat the file as commercially cleared.

## CodeFormer

Project and license:

<https://github.com/sczhou/CodeFormer>

The weights are S-Lab License 1.0 (non-commercial research). Export to `models/codeformer.onnx` with the same 512 NCHW RGB [-1, 1] contract. If the graph has a second, non-image input, this pipeline feeds `--enhancer-weight` (default 0.7) into it.

## Occlusion (xseg-style)

`download-models --with-masks` writes facefusion-assets `xseg_1.onnx` from models-3.1.0 to `models/occlusion_xseg.onnx`. The only accepted sha256 is `c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3`.

```text
https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx
https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx
```

The license of that file is unverified and likely non-commercial. This repository does not grant rights to it. The graph takes 0..1 RGB. The published file is channels-last, `[N, 256, 256, 3]`. A channels-first graph, or one gray channel, is also accepted. It returns one channel where higher means the pixel stays visible. A max above 1.5 is read as 0..255. Set `masks.occlusion_invert: true` when the graph predicts the occluder instead of the face. Asymmetric ConvTranspose padding is rewritten in memory at load so the CUDA provider does not drop those nodes to the CPU. The xseg idea is associated with DeepFaceLab (<https://github.com/iperov/DeepFaceLab>). That code is GPL-3.0 and must not be copied into this tree. `commercial_use` is false.

## Face parsing (BiSeNet)

Training code often cited for this export:

<https://github.com/zllrunning/face-parsing.PyTorch>

That repository's code is MIT. `--with-masks` also writes facefusion-assets `bisenet_resnet_34.onnx` from models-3.0.0 to `models/bisenet_face_parsing.onnx`. The only accepted sha256 is `4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96`.

```text
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx
https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx
```

The license of that file is unverified and likely non-commercial. Weights such as `79999_iter.pth` are commonly trained on CelebAMask-HQ, which is non-commercial research. Input is ImageNet-normalized RGB (mean 0.485, 0.456, 0.406; std 0.229, 0.224, 0.225). Output is either per-class scores or an integer label map. Class indexes are the 19 CelebAMask-HQ labels listed in `src/barrett_faceswap/parsing.py` (`background` is 0, `hat` is 18). Eye and brow classes are also used as the swap eye mask. `commercial_use` is false.

## RetinaFace

RetinaFace training and export notes live at <https://github.com/deepinsight/insightface/tree/master/detection/retinaface>. Phase 1 does not decode that graph. If a file is not SCRFD-shaped, the detector raises and names `det_10g.onnx` as the expected export. A later adapter can register another detector without changing paste-back.

## License

These weights are not covered by this repository's MIT license.

InsightFace states that pretrained models are for non-commercial research only. The 2025-11-24 update asks you to contact:

- `contact@insightface.ai` for the inswapper series
- `recognition-oss-pack@insightface.ai` for open recognition packs such as `buffalo_l`

Confirm the current text at <https://github.com/deepinsight/insightface#license> before you rely on a file. Commercial use needs their grant. SimSwap checkpoints are CC BY-NC 4.0. HyperSwap, BlendSwap, GPEN, and RestoreFormer++ packs are not commercially cleared here. GHOST ONNX packs are not commercially cleared here even where the training code is Apache-2.0. SimSwap and Ghost crossface converters are fetched with those graphs and are not commercially cleared here. GFPGAN, CodeFormer, the occlusion files, and the parsing file have the separate terms in the sections above. None of them are covered by this repository's MIT license. CI does not download any of these files.

## Pin the hash

After the files are in place:

```bash
barrett-faceswap inspect-models
```

Copy each printed sha256 into `configs/default.yaml` under `models.<role>.sha256`. The next load refuses a different file. An empty pin still hashes the file and logs the digest. It does not fail the run.

## fp16 siblings

`execution.precision: fp16` looks for a sibling next to the configured file. `models/inswapper_128.onnx` selects `models/inswapper_128_fp16.onnx` when that file is present, and stays on the fp32 file when it is not. The same `*_fp16` suffix applies to every role. This page does not fetch either variant. The default precision is `fp32`.

## A commercial weight

Every filename above is registered with `commercial_use: false`. `barrett-faceswap swap --commercial` refuses them. A weight you train yourself is not downloaded by `scripts/train_swapper.py`. After legal review, add your own registry row with `commercial_use: true`. The gate reads that flag and does not grant an exception for the rows in this table. See [docs/COMPLIANCE.md](../docs/COMPLIANCE.md).

## OpenVoice V2 converter (optional voice)

Not an ONNX face weight, and not fetched by `download-models`. Code and checkpoint are MIT. Layout and the RTX 5090 PyTorch wheel are in [docs/en/voice.md](../docs/en/voice.md).

```bash
huggingface-cli download myshell-ai/OpenVoiceV2 \
  converter/config.json converter/checkpoint.pth \
  --local-dir models/openvoice_v2
```

`models/openvoice_v2/converter/checkpoint.pth` is the tone-color converter. The base-speaker embeddings in that repo are not required. Do not commit the files.
