[English](download_models.md) | [中文](download_models.zh-CN.md)

# 下载模型权重

不要把文件提交进仓库。不要在 CI 里下载它们。请自行复制到 `models/`，然后可以用 `barrett-faceswap inspect-models` 锁定 sha256。

期望的文件名：

| 角色 | 文件 | 注册表中的对齐 | 输入 |
| --- | --- | --- | --- |
| 检测 | `models/det_10g.onnx` | `scrfd` | 640 |
| 识别 | `models/w600k_r50.onnx` | `arcface_112` | 112 |
| 换脸 | `models/inswapper_128.onnx` | `inswapper_128` | 128 |
| gfpgan | `models/gfpgan_1.4.onnx` | `ffhq_512` | 512 |
| codeformer | `models/codeformer.onnx` | `ffhq_512` | 512 |
| 遮挡 | `models/occlusion_xseg.onnx` | `xseg` | 256 |
| 解析 | `models/bisenet_face_parsing.onnx` | `bisenet_19` | 512 |
| hyperswap_1a | `models/hyperswap_1a_256.onnx` | `arcface_256` | 256 |
| hyperswap_1b | `models/hyperswap_1b_256.onnx` | `arcface_256` | 256 |
| hyperswap_1c | `models/hyperswap_1c_256.onnx` | `arcface_256` | 256 |
| simswap_256 | `models/simswap_256.onnx` | `arcface_256` | 256 |
| simswap_512 | `models/simswap_unofficial_512.onnx` | `arcface_512` | 512 |
| ghost_1 | `models/ghost_1_256.onnx` | `arcface_256` | 256 |
| ghost_2 | `models/ghost_2_256.onnx` | `arcface_256` | 256 |
| ghost_3 | `models/ghost_3_256.onnx` | `arcface_256` | 256 |
| crossface_hyperswap | `models/crossface_hyperswap.onnx` | `embedding_512` | 512 维 |
| crossface_simswap | `models/crossface_simswap.onnx` | `embedding_512` | 512 维 |
| crossface_ghost | `models/crossface_ghost.onnx` | `embedding_512` | 512 维 |

增强、蒙版、额外换脸器和 crossface 文件都是可选的。默认的 inswapper 换脸不需要它们。`face_enhancer` 需要你选中的那一个增强文件。遮挡和解析蒙版在文件缺失时退回占位。选中另一个换脸器时会打开那个文件和它的 crossface 图。本文档里的文件都不会被 CI 下载。

## InsightFace buffalo_l

`det_10g.onnx` 和 `w600k_r50.onnx` 是 `buffalo_l` 包里的检测器和识别器（SCRFD-10GF 与 ResNet50@WebFace600K）。这个包里还有本阶段不会加载的关键点和属性模型。

官方索引：

- 模型库说明：<https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md>
- 发布标签：<https://github.com/deepinsight/insightface/releases/tag/model-zoo>
- 模型库页面给出了 `buffalo_l` 的 Google Drive 压缩包。解压后复制上面两个文件名。除非以后的阶段用到，否则不必保留 `1k3d68.onnx`、`2d106det.onnx` 和 `genderage.onnx`。

识别器必须是这个 WebFace600K ArcFace 模型。inswapper 图是对着这个向量导出的。换一个识别器会得到错误的替换结果。

## inswapper_128.onnx

InsightFace 记录了一个直接文件地址：

<https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx>

放到 `models/inswapper_128.onnx`。他们的示例写明 128 的演示已不再维护，并建议更高分辨率的替换使用他们自己的产品。图中的 `emap` 初始值是必需的。本仓库从 ONNX 文件读取这个张量，不附带副本。

这里实现的 ONNX 约定：4 维 RGB 图像，范围 0..1，以及 2 维隐向量。隐向量是 512 维 ArcFace 向量与初始化器里 `emap` 矩阵（512×512）的乘积，再做 L2 归一化。裁剪使用 inswapper 模板（ArcFace 112 按 size/128 缩放，再在 x 上平移 `8 * size / 128` 像素）。输出按 0..1 RGB 解码。`commercial_use` 为 false。

## HyperSwap 256

本仓库没有权重文件。操作者查找 `hyperswap_1a_256`、`hyperswap_1b_256`、`hyperswap_1c_256` 包时使用的索引是：

<https://github.com/facefusion/facefusion-assets>

FaceFusion 本身是 OpenRAIL-AS。不要把那个项目的源码复制进本仓库。把 ONNX 放到 `models/hyperswap_1a_256.onnx`、`models/hyperswap_1b_256.onnx` 和 `models/hyperswap_1c_256.onnx`。再放一个操作者自行提供的转换器 `models/crossface_hyperswap.onnx`。

这里实现的约定是本仓库自己的：

- 换脸图像：NCHW RGB float32，`(pixel / 255 - 0.5) / 0.5`。不做像素提升时空间尺寸是 256。
- 换脸隐向量：float32 `[1, 512]`。
- 输入按秩绑定（4 维图像、2 维隐向量）。名称从会话读取。
- Crossface：输入一个 512 维向量，输出一个 512 维向量，再做 L2 归一化。本仓库不附带这张图。
- 裁剪模板：ArcFace 112 按 `size / 112` 缩放，没有 8 像素平移。
- 输出里若有值低于 `-0.01`，按 [-1, 1] 解码；最大值高于 `1.5` 时按 0..255；其余按 0..1。
- `commercial_use` 为 false。注册表不授予商业许可。

## SimSwap

项目和许可证（CC BY-NC 4.0）：

<https://github.com/neuralchen/SimSwap>

不要复制 SimSwap 的源码。请自行导出或取得 `models/simswap_256.onnx`；如果要用更大的非官方图，再放 `models/simswap_unofficial_512.onnx`。共用的转换器是 `models/crossface_simswap.onnx`。约定与上面的 HyperSwap 相同，原生尺寸是 256 和 512。像素提升必须是该原生尺寸的整数倍（768 对 256 有效，对 512 无效）。上游许可是非商业的，因此 `commercial_use` 为 false。

## GHOST

项目：

<https://github.com/ai-forever/ghost>

公开的训练代码是 Apache-2.0。流通中的 ONNX 包并没有被本注册表标成可商业使用，所以 `commercial_use` 保持 false。请放置 `models/ghost_1_256.onnx`、`models/ghost_2_256.onnx`、`models/ghost_3_256.onnx`，以及 `models/crossface_ghost.onnx`。图像、隐向量、模板和输出约定与 HyperSwap 一节相同。原生尺寸是 256。不要把 GHOST 的源码复制进本仓库。

## GFPGAN 1.4

官方检查点（PyTorch，不是 ONNX）：

<https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth>

请自行导出到 `models/gfpgan_1.4.onnx`。本仓库不附带转换器，也不下载该文件。本流水线运行的图是 NCHW RGB float32，用 `(pixel / 255 - 0.5) / 0.5` 归一化，空间尺寸在 ONNX 输入未写明时是 512。一个 4 维输入就够了。GFPGAN 的代码和公开的 v1.4 权重是 Apache-2.0。网络里包含 NVIDIA Source Code License 下的 StyleGAN2。模型注册表把 `commercial_use` 设为 false，不把该文件当成已获商业许可。

## CodeFormer

项目和许可证：

<https://github.com/sczhou/CodeFormer>

权重是 S-Lab License 1.0（非商业研究）。请导出到 `models/codeformer.onnx`，约定同样是 512 的 NCHW RGB [-1, 1]。如果图有第二个非图像输入，本流水线会把 `--enhancer-weight`（默认 0.5）送进去。

## 遮挡（xseg 风格）

把操作者自行准备的文件放到 `models/occlusion_xseg.onnx`。没有指定的下载地址。期望的图在 256 上接收 0..1 的 RGB（或单通道灰度），并返回一个通道，数值越高表示该像素保持可见。最大值高于 1.5 时按 0..255 读取。如果图预测的是遮挡物而不是人脸，把 `masks.occlusion_invert` 设为 true。xseg 这个想法与 DeepFaceLab（<https://github.com/iperov/DeepFaceLab>）有关。那段代码是 GPL-3.0，不得复制进本仓库。`commercial_use` 为 false。

## 人脸解析（BiSeNet）

这种导出常被引用的训练代码：

<https://github.com/zllrunning/face-parsing.PyTorch>

该仓库的代码是 MIT。`79999_iter.pth` 这类权重通常在 CelebAMask-HQ 上训练，而该数据集是非商业研究用途。请核对你实际放到 `models/bisenet_face_parsing.onnx` 的文件。输入是 ImageNet 归一化的 RGB（均值 0.485、0.456、0.406；标准差 0.229、0.224、0.225）。输出是逐类分数或整数标签图。类别序号是 `src/barrett_faceswap/parsing.py` 里的 19 个 CelebAMask-HQ 标签（`background` 是 0，`hat` 是 18）。`commercial_use` 为 false。

## RetinaFace

RetinaFace 的训练和导出说明在 <https://github.com/deepinsight/insightface/tree/master/detection/retinaface>。第一阶段不解码那种图。如果文件不是 SCRFD 的形状，检测器会报错，并指出期望的导出是 `det_10g.onnx`。以后的适配器可以注册另一个检测器，而不必改贴回。

## 许可证

这些权重不受本仓库 MIT 许可覆盖。

InsightFace 声明预训练模型仅供非商业研究。2025-11-24 的更新要求联系：

- inswapper 系列：`contact@insightface.ai`
- `buffalo_l` 这类开放识别包：`recognition-oss-pack@insightface.ai`

在依赖某个文件之前，请到 <https://github.com/deepinsight/insightface#license> 核对当前文本。商业使用需要他们的授权。SimSwap 的检查点是 CC BY-NC 4.0。HyperSwap 包由操作者自行提供，这里没有做商业许可。即使 GHOST 的训练代码是 Apache-2.0，这里也不把 GHOST 的 ONNX 包标成可商业使用。Crossface 转换器由操作者自行提供。GFPGAN、CodeFormer、遮挡文件和解析文件的条款见上面各节。它们都不在本仓库的 MIT 许可之内。CI 不会下载这些文件。

## 锁定哈希

文件放好之后：

```bash
barrett-faceswap inspect-models
```

把打印出的 sha256 抄进 `configs/default.yaml` 的 `models.<role>.sha256`。下一次加载会拒绝一个不同的文件。锁定值为空时仍会计算哈希并在日志里给出摘要，但不会因此失败。
