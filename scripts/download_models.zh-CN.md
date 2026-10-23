[English](download_models.md) | [中文](download_models.zh-CN.md)

# 下载模型权重

不要把文件提交进仓库。不要在 CI 里下载它们。请自行复制到 `models/`，然后可以用 `barrett-faceswap inspect-models` 锁定 sha256。

期望的文件名：

| 角色 | 文件 | 注册表中的对齐 | 输入 |
| --- | --- | --- | --- |
| 检测 | `models/det_10g.onnx` | `scrfd` | 640 |
| 识别 | `models/w600k_r50.onnx` | `arcface_112` | 112 |
| 换脸 | `models/inswapper_128.onnx` | `inswapper_128` | 128 |

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

## RetinaFace

RetinaFace 的训练和导出说明在 <https://github.com/deepinsight/insightface/tree/master/detection/retinaface>。第一阶段不解码那种图。如果文件不是 SCRFD 的形状，检测器会报错，并指出期望的导出是 `det_10g.onnx`。以后的适配器可以注册另一个检测器，而不必改贴回。

## 许可证

这些权重不受本仓库 MIT 许可覆盖。

InsightFace 声明预训练模型仅供非商业研究。2025-11-24 的更新要求联系：

- inswapper 系列：`contact@insightface.ai`
- `buffalo_l` 这类开放识别包：`recognition-oss-pack@insightface.ai`

在依赖某个文件之前，请到 <https://github.com/deepinsight/insightface#license> 核对当前文本。商业使用需要他们的授权。SimSwap 的检查点是 CC BY-NC 4.0，不属于这个目录布局。

## 锁定哈希

文件放好之后：

```bash
barrett-faceswap inspect-models
```

把打印出的 sha256 抄进 `configs/default.yaml` 的 `models.<role>.sha256`。下一次加载会拒绝一个不同的文件。锁定值为空时仍会计算哈希并在日志里给出摘要，但不会因此失败。
