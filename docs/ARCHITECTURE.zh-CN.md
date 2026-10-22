[English](ARCHITECTURE.md) | [中文](ARCHITECTURE.zh-CN.md)

# 架构

第一阶段是一个集成骨架。公开入口是 `FaceSwapPipeline.run`。它总会调用 `CompliancePolicy.before_process` 和 `after_process`。处理器只返回帧，不写输出文件，因此不能靠某个处理器提前存盘来绕过水印。

背景阅读在 [RESEARCH.zh-CN.md](RESEARCH.zh-CN.md)。

## 流水线

```text
SwapRequest
  -> CompliancePolicy.before_process
       同意、过滤权限、源/目标过滤        （仅在强制打开时）
  -> 按顺序执行处理器
       face_swapper：检测 -> 对齐 -> 向量 -> 替换 -> 蒙版 -> 贴回
       face_enhancer：钩子，真实运行时会拒绝
  -> CompliancePolicy.after_process
       输出过滤、水印、审计、写文件
```

`--dry-run` 会跳过 ONNX，但仍然经过策略。在随仓库提供的配置下，策略会记录强制已关闭，并写出不带标记的图像。

## 采纳的想法

**处理器协议。** `Processor` 有 `pre_check`、`process_frame` 和 `post_process`。`processors/registry.py` 把名称映射到工厂，工厂接收 pydantic 配置和提供者列表。`face_swapper` 已注册。`face_enhancer` 已注册，真实运行时会报错，这样链条可以列出它，又不会假装 GFPGAN 已经跑过。这是 FaceFusion 那类接缝。方法和注册表是本地代码。

**会话缓存。** `ModelStore.session` 用解析后的路径、sha256、请求的提供者元组以及设备号作为键。重复调用不会重建会话。`prefer_cuda` 在请求包含 `cuda` 时把它放到最前，`cpu` 留在最后。如果安装的 ONNX Runtime 没有 CUDA 提供者，这个名称会被跳过，CPU 仍然保留。可以写出 `tensorrt`，日志会说明它延后：第一阶段不配置引擎缓存或构建参数。

**对齐和贴回。** `align.py` 放着 ArcFace 五点模板。112 是识别裁剪。128 在此基础上向右平移 8 像素，这是 inswapper 导出所期望的。`blend.py` 做逆变换并按蒙版合成。`masks.py` 生成内缩方框、椭圆和全 1 的遮挡占位，再逐元素取最小值。遮挡占位已经接上，以后的模型可以返回真实图，而不必新写贴回函数。

**模型注册表。** `model_registry.py` 为检测器、识别器和换脸器记录：文件名、许可键、`commercial_use=False`、对齐名称和输入尺寸。sha256 锁定值在配置里，默认是 null，因为本仓库不重新分发这些文件。加载时会计算哈希。有锁定值就必须匹配。锁定值为空时会把摘要打出来，便于抄进 `models.<role>.sha256`。`inspect-models` 打印合并后的一行。

**配置。** `config.py` 是 pydantic 模型。CLI 从 YAML 加载同一个模型。相对路径相对工作目录解析。

**国际化。** `i18n/locales/en.yaml` 和 `zh_CN.yaml` 是仅有的用户可见文案目录。CLI 帮助、日志、错误、合规文本和 `ui.gradio_labels` 都走 `t(key)`。语言顺序是 `--lang`，然后 `BARRETT_FACESWAP_LANG`，然后 `ui.language`，最后是 `en`。

**合规。** 见 [COMPLIANCE.zh-CN.md](COMPLIANCE.zh-CN.md)。总开关默认关闭。策略对象仍然会执行。

## ONNX 约定

检测器把图像缩放到配置的正方形（默认 640），转成 RGB，并用 `(pixel - 127.5) / 128` 归一化。输出头按通道分组：1 是分数，4 是距离，10 是关键点。锚点放在 SCRFD 格子的原点。距离乘以步长。这对应公开的 `det_10g` 导出。RetinaFace 图会得到明确的错误。

识别器吃 112 的裁剪，归一化方式相同，再对 512 维向量做 L2 归一化。换脸器把该向量乘以图里名为 `emap` 的初始值，再次归一化，并跑一张按 `1/255` 缩放的 128 RGB 裁剪。输入名按秩选取（4 维图像、2 维向量），不把名字写死。输出是 0 到 1 的 NCHW RGB。

颜色对齐是蒙版内的逐通道均值和标准差匹配，然后做逆变换。

## 这个仓库的差别

- 代码是 MIT，没有打包 FaceFusion、Deep-Live-Cam、roop 或 SimSwap 的源码
- 安装器、CLI 和 CI 都不会下载模型
- 合规是完整的包，操作者不打开就保持关闭
- 文案走目录，而不是只有一种语言
- 增强器、摄像头、视频和 Gradio 只是命名好的接缝，不是实现
- 哈希锁定在操作者填写之前是可选的
- 内容过滤的默认提供者是 stub：放行该帧，并写明没有加载检测器

## 第一阶段不做的事

视频、实时采集、GFPGAN/CodeFormer、TensorRT 引擎构建、RetinaFace 解码器，以及再分发 InsightFace 或 SimSwap 的权重。
