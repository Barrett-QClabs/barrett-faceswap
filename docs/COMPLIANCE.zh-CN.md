[English](COMPLIANCE.md) | [中文](COMPLIANCE.zh-CN.md)

# 合规

这不是法律意见。它描述 `src/barrett_faceswap/compliance/` 里的代码行为，以及操作者必须自己遵守的限制。

## 总开关

`compliance.enabled` 默认是 **false**。

它为 false 时，`CompliancePolicy.before_process` 和 `after_process` 仍然会运行，但会跳过同意要求、水印、内容过滤拦截和审计日志。它们会留下一条警告：强制已关闭，真实肖像需要打开它。即使 YAML 写着 false，`--enable-compliance` 也会为这一轮打开强制。没有一个参数能把写着 true 的配置再关掉。

`configs/default.yaml` 里的嵌套块是开关打开之后生效的配置：

- 必须有同意记录
- 写入可见标识和内嵌元数据
- 运行内容过滤（默认提供者 `stub`）
- 每一次完成、拒绝或失败都追加一行 JSON

单元测试会构造 `enabled: true` 的配置并走这些路径。直接干跑随仓库提供的文件则不会。

## 可接受的使用

- 源人脸只能在该人授权了这次用途时使用。
- 不要做未经同意的换脸。
- 不要做色情深度伪造。
- 不要制作涉及未成年人的性内容，也不要换未成年人的脸。
- 不要用于欺诈、冒充或其他欺骗。
- 对外分享的合成结果要标明是合成的。

过滤器和水印不能代替同意记录。

## 同意

`compliance/consent.py` 接受两种形式：

- `--consent-file` 指向一个 JSON 对象，包含 `subject_id`、`statement`（至少 10 个字符）、`consented_on`（`YYYY-MM-DD`）和 `scope`。`expires_on` 可选。过期记录会被拒绝。
- `--acknowledge-authorized-use` 与 `--consent-subject` 一起使用。省略 `--consent-statement` 时，确认文本使用目录里的默认句子。

审计会保存 `consent_id`（`file:` 或 `ack:` 加一段短哈希）。两种形式同时传入会被拒绝。`examples/consent.example.json` 是模板，不是授权。

## 水印

`compliance/watermark.py` 在底部画一条带当前文案的色带。默认文案来自目录里的 `watermark.label`：英文是 `AI-generated`，中文是 `人工智能生成`。可以用 `compliance.watermark.text` 覆盖。

PNG 还会写入 `tEXt` 块（`barrett.faceswap.ai_generated=true` 和 Description）。JPEG 写入 EXIF ImageDescription。文件旁边会有一个 `.meta.json` 旁路文件。

视频运行在强制和水印都打开时给每一帧输出打水印，然后调用 `apply_video_metadata` 写容器注释和旁路文件。磁盘模式、管道模式和续跑的任务都走同一组 `before_process` / `complete_output` / `abort` 钩子。管道模式在水印之前抽检一组估计样本（时长乘以帧率，再加上最后一帧解码结果）。多人脸选择留在处理器内部，仍然只返回一帧，因此不能绕过水印。过滤检查用的是抽检（`video.filter_every_n_frames`，外加最后一帧），不是每一帧。写元数据的那次 ffmpeg 调用失败时，运行仍然成功：像素上已经有标记，旁路文件也会写上。如果完全没有 ffmpeg，抽帧会在编码之前失败；静态图像运行在没有 ffmpeg 时仍会写旁路文件。可见色带用 OpenCV 的 Hershey 字体画 `AI-generated`。`人工智能生成` 用 `glyphs.py` 里原创的 12×12 位图画出来，只覆盖这六个字。仓库不捆绑第三方字体。文案里还有其他字符时，把 `compliance.watermark.font_path` 设为你有权使用的 TTF 或 OTF。文件缺失时会记一条日志，并画出已知字形。磁盘续跑在打过水印的帧旁边写 `frame.png.wm`，已有该标记的路径会被跳过，因此色带不会画两次。实时会话在内存里给帧加水印，并记录 `live.session_marked`，不给容器打标签。

在强制打开时，`--no-watermark` 是跳过标识的显式、会被记录的方式。配置里把 `watermark.enabled` 设为 false 却不带这个参数，运行会被拒绝，这样只改 YAML 不能悄悄去掉标识。

## 内容过滤

`compliance/content_filter.py` 定义了 `ContentFilter.assess` 和一组构建器。

| `provider` | 行为 |
| --- | --- |
| `stub` | 默认。放行该帧，并写明没有加载真实检测器。 |
| `none` | 关闭过滤。这是削弱。 |
| `nudenet` | 只有被选中时才导入。不是项目依赖。 |
| `custom` | 从 `content_filter.custom_entrypoint` 加载 `module:factory`。 |

`content_filter.threshold` 和 `content_filter.block_on_minor` 的初始值是 `0.6` 和 `true`。

工厂收到 `FilterSettings`，返回带 `assess(image, role=...) -> FilterDecision` 的对象。决定可以设置 `allowed`、`nsfw_flag` 和 `minor_flag`。`stub` 和 NudeNet 的 `supports_minor_detection` 都是 false。NudeNet 不判断年龄。自定义提供者如果设置了 `minor_flag`，并且 `block_on_minor` 为 true，该帧会被拦截。stub 不会假装看到了未成年人；它会说明未成年人检测不可用。

### 打开一个真实检测器

改 YAML（这不是命令行覆盖，所以不需要权限变量）：

```yaml
compliance:
  enabled: true
  content_filter:
    enabled: true
    provider: nudenet
    threshold: 0.6
    block_on_minor: true
```

然后 `pip install nudenet`。这个包是可选项。没安装时，运行会停下来并提示如何安装。NudeNet 仍然不能满足未成年人检查。那是操作者的责任，不是放行证明。

自定义模块：

```python
def build(settings):
    class Gate:
        name = "custom"
        supports_minor_detection = True
        def assess(self, image, *, role):
            ...
    return Gate()
```

```yaml
content_filter:
  provider: custom
  custom_entrypoint: "my_package.filters:build"
```

### 权限开关

关闭过滤器、把提供者设为 `none`、提高阈值、关掉 `block_on_minor`，或者用命令行改提供者（`--content-filter-provider`、`--disable-content-filter`、`--no-block-on-minor`），需要下面之一：

- 环境变量 `BARRETT_FACESWAP_ALLOW_FILTER_DISABLE=1`
- `configs/admin.toml` 中写 `allow_filter_disable = true`（该文件已被 gitignore）

没有这个开关时，CLI 会拒绝。强制打开时，这次尝试会被记入审计。如果 YAML 自己把内容过滤写成 `enabled: false`、`provider: none` 或 `block_on_minor: false`，同样需要这个开关。在 YAML 里把 `provider` 写成 `nudenet` 是文档中的启用方式，不使用这个变量。

每一次被允许的削弱都会在审计行里记下 `permission_used` 和原因码。

过滤器是纵深防御。它不授权一张脸，stub 提供者也不审查一张脸。

## 审计日志

`compliance/audit_log.py` 把一个 JSON 对象追加到 `compliance.audit.path`（默认 `outputs/audit/audit.jsonl`）的一行。写入时以追加模式打开，并在 Unix 上加锁。它不会改写更早的行。

字段包括时间、事件（`completed`、`refused`、`failed`）、输入哈希、同意编号、输出路径、水印标记和过滤决定。强制关闭的运行不写这个文件。强制运行时如果日志无法追加，该轮失败。

## 法律提示

操作者必须遵守自己所在地的法律。下面是索引，不是对某一次部署的结论。

**中国。** 《互联网信息服务深度合成管理规定》（2023 年 1 月 10 日起施行）要求深度合成服务提供者对合成内容进行标识，尤其是结果可能被误认为真实记录的时候，并禁止把深度合成用于违法目的。《生成式人工智能服务管理暂行办法》也要求生成式人工智能服务对生成内容进行标识。本项目的水印路径是对标识义务的一种技术回应。某一次部署是否构成这些规则里的“提供服务”，需要咨询律师。

**欧盟。** 《人工智能法案》的透明度规则（第 50 条）期望：部署那些生成或操纵构成深度伪造的图像、音频或视频的系统时，披露内容是人工生成或操纵的。艺术作品等有更窄的披露路径。这里同样只是指向该法律文件，而不是合规结论。

**权重。** InsightFace 的库代码是 MIT。它的预训练模型，包括 `buffalo_l` 和 `inswapper_128.onnx`，在 InsightFace 另行授权之前只供非商业研究。他们 2025-11-24 的更新说明：inswapper 系列联系 `contact@insightface.ai`，开放识别包联系 `recognition-oss-pack@insightface.ai`。SimSwap 是 CC BY-NC 4.0，这里没有附带。见 [RESEARCH.zh-CN.md](RESEARCH.zh-CN.md)。

## 代码不会替你做的事

它不会把一份模板 JSON 当成真实同意。提供者是 `stub` 或 NudeNet 时，它不会检测未成年人。总开关关闭时，它不会给输出打标识。处理真实人脸之前，请打开开关。
