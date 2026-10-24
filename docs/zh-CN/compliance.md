# 合规

`compliance.enabled` 默认为 true。本地开关把它关掉时，策略对象仍然会运行，并记录强制处于关闭。本地开关是 `BARRETT_DEV_MODE=1`、`BARRETT_COMPLIANCE=off`，以及被 git 忽略的 `configs/local.yaml`。

文件运行和实时会话都要求同意记录，给输出加水印，运行内容过滤，并追加一行审计。随仓库提供的过滤提供者是 `stub`（放行该帧，并写明没有加载检测器）。`nudenet` 是可选项，不判断年龄。实时会话检查源图和第一帧摄像头画面，在内存里给每一帧输出加水印，不给容器打标签。Gradio 摄像头页使用同一组钩子。

可见文案是 `AI-generated` 或 `人工智能生成`。这六个汉字是原创位图。其他字符需要把 `compliance.watermark.font_path` 指向你有权使用的 TTF 或 OTF。仓库不捆绑第三方字体。

磁盘续跑会跳过已经有同名 `.wm` 标记的帧，因此色带不会画两次。

完整规则在 `docs/COMPLIANCE.zh-CN.md`。本页不是法律意见。
