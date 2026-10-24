# Barrett FaceSwap

Barrett FaceSwap 可以对静态图像、视频或摄像头预览换脸。代码是 MIT。它能调用的 InsightFace 权重不是 MIT，那些权重的商业使用需要另行授权。

随仓库提供的配置里 `compliance.enabled` 是 `true`。运行前需要确认授权。`BARRETT_DEV_MODE=1` 会在本地关掉这项检查。

- [架构](architecture.md)
- [实时摄像头](live.md)
- [商业路线](commercial.md)
- [合规](compliance.md)
- [命令行](cli.md)
- [声音克隆](voice.md)

更长的叙述在站点旁边的 `docs/ARCHITECTURE.zh-CN.md` 和 `docs/COMPLIANCE.zh-CN.md`。这个站点在 CI 里构建，不会部署。
