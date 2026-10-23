# 实时摄像头

```bash
barrett-faceswap live \
  --source path/to/authorized-source.png \
  --frames 30 \
  --no-preview \
  --dry-run
```

`--frames 0` 会一直运行到摄像头关闭。`--camera` 选择 OpenCV 序号（`live.camera`，默认 0）。`--no-preview` 跳过窗口。没有 `DISPLAY` 时窗口会被跳过，并写一条日志。

`--virtual-camera /dev/video2` 经 ffmpeg 把原始 BGR 写到 v4l2loopback 设备。设备缺失时记一条日志。预览继续。

`live.queue_size`（默认 1）限制每个队列。队列满时丢掉最旧的帧。`live.detect_every`（默认 3）按这个间隔做检测。其余帧按上次的速度移动框和关键点。一次没有人脸的检测会清掉这份缓存，这一帧保持原样。源脸向量只准备一次。

Gradio 摄像头页是同一 `LiveSession` 上的流式回调。先点开始，再允许摄像头。合规打开时，第一帧做同意检查。停止写一行审计。线程本身在命令行里。

测试把合成帧送进 `LivePipeline`，不会打开摄像头。
