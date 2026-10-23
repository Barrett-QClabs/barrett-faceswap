# Live webcam

```bash
barrett-faceswap live \
  --source path/to/authorized-source.png \
  --frames 30 \
  --no-preview \
  --dry-run
```

`--frames 0` runs until the camera closes. `--camera` selects the OpenCV index (`live.camera`, default 0). `--no-preview` skips the window. Without `DISPLAY`, the window is skipped and a log line says so.

`--virtual-camera /dev/video2` writes raw BGR through ffmpeg to a v4l2loopback device. A missing device is logged once. The preview continues.

`live.queue_size` (default 1) bounds each queue. A full queue drops the oldest frame. `live.detect_every` (default 3) runs detection on that stride. Other frames move the last box and landmarks by the last velocity. The source embedding is prepared once.

The Gradio webcam tab is a streaming callback over the same `LiveSession`. Press Start, then allow the camera. The first frame runs the consent check when compliance is on. Stop writes one audit line. The threads themselves are the CLI command.

Tests pass synthetic frames into `LivePipeline` and do not open a camera.
