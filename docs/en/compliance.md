# Compliance

`compliance.enabled` defaults to true. The policy object still runs when a local switch turns it off, and then it logs that enforcement is off. Local switches are `BARRETT_DEV_MODE=1`, `BARRETT_COMPLIANCE=off`, and a gitignored `configs/local.yaml`.

A file run and a live session both require a consent record, watermark output, run the content filter, and append one audit line. The shipped filter provider is `stub` (allows the frame, logs that no detector is loaded). `nudenet` is optional and does not classify age. A live session screens the source image and the first camera frame, watermarks each output frame in memory, and does not tag a container. The Gradio webcam tab uses the same hooks.

The visible label is `AI-generated` or `人工智能生成`. The six Chinese characters are original bitmaps. Other characters need `compliance.watermark.font_path` pointing at a TTF or OTF you are allowed to use. No third-party font is bundled.

A disk resume skips a frame that already has a sibling `.wm` stamp, so the bar is not drawn twice.

The full rules are in `docs/COMPLIANCE.md`. This page is not legal advice.
