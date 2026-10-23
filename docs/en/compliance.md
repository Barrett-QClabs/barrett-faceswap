# Compliance

`compliance.enabled` defaults to false and stays false until an official launch. The policy object still runs. With the default it logs that enforcement is off.

When enforcement is on, a file run and a live session both require a consent record, watermark output, run the content filter, and append one audit line. A live session screens the source image and the first camera frame, watermarks each output frame in memory, and does not tag a container. The Gradio webcam tab uses the same hooks.

The visible label is `AI-generated` or `人工智能生成`. The six Chinese characters are original bitmaps. Other characters need `compliance.watermark.font_path` pointing at a TTF or OTF you are allowed to use. No third-party font is bundled.

A disk resume skips a frame that already has a sibling `.wm` stamp, so the bar is not drawn twice.

The full rules are in `docs/COMPLIANCE.md`. This page is not legal advice.
