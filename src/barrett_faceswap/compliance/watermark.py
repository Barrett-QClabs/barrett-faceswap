"""Visible label plus file metadata for synthetic output.

Marking runs only while compliance enforcement is on and the operator has not
passed ``--no-watermark``. The visible bar and the embedded text both use the
configured label (default ``AI-generated``).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

from barrett_faceswap.errors import WatermarkError
from barrett_faceswap.i18n import t

AI_TEXT_KEY = "barrett.faceswap.ai_generated"
SOFTWARE_TAG = "barrett-faceswap"


@dataclass(frozen=True)
class VideoMarkResult:
    """How a video file was labeled. Phase 1 does not emit video frames."""

    container_tagged: bool
    sidecar: Path | None
    detail: str


def apply_visible_watermark(image_bgr: np.ndarray, text: str) -> np.ndarray:
    """Draw a translucent bottom bar and the label. The input array is not mutated."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise WatermarkError(t("watermark.channels"))
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    bar_height = max(32, height // 14)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, height - bar_height), (width, height), (16, 16, 16), thickness=-1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, dst=canvas)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.2, width / 900))
    thickness = 1 if width < 400 else 2
    origin_y = height - max(8, bar_height // 3)
    cv2.putText(canvas, text, (12, origin_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def save_image(
    image_bgr: np.ndarray,
    path: Path,
    *,
    watermark: bool,
    text: str,
) -> None:
    """Write ``image_bgr``. When ``watermark`` is set, embed a visible label and metadata."""

    marked = apply_visible_watermark(image_bgr, text) if watermark else image_bgr
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    suffix = path.suffix.lower()
    if suffix == ".png":
        info = PngImagePlugin.PngInfo()
        if watermark:
            _add_png_text(info, text)
        image.save(path, pnginfo=info)
    elif suffix in {".jpg", ".jpeg"}:
        exif = Image.Exif()
        if watermark:
            exif[0x010E] = text
            exif[0x0131] = SOFTWARE_TAG
            exif[0x9286] = f"AI-generated=true; label={text}".encode("utf-8")
        image.save(path, quality=95, exif=exif.tobytes() if watermark else None)
    elif suffix == ".webp":
        image.save(path)
    elif suffix == ".bmp":
        image.save(path)
    else:
        raise WatermarkError(t("watermark.suffix", suffix=suffix))
    if watermark:
        write_metadata_sidecar(path, text)


def write_metadata_sidecar(path: Path, text: str) -> Path:
    """Write a JSON sidecar next to an output. Used for images and for video."""

    sidecar = path.with_suffix(path.suffix + ".meta.json")
    payload = {
        "ai_generated": True,
        "label": text,
        "software": SOFTWARE_TAG,
        "output": path.name,
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sidecar


def apply_video_metadata(path: Path, text: str) -> VideoMarkResult:
    """Attach container comments when ffmpeg is on PATH, and always write a sidecar.

    Phase 1 refuses to process video frames. This helper is the marking path a
    later video stage will call. It does not invent a transcoder when ffmpeg
    is absent: the sidecar still records the label.
    """

    sidecar = write_metadata_sidecar(path, text)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return VideoMarkResult(
            container_tagged=False,
            sidecar=sidecar,
            detail=t("watermark.ffmpeg_missing"),
        )
    temporary = path.with_suffix(".tagged" + path.suffix)
    import subprocess

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        *video_metadata_args(text),
        "-c",
        "copy",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        return VideoMarkResult(
            container_tagged=False,
            sidecar=sidecar,
            detail=t("watermark.ffmpeg_failed"),
        )
    temporary.replace(path)
    return VideoMarkResult(container_tagged=True, sidecar=sidecar, detail=t("watermark.ffmpeg_ok"))


def video_metadata_args(text: str) -> list[str]:
    """ffmpeg arguments that label a container as synthetic."""

    comment = t("watermark.container_comment", text=text, software=SOFTWARE_TAG)
    return ["-metadata", f"comment={comment}", "-metadata", f"description={text}"]


def _add_png_text(info: PngImagePlugin.PngInfo, text: str) -> None:
    info.add_text("Description", text)
    info.add_text(AI_TEXT_KEY, "true")
    info.add_text("Software", SOFTWARE_TAG)
