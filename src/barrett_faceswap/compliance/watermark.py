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
    """How a finished video file was labeled."""

    container_tagged: bool
    sidecar: Path | None
    detail: str


def apply_visible_watermark(image_bgr: np.ndarray, text: str, font_path: Path | None = None) -> np.ndarray:
    """Draw a translucent bottom bar and the label. The input array is not mutated."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise WatermarkError(t("watermark.channels"))
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    bar_height = max(32, height // 14)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, height - bar_height), (width, height), (16, 16, 16), thickness=-1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, dst=canvas)
    _draw_label(canvas, text or "", bar_height, font_path)
    return canvas


def _draw_label(canvas: np.ndarray, text: str, bar_height: int, font_path: Path | None) -> None:
    from barrett_faceswap.glyphs import GLYPHS, blit, covers

    height, width = canvas.shape[:2]
    if text and not covers(text):
        if font_path is not None and Path(font_path).is_file() and _draw_truetype(canvas, text, font_path, bar_height):
            return
        logger_note = t("watermark.font_missing", path=font_path or "-")
        import logging

        logging.getLogger(__name__).info(logger_note)
        text = "".join(char for char in text if char in GLYPHS or ord(char) < 128)
    origin_y = height - bar_height + max(4, (bar_height - 12) // 2)
    cursor = 12
    limit = max(24, width // 2 - 8)
    scale = 1 if cursor + len(text) * 13 <= limit else 1
    for char in text:
        if cursor >= limit:
            break
        glyph = GLYPHS.get(char)
        if glyph is not None:
            blit(canvas, glyph, cursor, origin_y, scale, (255, 255, 255))
            cursor += (glyph.shape[1] + 2) * scale
            continue
        if ord(char) >= 128:
            continue
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        cv_scale = 0.45
        cv2.putText(canvas, char, (cursor, origin_y + 12), font, cv_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        (tw, _th), _baseline = cv2.getTextSize(char, font, cv_scale, thickness)
        cursor += max(tw, 4) + 1


def _draw_truetype(canvas: np.ndarray, text: str, font_path: Path, bar_height: int) -> bool:
    """Draw ``text`` with a caller-supplied font. False when the font cannot be read."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    try:
        font = ImageFont.truetype(str(font_path), size=max(14, bar_height // 2))
    except OSError:
        return False
    height, width = canvas.shape[:2]
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.text((12, height - bar_height + 4), text, fill=(255, 255, 255), font=font)
    canvas[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    del width
    return True


def save_image(
    image_bgr: np.ndarray,
    path: Path,
    *,
    watermark: bool,
    text: str,
    font_path: Path | None = None,
) -> None:
    """Write ``image_bgr``. When ``watermark`` is set, embed a visible label and metadata."""

    marked = apply_visible_watermark(image_bgr, text, font_path=font_path) if watermark else image_bgr
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

    The video pipeline calls this after frames are encoded. It does not invent
    a transcoder when ffmpeg is absent: the sidecar still records the label.
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
