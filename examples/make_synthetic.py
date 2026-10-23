"""Write solid-color test images. No real face is generated or downloaded."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

from barrett_faceswap.i18n import activate_language, t


def main() -> None:
    activate_language(cli=None, env=os.environ.get("BARRETT_FACESWAP_LANG"), config=None)
    root = Path(__file__).resolve().parent / "synthetic"
    root.mkdir(parents=True, exist_ok=True)
    _card(root / "source.png", (70, 110, 180), "SOURCE")
    _card(root / "target.png", (180, 120, 70), "TARGET")
    record = {
        "subject_id": "synthetic-demo",
        "statement": "Synthetic test pattern only. No real person is depicted.",
        "consented_on": "2026-01-01",
        "scope": "pipeline-dry-run",
        "expires_on": None,
    }
    (root / "consent.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(t("examples.wrote", path=root))
    print(t("examples.hint"))
    print(
        "  barrett-faceswap swap "
        f"--source {root / 'source.png'} "
        f"--target {root / 'target.png'} "
        f"--output outputs/dry-run.png --dry-run"
    )


def _card(path: Path, color: tuple[int, int, int], label: str) -> None:
    image = Image.new("RGB", (640, 480), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 600, 440), outline=(255, 255, 255), width=4)
    draw.text((48, 48), label, fill=(255, 255, 255))
    image.save(path)


if __name__ == "__main__":
    main()
