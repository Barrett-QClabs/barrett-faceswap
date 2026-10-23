"""Download the swap weights with sha256 checks, resume, and mirror fallback.

CI and the test suite must not call this against the network. Tests pass an
``opener``. The zip from the buffalo_l release is checked by the hashes of
``det_10g.onnx`` and ``w600k_r50.onnx``, not by a hash of the archive, so a
valid re-pack is accepted.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from pathlib import Path
from typing import Callable

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import BarrettFaceswapError
from barrett_faceswap.i18n import t
from barrett_faceswap.models import file_sha256

logger = logging.getLogger(__name__)

# Published hashes. A file that does not match is rejected.
KNOWN_SHA256 = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    "inswapper_128.onnx": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
    "gfpgan_1.4.onnx": "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385",
}

BUFFALO_URLS = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip",
)
INSWAPPER_URLS = (
    "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128.onnx",
)
GFPGAN_URLS = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx",
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
)

Opener = Callable[[str, dict[str, str]], object]
_ATTEMPTS = 3


def download_models(
    config: AppConfig,
    *,
    with_gfpgan: bool = False,
    force: bool = False,
    opener: Opener | None = None,
    pins: dict[str, str] | None = None,
    attempts: int = _ATTEMPTS,
) -> list[str]:
    """Fetch buffalo_l and inswapper. GFPGAN ONNX is optional.

    ``pins`` overrides :data:`KNOWN_SHA256` in tests. ``opener(url, headers)``
    replaces ``urllib`` so tests never touch the network.
    """

    expected = dict(KNOWN_SHA256)
    if pins:
        expected.update(pins)
    notes: list[str] = []
    detector = config.models.detector.path
    recognizer = config.models.recognizer.path
    members = {
        "det_10g.onnx": (detector, expected["det_10g.onnx"]),
        "w600k_r50.onnx": (recognizer, expected["w600k_r50.onnx"]),
    }
    scratch = detector.parent / "buffalo_l.zip.partial"
    notes.extend(
        _download_zip(
            BUFFALO_URLS,
            scratch,
            members,
            force=force,
            opener=opener,
            attempts=attempts,
            name="buffalo_l.zip",
        )
    )
    notes.extend(
        _download_file(
            INSWAPPER_URLS,
            config.models.swapper.path,
            expected["inswapper_128.onnx"],
            force=force,
            opener=opener,
            attempts=attempts,
            name="inswapper_128.onnx",
        )
    )
    if with_gfpgan:
        notes.extend(
            _download_file(
                GFPGAN_URLS,
                config.models.gfpgan.path,
                expected["gfpgan_1.4.onnx"],
                force=force,
                opener=opener,
                attempts=attempts,
                name="gfpgan_1.4.onnx",
            )
        )
    return notes


def _download_file(
    urls: tuple[str, ...],
    dest: Path,
    expected: str,
    *,
    force: bool,
    opener: Opener | None,
    attempts: int,
    name: str,
) -> list[str]:
    if dest.is_file() and not force:
        actual = file_sha256(dest)
        if actual == expected.lower():
            return [t("download.skip", path=dest, sha256=actual)]
    partial = Path(str(dest) + ".partial")
    return [
        _fetch_verified(
            urls,
            partial,
            dest,
            expected.lower(),
            opener=opener,
            attempts=attempts,
            name=name,
        )
    ]


def _download_zip(
    urls: tuple[str, ...],
    scratch: Path,
    members: dict[str, tuple[Path, str]],
    *,
    force: bool,
    opener: Opener | None,
    attempts: int,
    name: str,
) -> list[str]:
    if not force and _members_match(members):
        notes = []
        for dest, digest in ((path, pin) for path, pin in members.values()):
            notes.append(t("download.skip", path=dest, sha256=digest.lower()))
        return notes
    last_error = ""
    for url in urls:
        for _attempt in range(attempts):
            try:
                logger.info(t("download.fetch", name=name, url=url))
                _fetch_once(url, scratch, opener)
                notes = _extract_members(scratch, members)
                scratch.unlink(missing_ok=True)
                return notes
            except Exception as exc:
                last_error = str(exc)
                logger.info(t("download.retry", url=url, error=exc))
                if isinstance(exc, BarrettFaceswapError):
                    scratch.unlink(missing_ok=True)
    raise BarrettFaceswapError(t("download.exhausted", name=name, error=last_error))


def _fetch_verified(
    urls: tuple[str, ...],
    partial: Path,
    dest: Path,
    expected: str,
    *,
    opener: Opener | None,
    attempts: int,
    name: str,
) -> str:
    last_error = ""
    for url in urls:
        for _attempt in range(attempts):
            try:
                logger.info(t("download.fetch", name=name, url=url))
                _fetch_once(url, partial, opener)
                digest = file_sha256(partial)
                if digest != expected:
                    partial.unlink(missing_ok=True)
                    raise BarrettFaceswapError(
                        t("download.mismatch", name=name, expected=expected, actual=digest)
                    )
                dest.parent.mkdir(parents=True, exist_ok=True)
                partial.replace(dest)
                return t("download.wrote", path=dest, sha256=digest)
            except Exception as exc:
                last_error = str(exc)
                logger.info(t("download.retry", url=url, error=exc))
                if isinstance(exc, BarrettFaceswapError):
                    partial.unlink(missing_ok=True)
    raise BarrettFaceswapError(t("download.exhausted", name=name, error=last_error))


def _fetch_once(url: str, partial: Path, opener: Opener | None) -> None:
    """Download ``url`` into ``partial``. A 206 response appends; anything else rewrites."""

    existing = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "barrett-faceswap"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    response = _open(url, headers, opener)
    try:
        status = int(getattr(response, "status", None) or getattr(response, "code", 200))
        mode = "ab" if status == 206 and existing else "wb"
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()


def _open(url: str, headers: dict[str, str], opener: Opener | None) -> object:
    if opener is not None:
        return opener(url, headers)
    import urllib.request

    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=120)


def _members_match(members: dict[str, tuple[Path, str]]) -> bool:
    for dest, expected in members.values():
        if not dest.is_file() or file_sha256(dest) != expected.lower():
            return False
    return True


def _extract_members(archive_path: Path, members: dict[str, tuple[Path, str]]) -> list[str]:
    notes: list[str] = []
    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as exc:
        raise BarrettFaceswapError(t("download.exhausted", name=archive_path.name, error=exc)) from exc
    with archive:
        chosen: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            base = Path(info.filename).name
            if base in members and base not in chosen and not info.is_dir():
                chosen[base] = info
        for base, (dest, expected) in members.items():
            info = chosen.get(base)
            if info is None:
                raise BarrettFaceswapError(
                    t("download.member_missing", name=archive_path.name, member=base)
                )
            data = archive.read(info)
            digest = hashlib.sha256(data).hexdigest()
            if digest != expected.lower():
                raise BarrettFaceswapError(
                    t("download.mismatch", name=base, expected=expected.lower(), actual=digest)
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            notes.append(t("download.wrote", path=dest, sha256=digest))
    return notes
