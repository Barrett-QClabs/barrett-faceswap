"""Download the swap weights with sha256 checks, resume, and mirror fallback.

CI and the test suite must not call this against the network. Tests pass an
``opener``. The zip from the buffalo_l release is checked by the hashes of
``det_10g.onnx`` and ``w600k_r50.onnx``, not by a hash of the archive, so a
valid re-pack is accepted.
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import shutil
import socket
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from barrett_faceswap.config import AppConfig
from barrett_faceswap.errors import BarrettFaceswapError
from barrett_faceswap.i18n import t
from barrett_faceswap.models import file_sha256

logger = logging.getLogger(__name__)

# Published hashes. A file that does not match is rejected.
# xseg_1.onnx and bisenet_resnet_34.onnx are the facefusion-assets names.
# They are stored under this repo's mask filenames. The license of those
# files is unverified and likely non-commercial.
XSEG_SHA256 = "c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3"
PARSER_SHA256 = "4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96"

KNOWN_SHA256 = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    "inswapper_128.onnx": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
    "gfpgan_1.4.onnx": "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385",
    "occlusion_xseg.onnx": XSEG_SHA256,
    "xseg_1.onnx": XSEG_SHA256,
    "bisenet_face_parsing.onnx": PARSER_SHA256,
    "bisenet_resnet_34.onnx": PARSER_SHA256,
}

# Basename in a --from-dir tree -> the name this repo stores.
_IMPORT_ALIASES = {
    "xseg_1.onnx": "occlusion_xseg.onnx",
    "bisenet_resnet_34.onnx": "bisenet_face_parsing.onnx",
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
XSEG_URLS = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx",
    "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx",
)
PARSER_URLS = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx",
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx",
)

# sha256 of the facefusion-assets ONNX bytes. CI never fetches these.
_EXTRA_SHA256 = {
    "blendswap_256.onnx": "20ac5e3c002288e7abd4ea6b1af74103892a6628382d1fa905d13c0270c2667b",
    "simswap_256.onnx": "5b6374b2ffab838588d4f0c32dc7ed5bb8d99237d755a663ed02ee265a1714fb",
    "simswap_unofficial_512.onnx": "fe805d1ce7d9e66322e2a8811f593a821e7d92f9ff861dd233794bdb2bb7a586",
    "ghost_1_256.onnx": "4eac28021ae80128e9262080cf0da850e3190e41862650c9e805f9a516b3e924",
    "ghost_2_256.onnx": "27ae9d0b174b22c0426393e8239ca8d75e6ff47ec5dc596b002b182832a9454e",
    "ghost_3_256.onnx": "308d87f565e881b8872a5cbe711f97faeda6643e5d2b95ef757cc92a58662abd",
    "hyperswap_1a_256.onnx": "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add",
    "hyperswap_1b_256.onnx": "5124031789c42f71b9558fb71954ef7aedb6da7ed9fac79293e23c61a792a73e",
    "hyperswap_1c_256.onnx": "5528c2d76fe9986c99d829278987ef9f3a630cb606db7628d02b57b330f406a5",
    "crossface_simswap.onnx": "6452a261ec30cc30afdbe4a426d82c3b10a476f2df794e3494071c02574e6829",
    "crossface_ghost.onnx": "9ec5862d9ff1f723a7380ea89baf87cddec9c56670d4db766702657939284957",
    "codeformer.onnx": "21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f",
    "gpen_bfr_512.onnx": "d5f066b9068a8b74217f9712e28e875a6144629b108a6f7355acbdb3a2832c54",
    "gpen_bfr_1024.onnx": "bcd31aa52110a2005efc96abbab4546d57e42482648f08715b552423d96b381b",
    "gpen_bfr_2048.onnx": "66d12a637118d71b00f5a290b8dac13c81c1c0326a127a1978799c6e17bb8d1f",
    "restoreformer_plus_plus.onnx": "164818b70d1745da4108a45c9263b05452236d14ac0ffab2202af0ac4a438195",
    "xseg_2.onnx": "cd9a0879eaf43841d765472cf1f8c330dbf9dcb03da0eace93e95f3bcc399042",
    "xseg_3.onnx": "48ccd7e8541e159a5a754ec9e62df2f12065f7df8f9af842c1750342c6533559",
    "2dfan4.onnx": "678c6fa539d52335a31c980feefdf4a6e02d781d83dce00af8a894f114557285",
}
KNOWN_SHA256.update(_EXTRA_SHA256)


def _assets(version: str, filename: str) -> tuple[str, ...]:
    return (
        f"https://github.com/facefusion/facefusion-assets/releases/download/{version}/{filename}",
        f"https://huggingface.co/facefusion/{version}/resolve/main/{filename}",
    )


# name -> (release, filename, config role, companion role or None)
_SWAPPER_FILES = {
    "inswapper_128": ("models-3.0.0", "inswapper_128.onnx", "swapper", None),
    "hyperswap_1a_256": ("models-3.3.0", "hyperswap_1a_256.onnx", "hyperswap_1a", None),
    "hyperswap_1b_256": ("models-3.3.0", "hyperswap_1b_256.onnx", "hyperswap_1b", None),
    "hyperswap_1c_256": ("models-3.3.0", "hyperswap_1c_256.onnx", "hyperswap_1c", None),
    "simswap_256": ("models-3.0.0", "simswap_256.onnx", "simswap_256", "crossface_simswap"),
    "simswap_unofficial_512": ("models-3.0.0", "simswap_unofficial_512.onnx", "simswap_512", "crossface_simswap"),
    "blendswap_256": ("models-3.0.0", "blendswap_256.onnx", "blendswap_256", None),
    "ghost_1_256": ("models-3.0.0", "ghost_1_256.onnx", "ghost_1", "crossface_ghost"),
    "ghost_2_256": ("models-3.0.0", "ghost_2_256.onnx", "ghost_2", "crossface_ghost"),
    "ghost_3_256": ("models-3.0.0", "ghost_3_256.onnx", "ghost_3", "crossface_ghost"),
}
_ENHANCER_FILES = {
    "gfpgan_1.4": ("models-3.0.0", "gfpgan_1.4.onnx", "gfpgan"),
    "codeformer": ("models-3.0.0", "codeformer.onnx", "codeformer"),
    "gpen_bfr_512": ("models-3.0.0", "gpen_bfr_512.onnx", "gpen_bfr_512"),
    "gpen_bfr_1024": ("models-3.0.0", "gpen_bfr_1024.onnx", "gpen_bfr_1024"),
    "gpen_bfr_2048": ("models-3.0.0", "gpen_bfr_2048.onnx", "gpen_bfr_2048"),
    "restoreformer_plus_plus": ("models-3.0.0", "restoreformer_plus_plus.onnx", "restoreformer_plus_plus"),
}
_OCCLUDER_FILES = {
    "xseg_1": ("models-3.1.0", "xseg_1.onnx", "occluder", "occlusion_xseg.onnx"),
    "xseg_2": ("models-3.1.0", "xseg_2.onnx", "xseg_2", "xseg_2.onnx"),
    "xseg_3": ("models-3.2.0", "xseg_3.onnx", "xseg_3", "xseg_3.onnx"),
}
_COMPANION_FILES = {
    "crossface_simswap": ("models-3.4.0", "crossface_simswap.onnx"),
    "crossface_ghost": ("models-3.4.0", "crossface_ghost.onnx"),
}

Opener = Callable[[str, dict[str, str]], object]
_ATTEMPTS = 3
CONNECT_TIMEOUT_S = 15
READ_TIMEOUT_S = 30


def download_models(
    config: AppConfig,
    *,
    with_gfpgan: bool = False,
    with_masks: bool = False,
    force: bool = False,
    opener: Opener | None = None,
    pins: dict[str, str] | None = None,
    attempts: int = _ATTEMPTS,
    from_dir: Path | None = None,
    swappers: list[str] | None = None,
    enhancers: list[str] | None = None,
    occluders: list[str] | None = None,
    with_landmarks: bool = False,
) -> list[str]:
    """Fetch buffalo_l and inswapper. GFPGAN and the mask graphs are optional.

    ``pins`` overrides :data:`KNOWN_SHA256` in tests. ``opener(url, headers)``
    replaces ``urllib`` so tests never touch the network.
    """

    expected = dict(KNOWN_SHA256)
    if pins:
        expected.update(pins)
    notes: list[str] = []
    if from_dir is not None:
        notes.extend(
            _import_from_dir(
                Path(from_dir),
                config,
                expected,
            with_gfpgan=with_gfpgan,
            with_masks=with_masks,
            force=force,
            swappers=swappers,
            enhancers=enhancers,
            occluders=occluders,
            with_landmarks=with_landmarks,
        )
    )
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
    if with_landmarks:
        notes.extend(
            _download_file(
                _assets("models-3.0.0", "2dfan4.onnx"),
                config.models.landmark.path,
                expected["2dfan4.onnx"],
                force=force,
                opener=opener,
                attempts=attempts,
                name="2dfan4.onnx",
            )
        )
    if with_masks:
        notes.extend(
            _download_file(
                XSEG_URLS,
                config.models.occluder.path,
                expected["occlusion_xseg.onnx"],
                force=force,
                opener=opener,
                attempts=attempts,
                name="xseg_1.onnx",
            )
        )
        notes.extend(
            _download_file(
                PARSER_URLS,
                config.models.parser.path,
                expected["bisenet_face_parsing.onnx"],
                force=force,
                opener=opener,
                attempts=attempts,
                name="bisenet_resnet_34.onnx",
            )
        )
    notes.extend(
        _download_selected(
            config,
            expected,
            swappers=list(swappers or []),
            enhancers=list(enhancers or []),
            occluders=list(occluders or []),
            force=force,
            opener=opener,
            attempts=attempts,
        )
    )
    return notes


def _download_selected(
    config: AppConfig,
    expected: dict[str, str],
    *,
    swappers: list[str],
    enhancers: list[str],
    occluders: list[str],
    force: bool,
    opener: Opener | None,
    attempts: int,
) -> list[str]:
    notes: list[str] = []
    for name in swappers:
        if name == "inswapper_128":
            continue
        version, filename, role, companion = _SWAPPER_FILES[name]
        notes.extend(
            _download_file(
                _assets(version, filename),
                getattr(config.models, role).path,
                expected[filename],
                force=force,
                opener=opener,
                attempts=attempts,
                name=filename,
            )
        )
        if companion:
            companion_version, companion_file = _COMPANION_FILES[companion]
            notes.extend(
                _download_file(
                    _assets(companion_version, companion_file),
                    getattr(config.models, companion).path,
                    expected[companion_file],
                    force=force,
                    opener=opener,
                    attempts=attempts,
                    name=companion_file,
                )
            )
    for name in enhancers:
        if name == "gfpgan_1.4":
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
            continue
        version, filename, role = _ENHANCER_FILES[name]
        notes.extend(
            _download_file(
                _assets(version, filename),
                getattr(config.models, role).path,
                expected[filename],
                force=force,
                opener=opener,
                attempts=attempts,
                name=filename,
            )
        )
    for name in occluders:
        if name == "sam2_refine":
            notes.append(t("download.sam2_local", path=config.models.sam2_refine.path))
            name = "xseg_1"
        if name == "xseg_1":
            notes.extend(
                _download_file(
                    XSEG_URLS,
                    config.models.occluder.path,
                    expected["occlusion_xseg.onnx"],
                    force=force,
                    opener=opener,
                    attempts=attempts,
                    name="xseg_1.onnx",
                )
            )
            continue
        version, filename, role, stored = _OCCLUDER_FILES[name]
        notes.extend(
            _download_file(
                _assets(version, filename),
                getattr(config.models, role).path,
                expected[stored],
                force=force,
                opener=opener,
                attempts=attempts,
                name=filename,
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
    request = urllib.request.Request(url, headers=headers)
    opener_obj = urllib.request.build_opener(_TimeoutHTTPHandler, _TimeoutHTTPSHandler)
    return opener_obj.open(request, timeout=READ_TIMEOUT_S)


class _ConnectHTTP(http.client.HTTPConnection):
    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), CONNECT_TIMEOUT_S, self.source_address)
        if self.sock is not None:
            self.sock.settimeout(READ_TIMEOUT_S)


class _ConnectHTTPS(http.client.HTTPSConnection):
    def connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), CONNECT_TIMEOUT_S, self.source_address)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        self.sock.settimeout(READ_TIMEOUT_S)


class _TimeoutHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_ConnectHTTP, req)


class _TimeoutHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_ConnectHTTPS, req, context=self._context)


def _import_from_dir(
    directory: Path,
    config: AppConfig,
    expected: dict[str, str],
    *,
    with_gfpgan: bool,
    with_masks: bool,
    force: bool,
    swappers: list[str] | None = None,
    enhancers: list[str] | None = None,
    occluders: list[str] | None = None,
    with_landmarks: bool = False,
) -> list[str]:
    """Copy matching files by basename. A bad hash fails before any download."""

    if not directory.is_dir():
        raise BarrettFaceswapError(t("download.not_dir", path=directory))
    wanted = {
        "det_10g.onnx": (config.models.detector.path, expected["det_10g.onnx"]),
        "w600k_r50.onnx": (config.models.recognizer.path, expected["w600k_r50.onnx"]),
        "inswapper_128.onnx": (config.models.swapper.path, expected["inswapper_128.onnx"]),
    }
    if with_gfpgan:
        wanted["gfpgan_1.4.onnx"] = (config.models.gfpgan.path, expected["gfpgan_1.4.onnx"])
    if with_masks:
        wanted["occlusion_xseg.onnx"] = (config.models.occluder.path, expected["occlusion_xseg.onnx"])
        wanted["bisenet_face_parsing.onnx"] = (config.models.parser.path, expected["bisenet_face_parsing.onnx"])
    if with_landmarks:
        wanted["2dfan4.onnx"] = (config.models.landmark.path, expected["2dfan4.onnx"])
    for name in swappers or []:
        if name == "inswapper_128":
            continue
        _version, filename, role, companion = _SWAPPER_FILES[name]
        wanted[filename] = (getattr(config.models, role).path, expected[filename])
        if companion:
            _companion_version, companion_file = _COMPANION_FILES[companion]
            wanted[companion_file] = (getattr(config.models, companion).path, expected[companion_file])
    for name in enhancers or []:
        if name == "gfpgan_1.4":
            continue
        _version, filename, role = _ENHANCER_FILES[name]
        wanted[filename] = (getattr(config.models, role).path, expected[filename])
    for name in occluders or []:
        if name in {"sam2_refine", "xseg_1"}:
            continue
        _version, filename, role, stored = _OCCLUDER_FILES[name]
        wanted[filename] = (getattr(config.models, role).path, expected[stored])
    found: dict[str, Path] = {}
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        canonical = _IMPORT_ALIASES.get(path.name, path.name)
        if canonical not in wanted:
            continue
        current = found.get(canonical)
        # Prefer the stored basename over an upstream alias when both are present.
        if current is None or path.name == canonical:
            found[canonical] = path
    notes: list[str] = []
    for name, (dest, digest) in wanted.items():
        source = found.get(name)
        if source is None:
            continue
        pin = digest.lower()
        if dest.is_file() and not force and file_sha256(dest) == pin:
            notes.append(t("download.skip", path=dest, sha256=pin))
            continue
        actual = file_sha256(source)
        if actual != pin:
            raise BarrettFaceswapError(t("download.mismatch", name=name, expected=pin, actual=actual))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        notes.append(t("download.imported", path=dest, sha256=actual))
    members = {
        "det_10g.onnx": wanted["det_10g.onnx"],
        "w600k_r50.onnx": wanted["w600k_r50.onnx"],
    }
    pending = {
        name: pair
        for name, pair in members.items()
        if not pair[0].is_file() or file_sha256(pair[0]) != pair[1].lower()
    }
    if pending:
        archive = next((path for path in directory.rglob("buffalo_l.zip") if path.is_file()), None)
        if archive is not None:
            notes.extend(_extract_members(archive, pending))
    return notes


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
