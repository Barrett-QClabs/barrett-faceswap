"""Source-face authorization records.

A record is either a JSON manifest on disk or an explicit CLI acknowledgement
that names the subject. Both paths produce a stable ``consent_id`` for the
audit log. Neither path is consulted while compliance enforcement is off.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from barrett_faceswap.errors import ConsentError
from barrett_faceswap.i18n import t

_REQUIRED = ("subject_id", "statement", "consented_on", "scope")


@dataclass(frozen=True)
class ConsentRecord:
    """One authorization decision."""

    consent_id: str
    subject_id: str
    statement: str
    consented_on: date
    scope: str
    source: str
    expires_on: date | None = None


def load_consent_file(path: Path) -> ConsentRecord:
    """Parse a consent manifest.

    Required keys: ``subject_id``, ``statement``, ``consented_on``, ``scope``.
    ``expires_on`` is optional. Dates are ``YYYY-MM-DD``.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsentError(t("consent.read", path=path, error=exc)) from exc
    if not isinstance(payload, dict):
        raise ConsentError(t("consent.not_object"))
    missing = [key for key in _REQUIRED if not str(payload.get(key, "")).strip()]
    if missing:
        raise ConsentError(t("consent.missing_fields", fields=", ".join(missing)))
    statement = str(payload["statement"]).strip()
    if len(statement) < 10:
        raise ConsentError(t("consent.statement_short"))
    consented_on = _parse_date(payload["consented_on"], "consented_on")
    expires_raw = payload.get("expires_on")
    expires_on = _parse_date(expires_raw, "expires_on") if expires_raw else None
    if expires_on is not None and expires_on < date.today():
        raise ConsentError(t("consent.expired", day=expires_on.isoformat()))
    subject_id = str(payload["subject_id"]).strip()
    scope = str(payload["scope"]).strip()
    canonical = json.dumps(
        {
            "consented_on": consented_on.isoformat(),
            "expires_on": expires_on.isoformat() if expires_on else None,
            "scope": scope,
            "statement": statement,
            "subject_id": subject_id,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return ConsentRecord(
        consent_id=f"file:{digest}",
        subject_id=subject_id,
        statement=statement,
        consented_on=consented_on,
        scope=scope,
        source=str(path),
        expires_on=expires_on,
    )


def consent_from_acknowledgement(
    subject_id: str | None,
    statement: str | None,
    *,
    today: date | None = None,
) -> ConsentRecord:
    """Build a record from the explicit CLI acknowledgement."""

    subject = (subject_id or "").strip()
    if not subject:
        raise ConsentError(t("consent.need_subject"))
    text = (statement or t("consent.default_ack")).strip()
    if len(text) < 10:
        raise ConsentError(t("consent.statement_short"))
    day = today or date.today()
    digest = hashlib.sha256(f"{subject}|{text}|{day.isoformat()}".encode("utf-8")).hexdigest()[:16]
    return ConsentRecord(
        consent_id=f"ack:{digest}",
        subject_id=subject,
        statement=text,
        consented_on=day,
        scope="cli-acknowledgement",
        source="cli",
        expires_on=None,
    )


def _parse_date(value: object, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ConsentError(t("consent.bad_date", field=field_name)) from exc
