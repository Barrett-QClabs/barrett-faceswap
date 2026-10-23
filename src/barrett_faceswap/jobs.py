"""Persist a swap job so an interrupted video can resume.

The directory ``jobs/<id>/`` holds ``job.json`` plus, while a video is in
progress, ``raw/`` and ``processed/`` frames. ``next_frame`` is the first
frame index that still needs work. A hard stop leaves the state at
``running``; resume accepts ``pending``, ``running``, and ``interrupted``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from barrett_faceswap.errors import ConfigError
from barrett_faceswap.i18n import t
from barrett_faceswap.types import SwapRequest

_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
RESUMABLE = {"pending", "running", "interrupted"}


@dataclass
class JobState:
    """One queued or running swap."""

    id: str
    state: str
    spec: dict
    next_frame: int = 0
    total_frames: int | None = None
    message: str = ""


class JobStore:
    """Filesystem queue. One JSON file per job."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def directory(self, job_id: str) -> Path:
        check_job_id(job_id)
        return self.root / job_id

    def exists(self, job_id: str) -> bool:
        return (self.directory(job_id) / "job.json").is_file()

    def create(self, job_id: str, request: SwapRequest) -> JobState:
        """Write a pending job. Refuses to overwrite an existing id."""

        check_job_id(job_id)
        path = self.directory(job_id) / "job.json"
        if path.is_file():
            raise ConfigError(t("jobs.exists", job=job_id))
        state = JobState(
            id=job_id,
            state="pending",
            spec=request.model_dump(mode="json"),
            next_frame=0,
        )
        self.save(state)
        return state

    def load(self, job_id: str) -> JobState:
        path = self.directory(job_id) / "job.json"
        if not path.is_file():
            raise ConfigError(t("jobs.missing", job=job_id))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(t("jobs.corrupt", job=job_id, error=exc)) from exc
        return JobState(
            id=str(payload.get("id") or job_id),
            state=str(payload.get("state") or "interrupted"),
            spec=dict(payload.get("spec") or {}),
            next_frame=int(payload.get("next_frame") or 0),
            total_frames=payload.get("total_frames"),
            message=str(payload.get("message") or ""),
        )

    def save(self, state: JobState) -> None:
        path = self.directory(state.id) / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": state.id,
            "state": state.state,
            "spec": state.spec,
            "next_frame": int(state.next_frame),
            "total_frames": state.total_frames,
            "message": state.message,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def list_jobs(self) -> list[JobState]:
        if not self.root.is_dir():
            return []
        jobs = []
        for directory in sorted(self.root.iterdir()):
            if (directory / "job.json").is_file():
                jobs.append(self.load(directory.name))
        return jobs

    def next_runnable(self) -> JobState | None:
        """Oldest pending, running, or interrupted job."""

        for state in self.list_jobs():
            if state.state in RESUMABLE:
                return state
        return None


def check_job_id(job_id: str) -> str:
    """Reject ids that could escape the jobs directory."""

    if not _JOB_ID.match(job_id or ""):
        raise ConfigError(t("jobs.bad_id", job=job_id))
    return job_id


def open_for_run(store: JobStore, request: SwapRequest) -> JobState | None:
    """Create or resume the job named on ``request``. ``None`` when no job is set."""

    if not request.job_id:
        return None
    job_id = check_job_id(request.job_id)
    if store.exists(job_id):
        state = store.load(job_id)
        if state.state == "done":
            raise ConfigError(t("jobs.already_done", job=job_id))
        if state.state in {"running", "interrupted"} and not request.resume:
            raise ConfigError(t("jobs.use_resume", job=job_id))
        if not request.resume:
            state.next_frame = 0
        state.state = "running"
        state.message = ""
        store.save(state)
        return state
    state = JobStore.create(store, job_id, request)
    state.state = "running"
    store.save(state)
    return state


def request_from_job(state: JobState) -> SwapRequest:
    """Rebuild the saved request and mark it as a resume."""

    request = SwapRequest.model_validate(state.spec)
    return request.model_copy(update={"job_id": state.id, "resume": True})
