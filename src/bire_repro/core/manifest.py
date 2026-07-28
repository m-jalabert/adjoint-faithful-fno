"""Checksummed, append-only provenance-manifest utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": sha256_file(path)}


def tree_records(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    return [file_record(path) for path in sorted(root.rglob("*")) if path.is_file()]


def runtime_record() -> dict[str, Any]:
    def command_output(command: list[str]) -> str | None:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": command_output(["git", "status", "--porcelain"]),
    }


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically create a manifest; refuse to overwrite provenance."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"manifest already exists: {path}")
    complete = {"runtime": runtime_record(), **payload}
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(complete, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def verify_records(records: Iterable[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            errors.append(f"missing: {path}")
        elif path.stat().st_size != record["bytes"]:
            errors.append(f"size mismatch: {path}")
        elif sha256_file(path) != record["sha256"]:
            errors.append(f"checksum mismatch: {path}")
    return errors
