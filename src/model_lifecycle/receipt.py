from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def git_state(repo: Path | None = None) -> tuple[str, bool]:
    cwd = repo or Path.cwd()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def environment() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "processor": platform.processor() or "not reported",
        "logical_cpu_count": os.cpu_count(),
    }


def receipt_base(command: str, revision: str, dirty: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_revision": revision,
        "dirty_tree": dirty,
        "command": command,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": environment(),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

