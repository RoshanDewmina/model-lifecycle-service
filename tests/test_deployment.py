from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_root_asgi_wrapper_imports() -> None:
    repository = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from app import app; "
                "assert {'/', '/health', '/predict', '/evidence'} <= "
                "{route.path for route in app.routes}"
            ),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
