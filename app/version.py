from __future__ import annotations

import subprocess
from pathlib import Path

VERSION = "0.5.0"
ROOT = Path(__file__).resolve().parent.parent


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    rev = (result.stdout or "").strip()
    return rev or None


REVISION = _git_revision()
DISPLAY = VERSION if not REVISION else f"{VERSION} · {REVISION}"


def version_payload() -> dict[str, str | None]:
    return {
        "version": VERSION,
        "revision": REVISION,
        "display": DISPLAY,
    }
