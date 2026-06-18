"""Schema documentation stays aligned with code DDL."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_schema_docs_match_code() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "check_schema_docs.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
