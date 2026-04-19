"""Read-only DuckDB connection helpers for the website."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Iterable

import duckdb


_DB_PATH_ENV = "CHGK_DB_PATH"
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "chgk.duckdb"

_lock = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None


def db_path() -> Path:
    """Resolve the DuckDB path: env var > default."""
    p = os.environ.get(_DB_PATH_ENV)
    return Path(p) if p else _DEFAULT_PATH


def get_conn() -> duckdb.DuckDBPyConnection:
    """
    Return a process-wide read-only DuckDB connection.

    DuckDB connections are not thread-safe for concurrent writes, but
    multiple read cursors via .cursor() are safe. We use a single
    read-only connection and create a per-call cursor for queries.
    """
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                path = db_path()
                if not path.exists():
                    raise FileNotFoundError(
                        f"chgk.duckdb not found at {path}. "
                        f"Build it with `python website/build/build_db.py`."
                    )
                _conn = duckdb.connect(str(path), read_only=True)
    return _conn


def reload_conn() -> dict:
    """
    Close and re-open the DuckDB connection.

    Used after `scripts/refresh_data.sh` swaps the file on disk: the
    running process keeps the previous file open via its fd, so a
    reload is required for new data to become visible. Returns a small
    dict with the new file size and mtime so the caller can confirm
    the swap happened.
    """
    global _conn
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
        path = db_path()
        if not path.exists():
            raise FileNotFoundError(f"chgk.duckdb not found at {path}.")
        _conn = duckdb.connect(str(path), read_only=True)
        st = path.stat()
        return {
            "path": str(path),
            "size_bytes": int(st.st_size),
            "mtime": st.st_mtime,
        }


def query(sql: str, params: Iterable[Any] | None = None) -> list[dict]:
    """Run a SQL query and return list of dicts."""
    cur = get_conn().cursor()
    if params is not None:
        cur.execute(sql, list(params))
    else:
        cur.execute(sql)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def query_one(sql: str, params: Iterable[Any] | None = None) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None
