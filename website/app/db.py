"""Read-only DuckDB connection helpers for the website."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

import duckdb
import numpy as np


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
    global _conn, _model_params_cache
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
        _model_params_cache = None
        # Drop the API cache too; the model_params and tournament/player
        # tables can shift between builds, and stale rosters during a
        # build swap would mismatch the now-current θ values.
        try:
            from . import forecast_api as _fa
            _fa.clear_cache()
        except Exception:
            pass
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


_model_params_cache: Optional[dict] = None


def get_model_params() -> dict:
    """
    Return model auxiliary parameters needed to predict probabilities at
    runtime (``delta_size``, ``team_size_anchor``, ``delta_pos``,
    ``pos_anchor``, ``lapse``, ``recal``) as numpy arrays / Python ints.

    Loaded once and cached for the lifetime of the connection; cleared
    whenever ``reload_conn()`` is called so a hot DB swap picks up new
    values.  Missing arrays come back as ``None`` (identity calibration
    is then used by the simulation kernel).
    """
    global _model_params_cache
    if _model_params_cache is not None:
        return _model_params_cache
    try:
        row = query_one("SELECT params FROM model_params LIMIT 1")
    except Exception:
        row = None
    raw = (row or {}).get("params")
    if raw is None:
        _model_params_cache = {
            "delta_size": None,
            "team_size_anchor": None,
            "delta_pos": None,
            "pos_anchor": None,
            "lapse": None,
            "recal": None,
        }
        return _model_params_cache
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    def _arr(x):
        return None if x is None else np.asarray(x, dtype=np.float64)
    _model_params_cache = {
        "delta_size": _arr(payload.get("delta_size")),
        "team_size_anchor": payload.get("team_size_anchor"),
        "delta_pos": _arr(payload.get("delta_pos")),
        "pos_anchor": payload.get("pos_anchor"),
        "lapse": _arr(payload.get("lapse")),
        "recal": _arr(payload.get("recal")),
    }
    return _model_params_cache


def get_site_meta() -> dict | None:
    """
    Single-row footer metadata from ``site_meta`` (written by ``build_db``).

    Returns ``None`` if the table is missing (pre-migration DuckDB) or empty.
    """
    try:
        return query_one(
            "SELECT CAST(data_as_of AS VARCHAR) AS data_as_of_iso, "
            "       model_built_at "
            "FROM site_meta LIMIT 1"
        )
    except Exception:
        return None
