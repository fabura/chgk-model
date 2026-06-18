#!/usr/bin/env python3
"""Check that docs/schema/*.md document tables from code DDL.

Exit 0 if OK, 1 if any documented table is missing from markdown.
Run: python scripts/check_schema_docs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (code file relative to ROOT, constant name, markdown doc relative to ROOT)
DDL_BINDINGS: list[tuple[str, str, str]] = [
    ("website/build/build_db.py", "DDL", "docs/schema/duckdb.md"),
    ("website/build/map_tables.py", "MAP_DDL", "docs/schema/duckdb.md"),
    ("venue_overlay/store.py", "DDL", "docs/schema/venue-overlay.md"),
    ("rating_api/pg_state.py", "DDL", "docs/schema/api-overlay.md"),
]

CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:[\w.]+\.)?(\w+)",
    re.IGNORECASE,
)


def _extract_triple_quoted(source: str, const_name: str) -> str:
    """Return the first triple-quoted string assigned to ``const_name``."""
    pattern = re.compile(
        rf"{re.escape(const_name)}\s*=\s*\"\"\"(.*?)\"\"\"",
        re.DOTALL,
    )
    m = pattern.search(source)
    if not m:
        raise ValueError(f"constant {const_name!r} not found")
    return m.group(1)


def tables_in_ddl(source: str, const_name: str) -> set[str]:
    ddl = _extract_triple_quoted(source, const_name)
    return {m.group(1) for m in CREATE_TABLE_RE.finditer(ddl)}


def check_ddl_bindings() -> list[str]:
    errors: list[str] = []
    doc_tables: dict[str, set[str]] = {}

    for rel_path, const_name, doc_rel in DDL_BINDINGS:
        code = (ROOT / rel_path).read_text(encoding="utf-8")
        doc_path = ROOT / doc_rel
        doc_text = doc_path.read_text(encoding="utf-8")
        tables = tables_in_ddl(code, const_name)
        doc_tables.setdefault(doc_rel, set()).update(tables)

    for doc_rel, tables in doc_tables.items():
        doc_text = (ROOT / doc_rel).read_text(encoding="utf-8")
        for table in sorted(tables):
            # Require markdown heading or backtick mention
            if f"`{table}`" not in doc_text and f"### `{table}`" not in doc_text:
                if f"### {table}" not in doc_text and f"| `{table}`" not in doc_text:
                    if table not in doc_text:
                        errors.append(
                            f"{doc_rel}: table {table!r} in code DDL but not mentioned in doc"
                        )
    return errors


def check_cache_version() -> list[str]:
    errors: list[str] = []
    data_py = (ROOT / "data.py").read_text(encoding="utf-8")
    cache_md = (ROOT / "docs/schema/cache.md").read_text(encoding="utf-8")
    m = re.search(r"CACHE_VERSION_NPZ\s*=\s*(\d+)", data_py)
    if not m:
        errors.append("data.py: CACHE_VERSION_NPZ not found")
        return errors
    version = m.group(1)
    if f"CACHE_VERSION_NPZ = {version}" not in cache_md and f"= {version}" not in cache_md:
        errors.append(
            f"docs/schema/cache.md: expected CACHE_VERSION_NPZ {version} "
            f"(from data.py)"
        )
    return errors


def main() -> int:
    errors = check_ddl_bindings() + check_cache_version()
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("schema docs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
