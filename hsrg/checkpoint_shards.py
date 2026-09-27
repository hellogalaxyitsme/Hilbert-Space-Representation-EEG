"""Fingerprint-checked, atomic persistence for restartable checkpoint audits."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def fingerprint(payload: dict[str, Any]) -> str:
    """Return a canonical digest of all result-defining audit inputs."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_verified_shard(
    path: Path,
    expected_fingerprint: str,
    *,
    expected_rows: int | None = None,
    row_keys: set[tuple[Any, ...]] | None = None,
    row_key_fields: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    """Load a completed shard only when its full input fingerprint matches."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("inputs"), dict) or fingerprint(payload["inputs"]) != payload.get("fingerprint"):
        return None
    if payload.get("fingerprint") != expected_fingerprint:
        return None
    if not isinstance(payload.get("rows"), list) or not payload["rows"] or not isinstance(payload.get("check"), dict) or not payload["check"]:
        return None
    if expected_rows is not None and len(payload["rows"]) != expected_rows:
        return None
    if row_keys is not None:
        if not row_key_fields:
            raise ValueError("row_key_fields is required when row_keys is provided")
        if not all(isinstance(row, dict) for row in payload["rows"]):
            return None
        actual = {
            tuple(row.get(field) for field in row_key_fields)
            for row in payload["rows"]
        }
        if len(actual) != len(payload["rows"]) or actual != row_keys:
            return None
    return payload


def write_verified_shard(path: Path, *, inputs: dict[str, Any], rows: list[dict[str, Any]], check: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Atomically persist a complete shard and return its fingerprint."""
    digest = fingerprint(inputs)
    atomic_write_json(path, {"fingerprint": digest, "inputs": inputs, "rows": rows, "check": check, "manifest": manifest})
    return digest
