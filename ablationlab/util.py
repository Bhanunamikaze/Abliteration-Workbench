"""Deterministic serialization, atomic files and reproducibility helpers."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

class LabError(RuntimeError):
    """An actionable configuration/data/capability failure."""

class BudgetReached(LabError):
    pass

class Unsupported(LabError):
    pass

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, str):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def dumps(value: Any, pretty: bool = False) -> str:
    return json.dumps(clean(value), sort_keys=True, ensure_ascii=False,
                      indent=2 if pretty else None, allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()

def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def write_json(path: str | Path, value: Any) -> None:
    atomic_text(path, dumps(value, pretty=True) + "\n")

def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8-sig") as f:
        return json.load(f)

def read_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        return read_journal(p)
    value = read_json(p)
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise LabError(f"Expected an array of objects: {p}")
    return value

def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    import io
    rows = [clean(r) for r in rows]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = io.StringIO(newline="")
    w = csv.DictWriter(out, fieldnames=keys)
    w.writeheader()
    for row in rows:
        w.writerow({k: dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    atomic_text(path, out.getvalue())

def append_jsonl(path: str | Path, value: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(dumps(value) + "\n")
        f.flush()
        os.fsync(f.fileno())

def read_journal(path: str | Path, repair_tail: bool = False, ignore_tail: bool = False) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    data = p.read_bytes()
    lines = data.splitlines(keepends=True)
    out, offset = [], 0
    for i, line in enumerate(lines):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("journal rows must be objects")
        except (ValueError, UnicodeDecodeError):
            if (repair_tail or ignore_tail) and i == len(lines) - 1:
                if repair_tail:
                    with p.open("r+b") as f:
                        f.truncate(offset)
                break
            raise LabError(f"Invalid JSONL row {i+1} in {p}; not silently skipped")
        out.append(value)
        offset += len(line)
        if repair_tail and i == len(lines)-1 and not line.endswith(b"\n"):
            with p.open("ab") as f: f.write(b"\n")
    return out

def as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if str(v).lower() in {"true", "1"}:
        return True
    if str(v).lower() in {"false", "0"}:
        return False
    raise LabError(f"Not a boolean: {v!r}")

def environment() -> dict:
    versions = {}
    for name in ["torch", "transformers", "accelerate", "safetensors", "numpy", "bitsandbytes"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


def save_tensors(path, tensors):
    """Atomic safetensors checkpoint for interrupted capture stages."""
    from safetensors.torch import save_file
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".tensor-",dir=p.parent);os.close(fd)
    try:
        save_file(tensors,tmp)
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
