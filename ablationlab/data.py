"""File-driven paired conditions with group-level split isolation."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from .util import LabError, read_rows, digest, write_json

ROLES = {"system", "user", "assistant"}
SPLITS = {"train", "validation", "test", "control"}

def messages(value, label: str) -> list[dict]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise LabError(f"{label} must be a nonempty message array or string")
    out = []
    for m in value:
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            raise LabError(f"Invalid text-only message in {label}")
        out.append({"role": m["role"], "content": m["content"]})
    return out

def load_dataset(path: str | Path, min_pairs: int = 2, *, replication: bool = False) -> tuple[list[dict], dict]:
    raw = read_rows(path)
    rows, ids, groups, fingerprints = [], set(), {}, {}
    for i, row in enumerate(raw):
        key = str(row.get("id", ""))
        split = row.get("split")
        if not key or key in ids:
            raise LabError(f"Missing/duplicate id at dataset row {i+1}: {key!r}")
        ids.add(key)
        allowed={"test","control"} if replication else SPLITS
        if split not in allowed:
            raise LabError(f"Row {key}: split must be one of {sorted(allowed)}")
        group = str(row.get("group", key))
        if group in groups and groups[group] != split:
            raise LabError(f"Group leakage: {group!r} appears in {groups[group]} and {split}")
        groups[group] = split
        r = dict(row, id=key, group=group)
        if split in {"train", "validation"}:
            r["positive"] = messages(row.get("positive"), key+".positive")
            r["negative"] = messages(row.get("negative"), key+".negative")
            if r["positive"] == r["negative"]:
                raise LabError(f"Row {key} has identical positive/negative conditions")
        r["neutral"] = messages(row.get("neutral", row.get("prompt")), key+".neutral")
        # Exact neutral duplicates in different splits are also leakage.
        h = digest(r["neutral"])
        if h in fingerprints and (replication or fingerprints[h] != split):
            if fingerprints[h] != split:
                raise LabError(f"Duplicate neutral prompt crosses splits at {key}")
            raise LabError(f"Duplicate replication neutral prompt at {key}")
        fingerprints[h] = split
        if "required_terms" in r and not isinstance(r["required_terms"], list):
            raise LabError("required_terms must be a list of strings")
        rows.append(r)
    counts = Counter(r["split"] for r in rows)
    if not replication:
        for split in ["train", "validation"]:
            if counts[split] < min_pairs:
                raise LabError(f"Need >= {min_pairs} paired {split} rows; got {counts[split]}")
    for split in ["test", "control"]:
        if counts[split] < 1:
            raise LabError(f"Need >=1 {split} row for evaluation/control drift")
    return rows, {"counts": dict(counts), "rows": len(rows), "groups": len(groups),
                  "dataset_hash": digest(rows),
                  "notes": (["Frozen-intervention replication: test/control only; no discovery pairs"] if replication else
                            ["positive means condition A; this does not prove a pure behavioral feature",
                             "manual groups must isolate paraphrases/topics across splits"])}


def load_replication_dataset(path: str | Path) -> tuple[list[dict], dict]:
    """Validate fresh test/control prompts without requiring discovery pairs."""
    return load_dataset(path, replication=True)

def split_rows(rows: list[dict], split: str) -> list[dict]:
    return [r for r in rows if r["split"] == split]
