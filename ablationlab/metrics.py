"""Explicit behavior scores, paired statistics and cautious projection diagnostics."""
from __future__ import annotations
import importlib
import importlib.util
import sys
import types
from pathlib import Path
import json
import math
import re
import numpy as np
from .util import LabError, file_hash


def scorer_provenance(plugin: str) -> dict:
    """Identify the exact source supplying a custom scorer, without executing it."""
    if not isinstance(plugin, str) or plugin.count(":") != 1:
        raise LabError("Custom scorer must be package.module:function or /path/scorer.py:function")
    source, entry = plugin.rsplit(":", 1)
    if not source or not entry.isidentifier():
        raise LabError(f"Invalid custom scorer entry point: {plugin!r}")
    local = source.endswith(".py") or source.startswith(("./", "../", "/"))
    if local:
        path = Path(source).resolve()
        if path.suffix != ".py":
            raise LabError(f"Custom scorer file must end in .py: {path}")
    else:
        try:
            spec = importlib.util.find_spec(source)
        except (ImportError, ValueError, AttributeError) as error:
            raise LabError(f"Cannot resolve custom scorer module {source!r}; use an explicit .py path relative to the source config") from error
        if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
            raise LabError(f"Cannot locate a source file for custom scorer module {source!r}; use an explicit .py path")
        path = Path(spec.origin).resolve()
    if not path.is_file():
        raise LabError(f"Custom scorer source does not exist: {path}")
    return {"plugin": plugin, "kind": "file" if local else "module", "source_path": str(path),
            "entrypoint": entry, "sha256": file_hash(path)}


def load_custom_scorer(plugin: str):
    provenance = scorer_provenance(plugin)
    source, name = plugin.rsplit(":", 1)
    # Execute exactly the bytes that were fingerprinted, avoiding stale .pyc
    # files or an unrelated module already present in sys.modules.
    path = Path(provenance["source_path"])
    source_bytes = path.read_bytes()
    import hashlib
    if hashlib.sha256(source_bytes).hexdigest() != provenance["sha256"]:
        raise LabError(f"Custom scorer changed while loading: {path}")
    if provenance["kind"] == "module":
        parent = source.rpartition(".")[0]
        if parent:
            importlib.import_module(parent)
        module_name = source
    else:
        parent = ""
        module_name = f"_ablationlab_scorer_{provenance['sha256']}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = parent
    module.__spec__ = importlib.util.spec_from_file_location(module_name, path)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(source_bytes, str(path), "exec"), module.__dict__)
    except Exception as error:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise LabError(f"Cannot load custom scorer file {path}: {error}") from error
    scorer = getattr(module, name, None)
    if not callable(scorer):
        raise LabError(f"Custom scorer {plugin!r} does not define a callable {name}")
    return scorer

class Scorer:
    def __init__(self, config: dict):
        self.config = config
        self.custom = None
        self.regex = re.compile(config["pattern"], re.I | re.S) if config["metric"] == "regex" else None
        if config["metric"] == "custom":
            self.custom = load_custom_scorer(config["plugin"])
    def __call__(self, text: str, token_count: int, record: dict) -> float:
        return self.evaluate(text, token_count, record)["score"]
    def evaluate(self, text: str, token_count: int, record: dict) -> dict:
        metric = self.config["metric"]
        details = None
        if metric == "tokens": value = token_count
        elif metric == "words": value = len(text.split())
        elif metric == "regex": value = float(bool(self.regex.search(text)))
        elif metric == "json_valid":
            try:
                json.loads(text)
                value = 1.0
            except (ValueError, TypeError): value = 0.0
        elif metric == "exact":
            if "expected" not in record: raise LabError("exact scorer needs expected in every scored row")
            value = float(text.strip() == str(record["expected"]).strip())
        elif metric == "contains":
            if not record.get("required_terms"): raise LabError("contains scorer needs required_terms")
            value = sum(str(t).lower() in text.lower() for t in record["required_terms"]) / len(record["required_terms"])
        else:
            value = self.custom(text=text, token_count=token_count, record=record)
            if isinstance(value, dict):
                if set(value) != {"score", "details"} or not isinstance(value["details"], dict):
                    raise LabError("Structured custom scorer needs score and details object")
                details = value["details"]
                try: json.dumps(details, allow_nan=False)
                except (ValueError, TypeError) as error:
                    raise LabError("Custom scorer details must be finite JSON data") from error
                value = value["score"]
        value = float(value)
        if not math.isfinite(value): raise LabError("Scorer returned nonfinite value")
        result = {"score": value}
        if details is not None: result["score_details"] = details
        return result

def repetition(text: str, n: int = 4) -> float:
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return 1 - len(set(grams))/len(grams) if grams else 0.0

def lexical_checks(text: str, record: dict) -> dict:
    terms = record.get("required_terms", [])
    return {"required_term_fraction": (sum(str(t).lower() in text.lower() for t in terms)/len(terms)) if terms else None,
            "repeat4": repetition(text), "empty": not bool(text.strip())}

def corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12: return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def paired_stats(base, intervention, goal: str = "decrease", seed: int = 17,
                 scale_floor: float = 1.0) -> dict:
    b, a = np.asarray(base, float), np.asarray(intervention, float)
    if b.shape != a.shape or not b.size or not np.isfinite(b).all() or not np.isfinite(a).all():
        raise LabError("paired_stats requires equal nonempty finite vectors")
    sign = 1 if goal == "increase" else -1
    gains = sign * (a-b)
    n = len(b)
    rng = np.random.default_rng(seed)
    boots = gains[rng.integers(0, n, (1500, n))].mean(axis=1)
    scale = max(float(np.mean(np.abs(b))), scale_floor)
    return {"n": n, "mean_baseline": float(b.mean()), "mean_intervention": float(a.mean()),
            "mean_gain": float(gains.mean()), "median_gain": float(np.median(gains)),
            "gain_fraction_of_baseline_mean": float(gains.mean()/scale),
            "mean_per_example_fraction": float((gains/np.maximum(np.abs(b), scale_floor)).mean()),
            "win_rate": float((gains > 0).mean()), "tie_rate": float((gains == 0).mean()),
            "ci95_low": float(np.quantile(boots, .025)), "ci95_high": float(np.quantile(boots, .975))}

def projection_diagnostics(base, changed, floor: float = .1) -> dict:
    b, a = np.asarray(base, float), np.asarray(changed, float)
    if b.shape != a.shape or not b.size: raise LabError("Projection arrays must align")
    mean_abs = float(np.mean(np.abs(b)))
    eligible = np.abs(b) >= floor
    error = a-b
    return {"mean_baseline_signed": float(b.mean()), "mean_changed_signed": float(a.mean()),
            "mean_signed_delta": float(error.mean()),
            "mean_baseline_abs": mean_abs, "mean_changed_abs": float(np.mean(np.abs(a))),
            "ratio_of_mean_abs": float(np.mean(np.abs(a))/mean_abs) if mean_abs >= floor else None,
            "mean_abs_paired_error": float(np.mean(np.abs(error))),
            "rmse_paired": float(np.sqrt(np.mean(error**2))),
            "sign_agreement_eligible": float((np.sign(a[eligible]) == np.sign(b[eligible])).mean()) if eligible.any() else None,
            "eligible_fraction": float(eligible.mean()),
            "median_individual_abs_ratio_eligible": float(np.median(np.abs(a[eligible])/np.abs(b[eligible]))) if eligible.any() else None}
