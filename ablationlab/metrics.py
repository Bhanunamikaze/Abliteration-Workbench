"""Explicit behavior scores, paired statistics and cautious projection diagnostics."""
from __future__ import annotations
import importlib
import json
import math
import re
import numpy as np
from .util import LabError

class Scorer:
    def __init__(self, config: dict):
        self.config = config
        self.custom = None
        self.regex = re.compile(config["pattern"], re.I | re.S) if config["metric"] == "regex" else None
        if config["metric"] == "custom":
            mod, name = config["plugin"].split(":", 1)
            self.custom = getattr(importlib.import_module(mod), name)
    def __call__(self, text: str, token_count: int, record: dict) -> float:
        metric = self.config["metric"]
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
        else: value = self.custom(text=text, token_count=token_count, record=record)
        value = float(value)
        if not math.isfinite(value): raise LabError("Scorer returned nonfinite value")
        return value

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
