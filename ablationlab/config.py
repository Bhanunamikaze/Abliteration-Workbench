"""Validated JSON/YAML configuration. CLI overrides use dotted keys."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from .util import LabError, read_json

DEFAULTS = {
    "schema_version": 1,
    # Existing run snapshots without this key retain their original identity hash.
    "identity_schema": 2,
    "profile": "fast",
    "dataset": "data.jsonl",
    "model": {"id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "main",
              "device": "cuda", "dtype": "auto", "attention": "sdpa",
              "quantization": "none", "local_files_only": False,
              "trust_remote_code": False, "backend_plugin": "", "adapter": {}, "allow_offload": False},
    "behavior": {"name": "verbosity", "metric": "tokens", "goal": "decrease",
                 "pattern": "", "plugin": "", "scale_floor": 1.0},
    "generation": {"batch_size": 4, "max_new_tokens": 384, "max_input_tokens": 1024,
                   "seed": 17, "scope": "last", "phase": "both", "refine_max_new_tokens": 768,
                   "confirm_max_new_tokens": 1024},
    "discovery": {"rank": 1, "method": "mean", "min_pairs": 4},
    "search": {"layers": None, "top_k": 3, "strengths": [-0.25, 0.25],
               "ablations": [0.5, 1.0], "reference": "zero", "mode": "strict",
               "min_separation": 0.25, "min_effect_fraction": 0.10,
               "min_win_rate": 0.75, "max_cap_rate": 0.25,
               "max_repeat_increase": 0.10, "max_control_drift": 0.25,
               "random_controls": 1, "max_layer_tests": 8,
               "writer_fraction": 0.75, "refine_strength_factor": 0.5, "regions": ["peaks", "span"],
               "custom_regions": [], "persistent_max_span_layers": 8,
               "persistent_cluster_gap": 2, "max_cap_confirmations": 1,
               "overrides": {}},
    "evaluation": {"reference_tokens": 32, "max_reference_nll_increase": 2.0,
                   "baseline_include_test": False},
    "trace": {"prefix_tokens": [0, 8], "denominator_floor": 0.1},
    "budget": {"max_generations": 1600, "max_seconds": 14400},
}
PROFILES = {
    "fast": {},
    "normal": {"generation": {"max_new_tokens": 600},
               "search": {"strengths": [-0.5, -0.25, 0.25, 0.5],
                          "ablations": [0.25, 0.5, 0.75, 1.0], "max_layer_tests": 16}},
    "rigorous": {"generation": {"max_new_tokens": 900, "batch_size": 2},
                 "discovery": {"min_pairs": 12},
                 "search": {"strengths": [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5],
                            "ablations": [0.25, 0.5, 0.75, 1.0],
                            "max_layer_tests": 0, "random_controls": 3},
                 "trace": {"prefix_tokens": [0, 8, 32]}},
}

def merge(base: dict, updates: dict, strict: bool = True, prefix: str = "") -> dict:
    out = deepcopy(base)
    for k, v in updates.items():
        if strict and k not in out:
            raise LabError(f"Unknown configuration key: {prefix}{k}")
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            # Adapter maps / explicit route overrides are intentionally open maps.
            out[k] = merge(out[k], v, strict and k not in {"adapter", "overrides"}, prefix+k+".")
        else:
            out[k] = deepcopy(v)
    return out

def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    p = Path(path).resolve()
    if p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as e:
            raise LabError("Install ablationlab-local[yaml] or use JSON config") from e
        value = yaml.safe_load(p.read_text())
    else:
        value = read_json(p)
    if not isinstance(value, dict):
        raise LabError("Configuration must be an object")
    import json
    value = deepcopy(value)
    for item in overrides or []:
        if "=" not in item:
            raise LabError("Overrides must be dotted.key=JSON_VALUE")
        key, raw = item.split("=", 1)
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            v = raw
        cursor = value
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = v
    profile = value.get("profile", "fast")
    if profile not in PROFILES:
        raise LabError(f"Unknown profile {profile}; choose {list(PROFILES)}")
    config = merge(merge(DEFAULTS, PROFILES[profile]), value)
    dataset = Path(config["dataset"])
    config["dataset"] = str((p.parent / dataset).resolve() if not dataset.is_absolute() else dataset)
    model_id = config["model"]["id"]
    if model_id.startswith(("./", "../", "/")):
        config["model"]["id"] = str((p.parent / model_id).resolve())
    if config["behavior"]["metric"] == "custom":
        plugin = config["behavior"]["plugin"]
        if isinstance(plugin, str) and ":" in plugin:
            module, entry = plugin.rsplit(":", 1)
            if module.endswith(".py") or module.startswith(("./", "../", "/")):
                config["behavior"]["plugin"] = f"{(p.parent / module).resolve()}:{entry}"
            else:
                # A dotted module alongside a source config should also work when
                # the CLI is launched from another directory or installed as a wheel.
                relative = Path(*module.split("."))
                for base in (p.parent, p.parent.parent):
                    candidate = base / relative.with_suffix(".py")
                    if candidate.is_file():
                        config["behavior"]["plugin"] = f"{candidate.resolve()}:{entry}"
                        break
    validate_config(config)
    if config["behavior"]["metric"] == "custom":
        from .metrics import scorer_provenance
        scorer_provenance(config["behavior"]["plugin"])
    return config

def validate_config(c: dict) -> None:
    if c["schema_version"] != 1:
        raise LabError("Unsupported config schema_version")
    if c.get("identity_schema", 1) not in {1, 2}:
        raise LabError("Unsupported identity_schema")
    b, g, d, s, m = (c[k] for k in ["behavior", "generation", "discovery", "search", "model"])
    if b["goal"] not in {"increase", "decrease"}:
        raise LabError("behavior.goal must be increase or decrease")
    if b["metric"] not in {"tokens", "words", "regex", "json_valid", "exact", "contains", "custom"}:
        raise LabError("Unknown behavior.metric")
    if b["metric"] == "regex" and not b["pattern"]:
        raise LabError("regex metric requires behavior.pattern")
    if b["metric"] == "custom" and (not isinstance(b["plugin"], str) or b["plugin"].count(":") != 1
                                     or not all(b["plugin"].split(":"))):
        raise LabError("custom metric requires behavior.plugin=package.module:function or ./scorer.py:function")
    if b["scale_floor"] <= 0:
        raise LabError("behavior.scale_floor must be positive")
    if m["dtype"] not in {"auto", "fp16", "bf16", "fp32"}:
        raise LabError("Invalid dtype")
    if m["quantization"] not in {"none", "4bit", "8bit"}:
        raise LabError("Only none, 4bit and 8bit loading are supported")
    if m["device"] not in {"cpu", "cuda", "cuda:0", "auto"}:
        raise LabError("Supported devices: cpu, cuda, cuda:0, auto; no TPU backend")
    for key in ["batch_size", "max_new_tokens", "max_input_tokens"]:
        if not isinstance(g[key], int) or g[key] < 1:
            raise LabError(f"generation.{key} must be a positive integer")
    if g["scope"] not in {"last", "all"} or g["phase"] not in {"both", "prefill", "decode"}:
        raise LabError("scope=last|all; phase=both|prefill|decode")
    if d["method"] not in {"mean", "mean_svd"} or not 1 <= d["rank"] <= 16 or d["min_pairs"] < 2:
        raise LabError("discovery: method=mean|mean_svd, rank=1..16, min_pairs>=2")
    if d["method"] == "mean" and d["rank"] != 1:
        raise LabError("rank>1 requires method=mean_svd")
    if not isinstance(s["reference"], str) or s["reference"] not in {"zero", "negative", "auto"}:
        raise LabError("search.reference must be zero, negative, or auto")
    if not isinstance(s["mode"], str) or s["mode"] not in {"strict", "explore"}:
        raise LabError("search.mode must be strict or explore")
    if s["top_k"] < 1 or s["max_layer_tests"] < 0 or s["random_controls"] < 0:
        raise LabError("Invalid search counts")
    if not s["strengths"] or not all(isinstance(x, (float, int)) and x != 0 for x in s["strengths"]):
        raise LabError("Nonzero numeric search.strengths required")
    if not all(-x in s["strengths"] for x in s["strengths"]):
        raise LabError("Steering grid must contain symmetric positive/negative pairs")
    if not s["ablations"] or any(not 0 < x <= 1 for x in s["ablations"]):
        raise LabError("Ablation strengths must be in (0,1]")
    for key in ["min_win_rate", "max_cap_rate", "writer_fraction"]:
        if not 0 <= s[key] <= 1:
            raise LabError(f"search.{key} must be in [0,1]")
    if s["layers"] is not None and (not s["layers"] or any(not isinstance(x,int) or x < 0 for x in s["layers"])):
        raise LabError("search.layers must be null or nonempty nonnegative integer list")
    if any(x not in {"peaks", "span", "tail"} for x in s["regions"]):
        raise LabError("regions can contain peaks, span, tail")
    custom=s.get("custom_regions",[])
    if not isinstance(custom,list):
        raise LabError("search.custom_regions must be a list")
    labels=set(s["regions"])
    for entry in custom:
        if not isinstance(entry,dict) or set(entry)!={"label","layers"}:
            raise LabError("Each custom region needs label and layers")
        label,layers=entry["label"],entry["layers"]
        if not isinstance(label,str) or not label.strip() or label in labels:
            raise LabError("Custom region labels must be unique nonempty strings")
        if not isinstance(layers,list) or not layers or any(type(x) is not int or x<0 for x in layers) or len(layers)!=len(set(layers)):
            raise LabError("Custom region layers must be a nonempty list of unique nonnegative integers")
        labels.add(label)
    if c["evaluation"]["reference_tokens"] < 0:
        raise LabError("evaluation.reference_tokens must be nonnegative")
    if type(c["evaluation"].get("baseline_include_test", False)) is not bool:
        raise LabError("evaluation.baseline_include_test must be a boolean")
    if c["budget"]["max_generations"] < 1 or c["budget"]["max_seconds"] <= 0:
        raise LabError("Budget values must be positive")
    if not c["trace"]["prefix_tokens"] or any(not isinstance(x,int) or x < 0 for x in c["trace"]["prefix_tokens"]):
        raise LabError("trace.prefix_tokens must be nonnegative integers")

    # Explicit finite/type checks prevent NaN configs, duplicate jobs and accidental bool counts.
    import math
    import re
    def number(value, label, minimum=0.0, strictly_positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise LabError(f"{label} must be a finite number")
        if value < minimum or (strictly_positive and value <= minimum):
            raise LabError(f"{label} must be {'greater than' if strictly_positive else 'at least'} {minimum}")
    def integer(value, label, minimum=0):
        if type(value) is not int or value < minimum:
            raise LabError(f"{label} must be an integer >= {minimum}")
    for key in ["batch_size", "max_new_tokens", "max_input_tokens", "refine_max_new_tokens", "confirm_max_new_tokens"]:
        integer(g[key], "generation."+key, 1)
    integer(g["seed"], "generation.seed", 0)
    integer(d["min_pairs"], "discovery.min_pairs", 2)
    integer(d["rank"], "discovery.rank", 1)
    for key in ["top_k", "max_layer_tests", "random_controls", "persistent_max_span_layers",
                "persistent_cluster_gap", "max_cap_confirmations"]:
        integer(s[key], "search."+key, 1 if key == "top_k" else 0)
    if s["persistent_max_span_layers"] < 1:
        raise LabError("search.persistent_max_span_layers must be at least 1")
    if s["persistent_cluster_gap"] < 1:
        raise LabError("search.persistent_cluster_gap must be at least 1")
    if s["max_cap_confirmations"] > 2:
        raise LabError("search.max_cap_confirmations must be at most 2")
    for key in ["min_separation", "min_effect_fraction", "min_win_rate", "max_cap_rate",
                "max_repeat_increase", "max_control_drift", "writer_fraction", "refine_strength_factor"]:
        number(s[key], "search."+key, strictly_positive=key == "refine_strength_factor")
    number(b["scale_floor"], "behavior.scale_floor", strictly_positive=True)
    number(c["trace"]["denominator_floor"], "trace.denominator_floor", strictly_positive=True)
    number(c["evaluation"]["max_reference_nll_increase"], "evaluation.max_reference_nll_increase")
    integer(c["evaluation"]["reference_tokens"], "evaluation.reference_tokens", 0)
    integer(c["budget"]["max_generations"], "budget.max_generations", 1)
    number(c["budget"]["max_seconds"], "budget.max_seconds", strictly_positive=True)
    for label, values in [("search.strengths", s["strengths"]), ("search.ablations", s["ablations"])]:
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise LabError(f"{label} must be a list without duplicate values")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise LabError(f"{label} must contain finite numbers")
    if s["layers"] is not None:
        for value in s["layers"]: integer(value, "search.layers entry", 0)
    for value in c["trace"]["prefix_tokens"]: integer(value, "trace.prefix_tokens entry", 0)
    if b["metric"] == "regex":
        try: re.compile(b["pattern"])
        except re.error as error: raise LabError(f"Invalid behavior.pattern: {error}") from error
    if not isinstance(m["id"], str) or not m["id"].strip():
        raise LabError("model.id must be a nonempty model identifier or local checkpoint path")
