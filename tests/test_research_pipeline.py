"""End-to-end and stage-level regressions for exploratory research routing."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from ablationlab.data import load_dataset
from ablationlab.interventions import Intervention
from ablationlab.pipeline import Pipeline
from ablationlab.store import RunStore, STAGES
from test_research_evidence import strong_censored_stats


def _result_row(layers, *, gain, status, reference="negative", cap_rate=1.0):
    spec = Intervention(tuple(layers), site="residual", operation="ablate",
                        strength=1.0, reference=reference)
    return {"intervention": spec.serial(), "gain_fraction_of_baseline_mean": gain,
            "mean_baseline": 0.75, "mean_intervention": 0.75 - gain,
            "mean_gain": gain,
            "win_rate": 0.75, "ci95_low": 0.25, "ci95_high": 1.0,
            "random_control_max_gain_fraction": 0.0, "baseline_cap_rate": 0.25,
            "cap_rate": cap_rate, "repeat_increase": 0.01,
            "required_term_change": None,
            "uncensored_pair_fraction": 0.0 if cap_rate else 1.0,
            "evidence_status": status, "behaviorally_promising": True,
            "specificity_pass": True,
            "provisional_pass": False, "layer_count": len(layers),
            "region_width": max(layers) - min(layers) + 1}


def test_steering_sweep_retains_strong_capped_layer_in_ranking(store, monkeypatch, tmp_path):
    p = Pipeline(store)
    p.c["behavior"]["metric"] = "regex"
    p.c["search"].update(random_controls=1, top_k=1, min_win_rate=0.5)
    records = [{"id": f"v{i}"} for i in range(4)]
    monkeypatch.setattr(p, "_eval_rows", lambda split="validation": records)
    monkeypatch.setattr(p, "_base", lambda rows, limit=None:
                        [{"id": row["id"], "score": 1.0, "hit_limit": False,
                          "repeat4": 0.0, "empty": False} for row in rows])

    def variant(rows, spec, base, limit=None):
        score = (1.0 if spec.control_seed is not None else
                 (0.0 if spec.strength > 0 else 1.0) if spec.layers == (1,) else
                 (0.9 if spec.strength > 0 else 1.0))
        return ([{"id": row["id"], "score": score, "hit_limit": spec.layers == (1,),
                  "repeat4": 0.0, "empty": False} for row in rows], {})

    monkeypatch.setattr(p, "_run_variant", variant)
    result = p._steering_sweep(tmp_path, [1, 2], [-0.25, 0.25], 6)
    assert result["exploratory_candidates"][0]["layer"] == 1
    assert result["exploratory_candidates"][0]["evidence_status"] == "promising_censored"
    assert result["cap_heavy"] is True


@pytest.mark.parametrize("legacy_summary", [False, True])
def test_evaluate_selects_strong_censored_causal_effect_over_weak_steering(
        store, monkeypatch, tmp_path, legacy_summary):
    """The old `promising=[]` path silently selected a weaker steering effect."""
    p = Pipeline(store)
    p.c["behavior"]["metric"] = "custom"  # Historical run used a custom classifier.
    strong = _result_row([1], gain=0.75, status="promising_censored")
    if legacy_summary:
        for key in ["evidence_status", "behaviorally_promising", "layer_count", "region_width"]:
            strong.pop(key)
    monkeypatch.setattr(store, "completed", lambda stage: stage == "persistent")
    monkeypatch.setattr(store, "result", lambda stage: {"promising": [], "summary": [strong]})
    monkeypatch.setattr(p, "_candidates", lambda: [{"layer": 2, "orientation": 1,
                                                    "gain_fraction_of_baseline_mean": 0.125}])
    monkeypatch.setattr(p, "_eval_rows", lambda split="validation": [{"id": split}])
    monkeypatch.setattr(p, "_base", lambda records, limit=None:
                        [{"id": r["id"], "score": 0.75, "token_ids": [], "tokens": 0}
                         for r in records])
    used = []

    def variant(records, spec, base, limit=None):
        used.append(spec)
        return ([{"id": r["id"], "score": 0.75 if r["id"] == "control" else 0.0,
                  "token_ids": [], "tokens": 0}
                 for r in records],
                {"provisional_pass": True, "evidence_status": "promising",
                 "cap_rate": 0.0, "effect_pass": True, "win_rate_pass": True,
                 "confidence_pass": True, "repeat_pass": True,
                 "content_check_pass": True, "censoring_pass": True})

    monkeypatch.setattr(p, "_run_variant", variant)
    result = p.stage_evaluate(tmp_path)
    assert used and used[0].operation == "ablate"
    assert used[0].reference == "negative"
    assert tuple(used[0].layers) == (1,)
    assert result["candidate_source_stage"] == "persistent"
    assert result["candidate_evidence_status"] == "promising_censored"
    assert result["selection_reason"]
    assert result["heldout_effect_pass"] is True
    assert result["selection_quality_confirmed"] is False
    assert result["evaluation_status"] == "promising_censored"
    assert result["passed"] is False


def test_evaluate_can_validate_confirmed_causal_candidate(store, monkeypatch, tmp_path):
    p = Pipeline(store)
    confirmed = _result_row([1], gain=0.75, status="promising", cap_rate=0.0)
    confirmed["generation_cap"] = 12
    monkeypatch.setattr(store, "completed", lambda stage: stage == "trace")
    monkeypatch.setattr(store, "result", lambda stage: {"summary": [confirmed]})
    monkeypatch.setattr(p, "_candidates", lambda: [{"layer": 2, "orientation": 1,
                                                    "effect_fraction": 0.125}])
    monkeypatch.setattr(p, "_eval_rows", lambda split="validation": [{"id": split}])
    monkeypatch.setattr(p, "_base", lambda rows, limit=None:
                        [{"id": r["id"], "score": 0.75, "token_ids": [], "tokens": 0}
                         for r in rows])

    def variant(rows, spec, base, limit=None):
        assert limit == 12
        return ([{"id": r["id"], "score": 0.75 if r["id"] == "control" else 0.0,
                  "token_ids": [], "tokens": 0} for r in rows],
                {"evidence_status": "promising", "effect_pass": True,
                 "win_rate_pass": True, "confidence_pass": True,
                 "repeat_pass": True, "content_check_pass": True,
                 "censoring_pass": True})

    monkeypatch.setattr(p, "_run_variant", variant)
    result = p.stage_evaluate(tmp_path)
    assert result["candidate_source_stage"] == "trace"
    assert result["evaluation_status"] == "validated_candidate"
    assert result["passed"] is True


@pytest.mark.parametrize("resolves", [True, False])
def test_strong_censored_ablation_gets_one_matched_high_cap_confirmation(
        store, monkeypatch, tmp_path, resolves):
    p = Pipeline(store)
    p.c["behavior"]["metric"] = "regex"
    p.c["search"].update(ablations=[1.0], reference="zero", random_controls=1,
                          max_cap_confirmations=1)
    p.c["generation"]["confirm_max_new_tokens"] = 12
    monkeypatch.setattr(p, "_eval_rows", lambda split="validation": [{"id": "v0"}])
    baseline_caps = []

    def baseline(records, limit=None):
        cap = limit or p.c["generation"]["max_new_tokens"]
        baseline_caps.append(cap)
        return [{"id": "v0", "score": 0.75, "tokens": 1, "cap": cap}]

    monkeypatch.setattr(p, "_base", baseline)
    calls = []

    def variant(records, spec, base, limit=None):
        cap = limit or p.c["generation"]["max_new_tokens"]
        calls.append((spec.reference, spec.control_seed, cap, base[0]["cap"]))
        assert base[0]["cap"] == cap, "Baseline and intervention caps must match"
        stats = deepcopy(strong_censored_stats())
        if spec.control_seed is not None:
            stats.update(gain_fraction_of_baseline_mean=0.0, mean_gain=0.0,
                         mean_intervention=0.75, win_rate=0.0, ci95_low=0.0)
        elif cap == 12 and resolves:
            stats.update(baseline_cap_rate=0.0, cap_rate=0.0,
                         uncensored_pair_fraction=1.0)
        stats.update(intervention=spec.serial(), generation_cap=cap,
                     layer_count=len(spec.layers), region_width=1)
        return [{"id": "v0", "score": stats["mean_intervention"],
                 "token_ids": [], "tokens": 0}], stats

    monkeypatch.setattr(p, "_run_variant", variant)
    result = p._ablation_variants(tmp_path, [([1], "residual", "single")], "trace")
    assert baseline_caps == [6, 12]
    assert len(calls) == 4  # Candidate/control at each of the two finite caps.
    assert all(reference == "zero" and cap == baseline_cap
               for reference, _, cap, baseline_cap in calls)
    row = result["summary"][0]
    assert row["confirmation"]["attempted"] is True
    assert row["confirmation"]["bounded_attempts"] == 1
    assert row["confirmation"]["initial_status"] == "promising_censored"
    assert row["evidence_status"] == ("promising" if resolves else "promising_censored")
    assert row in result["promising"]


@pytest.mark.integration
def test_calibrated_toy_residual_references_are_distinct_and_weight_safe(
        tmp_path, toy_config, dataset_file):
    config = deepcopy(toy_config)
    config["search"].update(reference="auto", layers=[1], random_controls=0)
    rows, audit = load_dataset(dataset_file, 2)
    store = RunStore.create(tmp_path / "references", config, rows, audit)
    pipeline = Pipeline(store)
    with store.lock():
        pipeline.run_stage("directions", with_deps=True)
        before = {k: v.detach().clone() for k, v in pipeline.engine.backend.model.state_dict().items()}
        records = pipeline._eval_rows()
        baseline = pipeline._base(records)
        for reference in ["zero", "negative"]:
            spec = pipeline._spec([1], site="residual", operation="ablate",
                                  strength=1.0, reference=reference)
            output, summary = pipeline._run_variant(records, spec, baseline)
            assert len(output) == len(records)
            assert summary["intervention"]["reference"] == reference
        assert all(torch.equal(value, pipeline.engine.backend.model.state_dict()[name])
                   for name, value in before.items())


@pytest.mark.integration
def test_auto_reference_compares_residual_only_on_toy_run(tmp_path, toy_config, dataset_file):
    config = deepcopy(toy_config)
    config["search"].update(reference="auto", layers=[1], random_controls=0,
                            ablations=[1.0], persistent_max_span_layers=3)
    config["generation"]["confirm_max_new_tokens"] = config["generation"]["max_new_tokens"]
    rows, audit = load_dataset(dataset_file, 2)
    store = RunStore.create(tmp_path / "auto_reference", config, rows, audit)
    p = Pipeline(store)
    with store.lock():
        p.run_stage("trace", with_deps=True)
        result = store.result("trace")
        refs = {row["intervention"]["reference"] for row in result["summary"]}
        assert refs == {"zero", "negative"}
        assert result["reference_comparison_performed"] is True
        writer_path = tmp_path / "writer_probe"
        writer_path.mkdir()
        writer = p._ablation_variants(writer_path,
                                      [([1], "attention", "writer")], "ablate")
        assert {row["intervention"]["reference"] for row in writer["summary"]} == {"zero"}


def test_auto_reference_requires_calibrated_negative_centroid(store, monkeypatch):
    p = Pipeline(store)
    p.c["search"]["reference"] = "auto"
    monkeypatch.setattr(p, "bundle", lambda: SimpleNamespace(metadata={"calibrated": False},
                                                              negative=object()))
    assert p._residual_references("residual") == ["zero"]
    monkeypatch.setattr(p, "bundle", lambda: SimpleNamespace(metadata={"calibrated": True},
                                                              negative=None))
    assert p._residual_references("residual") == ["zero"]
    assert p._residual_references("mlp") == ["zero"]


def test_persistent_wide_span_requires_explicit_custom_region(store, monkeypatch, tmp_path):
    p = Pipeline(store)
    p.c["search"].update(regions=["peaks", "span"], persistent_max_span_layers=8,
                          persistent_cluster_gap=2,
                          custom_regions=[{"label": "manual_wide", "layers": list(range(19))}])
    monkeypatch.setattr(p, "_candidates", lambda: [{"layer": x} for x in [0, 1, 18]])
    monkeypatch.setattr(p, "bundle", lambda: SimpleNamespace(basis=torch.empty(24, 1, 1)))
    tested = []

    def variants(path, sites, source_stage):
        tested.extend(sites)
        return {"summary": [], "promising": []}

    monkeypatch.setattr(p, "_ablation_variants", variants)
    result = p.stage_persistent(tmp_path)
    assert result["regions"]["peaks"] == [0, 1, 18]
    assert result["regions"]["cluster_1"] == [0, 1]
    assert "span" not in result["regions"]
    assert "span" in result["omitted_regions"]
    assert result["regions"]["manual_wide"] == list(range(19))
    assert len(tested) == 3


@pytest.mark.integration
def test_explore_and_strict_auto_routes_are_bounded(tmp_path, toy_config, dataset_file):
    rows, audit = load_dataset(dataset_file, 2)
    for mode in ["strict", "explore"]:
        config = deepcopy(toy_config)
        config["search"].update(mode=mode, layers=[1], random_controls=0,
                                min_effect_fraction=10.0)
        store = RunStore.create(tmp_path / mode, config, rows, audit)
        p = Pipeline(store)
        with store.lock():
            decisions = p.auto(max_steps=len(STAGES) + 2)
        stages = [item["next_stage"] for item in decisions if item["next_stage"]]
        assert store.completed("report")
        assert len(stages) == len(set(stages))
        assert len(stages) <= len(STAGES)
        if mode == "explore" and store.result("refine").get("exploratory_candidates"):
            assert "writers" in stages and "trace" in stages and "persistent" in stages
        if mode == "strict" and not store.result("refine").get("candidates"):
            assert "writers" not in stages
        assert not (store.root / "model.safetensors").exists()


@pytest.mark.integration
def test_reference_strategy_fork_reuses_foundations_and_stage_locations(store, tmp_path):
    with store.lock():
        p = Pipeline(store)
        p.run_stage("baseline", with_deps=True)
        p.run_stage("directions", with_deps=True)
        original_cache = set(store.cache)
        changed = deepcopy(store.config)
        changed["search"].update(reference="auto", mode="explore")
        changed["generation"]["confirm_max_new_tokens"] += 4
        branch = store.fork(tmp_path / "reference_branch", changed)
    assert all(branch.completed(stage) for stage in
               ["inspect", "baseline", "capture", "directions"])
    assert not branch.completed("sweep")
    assert original_cache <= set(branch.cache)
    with branch.lock():
        Pipeline(branch)._base(Pipeline(branch)._eval_rows())
    assert branch.state()["generation_count"] == 0
    assert branch.stage_dir("trace").name == "09_trace"
    assert branch.stage_dir("evaluate").name == "11_evaluate"
