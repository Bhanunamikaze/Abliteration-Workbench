"""Regression tests for separating behavioral evidence from output censoring."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from ablationlab.config import DEFAULTS
from ablationlab.evidence import (classify_stats, representation_removal_signal,
                                  build_persistent_regions, candidate_priority)


def strong_censored_stats():
    """The shape and values observed in a calibrated semantic-scoring run."""
    return {
        "mean_baseline": 0.75,
        "mean_intervention": 0.0,
        "mean_gain": 0.75,
        "gain_fraction_of_baseline_mean": 0.75,
        "win_rate": 0.75,
        "ci95_low": 0.25,
        "ci95_high": 1.0,
        "random_control_max_gain_fraction": 0.0,
        "baseline_cap_rate": 0.25,
        "cap_rate": 1.0,
        "uncensored_pair_fraction": 0.0,
        "repeat_increase": 0.01,
        "required_term_change": None,
        "provisional_pass": False,
    }


def search_settings():
    return deepcopy(DEFAULTS["search"])


@pytest.mark.parametrize("metric", ["custom", "regex", "exact", "contains", "json_valid"])
def test_semantic_effect_survives_censoring(metric):
    evidence = classify_stats(strong_censored_stats(), metric, search_settings())
    assert evidence["effect_pass"] is True
    assert evidence["win_rate_pass"] is True
    assert evidence["confidence_pass"] is True
    assert evidence["specificity_pass"] is True
    assert evidence["censoring_pass"] is False
    assert evidence["behaviorally_promising"] is True
    assert evidence["quality_confirmed"] is False
    assert evidence["evidence_status"] == "promising_censored"


@pytest.mark.parametrize("metric", ["tokens", "words"])
def test_length_metric_censoring_blocks_behavioral_promotion(metric):
    evidence = classify_stats(strong_censored_stats(), metric, search_settings())
    assert evidence["effect_pass"] is True  # Raw measured difference is retained.
    assert evidence["censoring_pass"] is False
    assert evidence["quality_confirmed"] is False
    assert evidence["evidence_status"] != "validated_candidate"
    assert evidence["behaviorally_promising"] is False


def test_specificity_failure_is_not_explained_away_by_censoring():
    stats = strong_censored_stats()
    stats["random_control_max_gain_fraction"] = 0.9
    evidence = classify_stats(stats, "custom", search_settings())
    assert evidence["specificity_pass"] is False
    assert evidence["behaviorally_promising"] is False
    assert evidence["evidence_status"] != "promising_censored"


def test_uncensored_strong_effect_can_be_promising_before_heldout_validation():
    stats = strong_censored_stats()
    stats.update(baseline_cap_rate=0.0, cap_rate=0.0, uncensored_pair_fraction=1.0)
    evidence = classify_stats(stats, "custom", search_settings())
    assert evidence["behaviorally_promising"] is True
    assert evidence["quality_confirmed"] is True
    assert evidence["evidence_status"] == "promising"


def test_weak_effect_does_not_become_a_candidate():
    stats = strong_censored_stats()
    stats.update(mean_intervention=0.75, mean_gain=0.0,
                 gain_fraction_of_baseline_mean=0.0, win_rate=0.0,
                 ci95_low=0.0, ci95_high=0.0)
    evidence = classify_stats(stats, "custom", search_settings())
    assert evidence["effect_pass"] is False
    assert evidence["behaviorally_promising"] is False
    assert evidence["evidence_status"] in {"exploratory", "rejected"}


@pytest.mark.parametrize("stage", ["09_trace", "10_persistent"])
def test_historical_negative_reference_rows_remain_visible(stage):
    """Read old artifacts without rewriting them; CI may omit local run artifacts."""
    root = Path(__file__).resolve().parents[1] / "runs/llama32-refusal-negative-ref"
    result_file = root / "stages" / stage / "result.json"
    if not result_file.exists():
        pytest.skip("Historical local run artifact is absent")
    result = json.loads(result_file.read_text())
    rows = [x for x in result["summary"]
            if x["intervention"]["reference"] == "negative"
            and x["intervention"]["site"] == "residual"
            and x["intervention"]["strength"] == 1.0
            and x["gain_fraction_of_baseline_mean"] == 0.75
            and x["cap_rate"] == 1.0]
    assert rows, "Historical strong censored residual result must be present"
    for row in rows:
        evidence = classify_stats(row, "custom", search_settings())
        assert row["provisional_pass"] is False  # Old schema's collapsed decision.
        assert evidence["evidence_status"] == "promising_censored"
        assert evidence["behaviorally_promising"] is True


def test_representation_removal_signal_is_behavior_agnostic():
    projections = [{"source_layer": 7, "observed_layer": 7,
                    "prefix_tokens": 0, "ratio_of_mean_abs": 0.004}]
    summaries = [{"intervention": {"layers": [7], "site": "residual",
                                   "reference": "zero", "strength": 1.0},
                  "gain_fraction_of_baseline_mean": 0.0}]
    signal = representation_removal_signal(projections, summaries, search_settings())
    assert signal and signal[0]["layer"] == 7
    assert signal[0]["representation_removal_successful"] is True
    assert signal[0]["behavior_change_weak"] is True
    summaries[0]["gain_fraction_of_baseline_mean"] = 0.75
    assert not representation_removal_signal(projections, summaries, search_settings())
    summaries[0]["gain_fraction_of_baseline_mean"] = 0.0
    summaries[0]["intervention"]["reference"] = "negative"
    assert not representation_removal_signal(projections, summaries, search_settings())


def test_historical_zero_reference_geometry_signal():
    root = Path(__file__).resolve().parents[1] / "runs/llama32-refusal-smoke"
    result_file = root / "stages/09_trace/result.json"
    if not result_file.exists():
        pytest.skip("Historical local run artifact is absent")
    result = json.loads(result_file.read_text())
    signals = representation_removal_signal(result["projection_summary"],
                                            result["summary"], search_settings())
    assert any(row["representation_removal_successful"] and row["behavior_change_weak"]
               for row in signals)


def test_sparse_peak_span_is_guarded_without_losing_exact_peaks():
    search = search_settings()
    search.update(regions=["peaks", "span", "tail"], persistent_max_span_layers=8,
                  persistent_cluster_gap=2)
    regions, omitted = build_persistent_regions([0, 1, 18], 24, search)
    assert regions["peaks"] == [0, 1, 18]
    assert regions["cluster_1"] == [0, 1]
    assert "span" not in regions and "span" in omitted
    assert "tail" not in regions and "tail" in omitted
    assert list(range(19)) not in regions.values()


def test_small_span_and_explicit_complexity_tie_breaker():
    search = search_settings()
    search.update(regions=["peaks", "span"], persistent_max_span_layers=8)
    regions, omitted = build_persistent_regions([4, 6], 20, search)
    assert regions["span"] == [4, 5, 6]
    assert "span" not in omitted
    simple = {"evidence_status": "promising", "gain_fraction_of_baseline_mean": 0.5,
              "win_rate": 0.75, "ci95_low": 0.2, "layer_count": 1, "region_width": 1}
    complex_result = {**simple, "layer_count": 19, "region_width": 19}
    assert candidate_priority(simple) > candidate_priority(complex_result)
