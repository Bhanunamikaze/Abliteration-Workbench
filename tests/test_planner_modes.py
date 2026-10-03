"""Pure routing checks for bounded exploratory diagnostics."""
from ablationlab.planner import choose_next


def foundation():
    return {"inspect": {}, "baseline": {"no_op_exact": True}, "capture": {},
            "directions": {"eligible_layers": [1], "calibrated": True}}


def weak_refinement():
    result = foundation()
    result["sweep"] = {"candidates": [], "exploratory_candidates": [{"layer": 1}], "cap_heavy": False}
    result["refine"] = {"candidates": [], "exploratory_candidates": [{"layer": 1}], "cap_heavy": False}
    return result


def test_weak_refine_routes_by_mode():
    result = weak_refinement()
    assert choose_next(result, mode="strict").next_stage == "report"
    decision = choose_next(result, mode="explore")
    assert decision.next_stage == "writers"
    assert "exploratory" in decision.reason


def test_explore_route_is_finite_and_skips_unsupported_writer():
    result = weak_refinement()
    route = []
    for _ in range(8):
        stage = choose_next(result, mode="explore").next_stage
        route.append(stage)
        if stage == "writers":
            result[stage] = {"selected_sites": []}
        elif stage == "trace":
            result[stage] = {"summary": [], "promising": []}
        elif stage == "persistent":
            result[stage] = {"summary": [], "promising": []}
        elif stage == "report":
            break
        else:
            raise AssertionError(f"unexpected stage: {stage}")
    assert route == ["writers", "trace", "persistent", "report"]


def test_censored_causal_candidate_reaches_evaluation():
    result = weak_refinement()
    result.update(writers={"selected_sites": []}, trace={"summary": [], "promising": []},
                  persistent={"summary": [{"evidence_status": "promising_censored",
                                            "behaviorally_promising": True,
                                            "gain_fraction_of_baseline_mean": .75,
                                            "cap_rate": 1.0}],
                              "promising": []})
    assert choose_next(result, mode="explore").next_stage == "evaluate"
    result["evaluate"] = {"passed": False, "evaluation_status": "promising_censored"}
    assert choose_next(result, mode="explore").next_stage == "report"


def test_capped_length_score_does_not_trigger_heldout_claim():
    result = weak_refinement()
    result.update(writers={"selected_sites": []}, trace={"summary": [], "promising": []},
                  persistent={"summary": [{"evidence_status": "promising_censored",
                                            "behaviorally_promising": False,
                                            "metric_censoring_risk": "direct",
                                            "censoring_pass": False,
                                            "gain_fraction_of_baseline_mean": .75}],
                              "promising": []})
    assert choose_next(result, mode="explore").next_stage == "report"
