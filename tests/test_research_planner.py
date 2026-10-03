"""Pure planner routing checks for bounded exploratory diagnostics."""

from copy import deepcopy

from ablationlab.planner import choose_next


def weak_refined_results():
    candidate = {"layer": 3, "orientation": 1, "effect_fraction": 0.03,
                 "win_rate": 0.5, "eligible": False}
    return {
        "inspect": {}, "baseline": {"no_op_exact": True}, "capture": {},
        "directions": {"eligible_layers": [3], "calibrated": True},
        "sweep": {"candidates": [], "exploratory_candidates": [candidate], "cap_heavy": False},
        "refine": {"candidates": [], "exploratory_candidates": [candidate], "cap_heavy": False},
    }


def test_weak_refine_continues_only_in_explore_mode():
    results = weak_refined_results()
    exploratory = choose_next(results, mode="explore")
    conservative = choose_next(results, mode="strict")
    assert exploratory.next_stage == "writers"
    assert "explor" in exploratory.reason.lower()
    assert conservative.next_stage == "report"


def test_explore_route_remains_finite_with_weak_results():
    results = weak_refined_results()
    route = []
    stage_results = {
        "writers": {"selected_sites": [{"layer": 3, "site": "attention"}]},
        "ablate": {"promising": [], "summary": []},
        "trace": {"promising": [], "summary": []},
        "persistent": {"promising": [], "summary": []},
        "evaluate": {"passed": False},
        "report": {},
    }
    for _ in range(8):
        decision = choose_next(results, mode="explore")
        if decision.next_stage is None:
            break
        assert decision.next_stage not in route, "Planner repeated an automatic stage"
        route.append(decision.next_stage)
        results[decision.next_stage] = deepcopy(stage_results[decision.next_stage])
    else:
        raise AssertionError("Exploratory planner did not terminate")
    assert route[:4] == ["writers", "ablate", "trace", "persistent"]
    assert route[-1] == "report"


def test_no_exploratory_candidate_does_not_force_diagnostics():
    results = weak_refined_results()
    results["sweep"]["exploratory_candidates"] = []
    results["refine"]["exploratory_candidates"] = []
    decision = choose_next(results, mode="explore")
    assert decision.next_stage == "report"


def test_strong_censored_result_reaches_heldout_selection():
    results = weak_refined_results()
    results["writers"] = {"selected_sites": []}
    results["trace"] = {
        "promising": [{"evidence_status": "promising_censored",
                       "gain_fraction_of_baseline_mean": 0.75,
                       "intervention": {"layers": [3], "site": "residual",
                                        "reference": "negative", "operation": "ablate",
                                        "strength": 1.0}}]
    }
    results["persistent"] = {"promising": [], "summary": []}
    decision = choose_next(results, mode="explore")
    assert decision.next_stage == "evaluate"
