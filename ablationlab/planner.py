"""Bounded evidence-driven stage selection. Rules are explicit and unit-testable."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from .util import LabError
from .store import STAGES

@dataclass
class Decision:
    next_stage: str | None
    reason: str
    status: str = "continue"
    def serial(self): return asdict(self)

def _has_evaluable_effect(results: dict) -> bool:
    """Whether a causal result warrants a held-out check, including censored effects."""
    def measurable(row):
        return not (row.get("metric_censoring_risk") == "direct" and row.get("censoring_pass") is False)

    for stage in ("ablate", "trace", "persistent"):
        result = results.get(stage, {})
        if any(measurable(row) and (row.get("evidence_status") != "promising_censored"
                                    or row.get("behaviorally_promising", True))
               for row in result.get("promising", [])):
            return True
        for row in result.get("summary", []):
            if not measurable(row):
                continue
            if row.get("evidence_status") == "promising_censored" and row.get("behaviorally_promising") is False:
                continue
            if row.get("behaviorally_promising"):
                return True
            # Older run snapshots have only provisional_pass. An exploratory
            # effect remains worth testing if it is positive and specific.
            if (row.get("evidence_status") == "exploratory"
                    and row.get("gain_fraction_of_baseline_mean", 0) > 0
                    and row.get("specificity_pass", True)):
                return True
    return False


def choose_next(results: dict, overrides: dict | None = None, mode: str = "strict") -> Decision:
    """Pure function. Does not mutate checkpoints or invent scientific conclusions."""
    if mode not in {"strict", "explore"}:
        raise LabError(f"Unknown planner mode {mode!r}; expected strict or explore")
    completed=set(results)
    for needed in ["inspect","baseline","capture","directions"]:
        if needed not in completed:return Decision(needed,f"Required foundational stage: {needed}")
        if needed=="baseline" and results[needed].get("no_op_exact") is False:
            return Decision("report" if "report" not in completed else None,"Zero-strength control differed from baseline; resolve runtime nondeterminism before interpreting interventions","needs_runtime_review")
        if needed=="directions" and not results[needed].get("eligible_layers"):
            return Decision("report" if "report" not in completed else None,
                            "No held-out contrast passed calibration; revise data or manual plan", "needs_data")
    if not results["directions"].get("calibrated",True):
        return Decision("report" if "report" not in completed else None,"Legacy directions lack calibrated class centroids/revision. Recapture or run explicitly exploratory stages.","needs_review")
    if "sweep" not in completed:return Decision("sweep","Test behavioral effects, not just activation separation")
    if "evaluate" in completed:
        return Decision("report" if "report" not in completed else None,
                        "Held-out evaluation complete; report measured effects and quality limits",
                        "complete" if results["evaluate"].get("passed") else "needs_review")
    if "persistent" in completed:
        if _has_evaluable_effect(results):
            return Decision("evaluate","Select the strongest supported causal candidate, retaining censored effects, for held-out testing")
        return Decision("report" if "report" not in completed else None,
                        "No reliable behavioral effect found in the bounded diagnostic route", "needs_review")
    latest="refine" if "refine" in completed else "sweep"
    sweep=results[latest]
    weak_or_capped=not sweep.get("candidates") or sweep.get("cap_heavy",False)
    if weak_or_capped and "refine" not in completed:
        return Decision("refine","One bounded gentler/higher-cap steering refinement before interpreting weak or censored effects")
    if weak_or_capped:
        if mode == "strict" or not sweep.get("exploratory_candidates"):
            return Decision("report" if "report" not in completed else None,
                            "Refinement remains weak/censored; no further automatic diagnostic route",
                            "needs_review")
    if "writers" not in completed:
        reason=("No validated steering candidate; continuing bounded exploratory diagnostics on top-ranked candidates"
                if weak_or_capped else "Compare intervention sites inside supported candidate layers")
        return Decision("writers",reason)
    if "ablate" not in completed and results["writers"].get("selected_sites"):
        return Decision("ablate","Test natural writer-output components, distinct from additive steering")
    if mode == "strict" and "ablate" in completed and results["ablate"].get("promising"):
        return Decision("evaluate","Writer ablation has a behavioral candidate; check held-out data, including any censoring caveat")
    if "trace" not in completed:return Decision("trace","Weak isolated-writer effect; test complete residual and matched-prefix coordinates")
    if mode == "strict" and results["trace"].get("promising"):
        return Decision("evaluate","Whole-residual intervention has a behavioral candidate, possibly censored")
    return Decision("persistent","Single-site results inconclusive; one bounded multi-layer diagnostic with random controls")

def apply_override(decision: Decision, completed_order: list[str], overrides: dict) -> Decision:
    if not completed_order:return decision
    current=completed_order[-1]
    if current not in overrides:return decision
    nxt=overrides[current]
    if nxt=="stop":return Decision(None,f"Manual route override after {current}","manual_stop")
    if nxt not in STAGES:raise LabError(f"Invalid override stage {nxt}")
    if nxt in completed_order:raise LabError("Route override forms a completed-stage loop; fork/reset explicitly")
    return Decision(nxt,f"Manual route override {current} -> {nxt}; not an automatic scientific conclusion","manual")
