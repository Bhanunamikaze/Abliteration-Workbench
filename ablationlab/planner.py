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

def choose_next(results: dict, overrides: dict | None = None) -> Decision:
    """Pure function. Does not mutate checkpoints or invent scientific conclusions."""
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
    if "evaluate" in completed or "persistent" in completed:
        if "persistent" in completed and "evaluate" not in completed and results["persistent"].get("promising"):
            return Decision("evaluate","Persistent ablation has provisional candidates; use held-out test/control data")
        return Decision("report" if "report" not in completed else None,"Experiments complete; report measured limits","complete" if results.get("evaluate",{}).get("passed") else "needs_review")
    latest="refine" if "refine" in completed else "sweep"
    sweep=results[latest]
    if not sweep.get("candidates") or sweep.get("cap_heavy",False):
        if "refine" not in completed:
            return Decision("refine","One bounded gentler/higher-cap confirmation before interpreting weak or censored effects")
        return Decision("report" if "report" not in completed else None,
                        "Refinement remains weak/censored; no automatic escalation to weight editing","needs_review")
    if "writers" not in completed:return Decision("writers","Compare intervention sites inside supported candidate layers")
    if "ablate" not in completed and results["writers"].get("selected_sites"):
        return Decision("ablate","Test natural writer-output components, distinct from additive steering")
    if "ablate" in completed and results["ablate"].get("promising"):
        return Decision("evaluate","Writer ablation passed provisional thresholds; check held-out data")
    if "trace" not in completed:return Decision("trace","Weak isolated-writer effect; test complete residual and matched-prefix coordinates")
    if results["trace"].get("promising"):
        return Decision("evaluate","Whole-residual intervention has provisional candidates")
    if "persistent" not in completed:
        return Decision("persistent","Single-site results inconclusive; one bounded multi-layer diagnostic with random controls")
    return Decision("report" if "report" not in completed else None,"No further automatic action","needs_review")

def apply_override(decision: Decision, completed_order: list[str], overrides: dict) -> Decision:
    if not completed_order:return decision
    current=completed_order[-1]
    if current not in overrides:return decision
    nxt=overrides[current]
    if nxt=="stop":return Decision(None,f"Manual route override after {current}","manual_stop")
    if nxt not in STAGES:raise LabError(f"Invalid override stage {nxt}")
    if nxt in completed_order:raise LabError("Route override forms a completed-stage loop; fork/reset explicitly")
    return Decision(nxt,f"Manual route override {current} -> {nxt}; not an automatic scientific conclusion","manual")
