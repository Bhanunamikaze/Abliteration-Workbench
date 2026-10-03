import pytest
from ablationlab.planner import choose_next,apply_override,Decision
from ablationlab.util import LabError

def foundation():return {"inspect":{},"baseline":{},"capture":{},"directions":{"eligible_layers":[1,2],"calibrated":True}}

def test_start():assert choose_next({}).next_stage=="inspect"
def test_empty_direction_no_forced_winner():
    r=foundation();r["directions"]["eligible_layers"]=[]
    d=choose_next(r);assert d.next_stage=="report" and d.status=="needs_data"
def test_calibrated_to_sweep():assert choose_next(foundation()).next_stage=="sweep"
def test_legacy_no_fake_validation():
    r=foundation();r["directions"]["calibrated"]=False
    assert choose_next(r).status=="needs_review"
@pytest.mark.parametrize("weak,capped",[(True,False),(False,True),(True,True)])
def test_bounded_refine(weak,capped):
    r=foundation();r["sweep"]={"candidates":[] if weak else [{"layer":1}],"cap_heavy":capped}
    assert choose_next(r).next_stage=="refine"
    r["refine"]=r["sweep"]
    assert choose_next(r).next_stage=="report"
def test_good_sweep_to_writer():
    r=foundation();r["sweep"]={"candidates":[{"layer":1}]}
    assert choose_next(r).next_stage=="writers"
def test_no_supported_writer_to_trace():
    r=foundation();r.update(sweep={"candidates":[{}]},writers={"selected_sites":[]})
    assert choose_next(r).next_stage=="trace"
def test_weak_ablation_trace_persistent():
    r=foundation();r.update(sweep={"candidates":[{}]},writers={"selected_sites":[{}]},ablate={"promising":[]})
    assert choose_next(r).next_stage=="trace"
    r["trace"]={"promising":[]};assert choose_next(r).next_stage=="persistent"
    r["persistent"]={"promising":[]};assert choose_next(r).status=="needs_review"
def test_promising_to_evaluate_not_export():
    r=foundation();r.update(sweep={"candidates":[{}]},writers={"selected_sites":[{}]},ablate={"promising":[{}]})
    assert choose_next(r).next_stage=="evaluate"
    r["evaluate"]={"passed":True};assert choose_next(r).next_stage=="report"
    r["report"]={};assert choose_next(r).next_stage is None

def test_manual_loop_rejected():
    with pytest.raises(LabError):apply_override(Decision("writers","x"),["sweep"],{"sweep":"sweep"})
def test_manual_route():
    assert apply_override(Decision("writers","x"),["sweep"],{"sweep":"trace"}).next_stage=="trace"

def test_nondeterministic_noop_requires_review():
    r=foundation();r["baseline"]["no_op_exact"]=False
    d=choose_next(r)
    assert d.status=="needs_runtime_review" and d.next_stage=="report"
