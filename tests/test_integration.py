from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from ablationlab.pipeline import Pipeline
from ablationlab.store import RunStore,STAGES
from ablationlab.data import load_dataset
from ablationlab.util import LabError,read_json,write_json,digest
from ablationlab.export import project_linear,make_export_plan,export_checkpoint
from ablationlab.backend import Backend
from ablationlab.interventions import transform
from ablationlab.cli import main

@pytest.mark.integration
@pytest.mark.parametrize("moe",[False,True])
def test_full_stage_execution(tmp_path,toy_config,dataset_file,moe):
    toy_config["model"]["id"]="toy:moe" if moe else "toy:dense"
    toy_config["search"]["layers"]=[1,2]
    data,audit=load_dataset(dataset_file,2)
    store=RunStore.create(tmp_path/"pipeline",toy_config,data,audit)
    pipeline=Pipeline(store)
    original=None
    with store.lock():
        for stage in STAGES:
            pipeline.run_stage(stage,with_deps=True)
            assert store.completed(stage)
            if original is None: original={k:v.detach().clone() for k,v in pipeline.engine.backend.model.state_dict().items()}
        assert all(torch.equal(v,pipeline.engine.backend.model.state_dict()[k]) for k,v in original.items())
    assert (store.stage_dir("report")/"report.html").exists()
    assert (store.stage_dir("report")/"figures/heldout_response.png").exists()
    assert store.result("evaluate")["needs_human_quality_review"]
    assert store.result("inspect")["capabilities"]["toy_fixture"]
    before=len(store.cache)
    with store.lock():pipeline.run_stage("sweep")
    assert len(store.cache)==before

@pytest.mark.integration
def test_auto_stops_inconclusive_not_synthetic_success(store):
    p=Pipeline(store)
    with store.lock():decisions=p.auto(max_steps=20)
    assert store.completed("report")
    assert decisions[-1]["next_stage"] is None
    assert not any((store.root/x).exists() for x in ["model.safetensors","export"])

@pytest.mark.integration
def test_fork_preserves_compatible_directions(store,tmp_path):
    p=Pipeline(store)
    with store.lock():
        p.run_stage("directions",with_deps=True)
        changed=deepcopy(store.config);changed["search"]["ablations"]=[1.0]
        child=store.fork(tmp_path/"branch",changed)
    assert child.completed("inspect") and child.completed("capture") and child.completed("directions")
    assert not child.completed("sweep")
    other=deepcopy(store.config);other["model"]["dtype"]="bf16"
    with store.lock():child2=store.fork(tmp_path/"branch2",other)
    assert not child2.completed("directions")
    changed_identity=deepcopy(store.config);changed_identity['identity_schema']=1
    with store.lock():child3=store.fork(tmp_path/"branch3",changed_identity)
    assert not child3.completed("inspect")

@pytest.mark.integration
def test_cache_roundtrip_and_actual_noop_generation(store):
    from ablationlab.engine import Engine
    from ablationlab.interventions import Intervention
    p=Pipeline(store)
    with store.lock():
        p.run_stage("directions",with_deps=True)
        records=p._eval_rows();eng=p.engine
        first=eng.generate(records);cache_count=len(store.cache)
        second=eng.generate(records)
        assert [r["token_ids"] for r in first]==[r["token_ids"] for r in second]
        assert len(store.cache)==cache_count
        zero=eng.generate(records,p.bundle(),Intervention((1,),strength=0))
        assert [r["token_ids"] for r in first]==[r["token_ids"] for r in zero]

@pytest.mark.integration
def test_test_baseline_is_opt_in_and_invalidates_fork(store,tmp_path):
    with store.lock():Pipeline(store).run_stage("baseline",with_deps=True)
    original=store.result("baseline")
    assert original["test_baseline_included"] is False
    assert all(r["split"]!="test" for r in read_json(store.stage_dir("baseline")/"outputs.json"))
    changed=deepcopy(store.config)
    changed["evaluation"]["baseline_include_test"]=True
    with store.lock():child=store.fork(tmp_path/"with_test_baseline",changed)
    assert child.completed("inspect") and not child.completed("baseline")
    with child.lock():Pipeline(child).run_stage("baseline")
    result=child.result("baseline")
    assert result["test_baseline_included"] is True
    assert result["mean_score"]==original["mean_score"]
    assert result["cap_rate"]==original["cap_rate"]
    assert len([r for r in read_json(child.stage_dir("baseline")/"outputs.json") if r["split"]=="test"])==2

@pytest.mark.integration
def test_engine_records_custom_score_evidence(tmp_path,toy_config,dataset_file):
    config=deepcopy(toy_config)
    config["behavior"].update(metric="custom",plugin="examples.custom_scorer:score")
    data,audit=load_dataset(dataset_file,2)
    store=RunStore.create(tmp_path/"scorer_evidence",config,data,audit)
    pipeline=Pipeline(store)
    pipeline.engine.scorer.custom=lambda **kwargs:{"score":1,"details":{"status":"provisional"}}
    with store.lock():pipeline.run_stage("baseline",with_deps=True)
    outputs=read_json(store.stage_dir("baseline")/"outputs.json")
    assert all(r["score"]==1 and r["score_details"]=={"status":"provisional"} for r in outputs)

@pytest.mark.parametrize("bias",[False,True])
@pytest.mark.parametrize("strength",[0,.25,1.])
def test_weight_projection_matches_runtime(bias,strength):
    torch.manual_seed(1)
    module=torch.nn.Linear(7,5,bias=bias)
    q=torch.linalg.qr(torch.randn(5,2)).Q.T
    x=torch.randn(3,4,7)
    reference=transform(module(x),q,strength,"ablate")
    project_linear(module,q,strength)
    assert torch.allclose(reference,module(x),atol=1e-6)

@pytest.mark.integration
def test_dense_export_copy_restore_reload(store,tmp_path):
    p=Pipeline(store)
    with store.lock():
        p.run_stage("directions",with_deps=True)
        plan=make_export_plan(store)
        assert plan["edits"]==[]
        plan["edits"]=[{"layer":1,"site":"attention","strength":.25}]
        filename=tmp_path/"plan.json";write_json(filename,plan)
        output=tmp_path/"edited"
        result=export_checkpoint(store,filename,output)
        assert result["original_model_restored"]
        assert result["runtime_vs_weight_max_logit_delta"]<1e-5
        assert (output/"model.safetensors").exists()
        config=deepcopy(store.config);config["model"]["id"]=str(output)
        reloaded=Backend(config)
        assert reloaded.is_toy
        with pytest.raises(LabError,match="already exist"):export_checkpoint(store,filename,output)

def test_export_no_implicit_winner(store,tmp_path):
    with store.lock():
        Pipeline(store).run_stage("directions",with_deps=True)
        filename=tmp_path/"plan.json";write_json(filename,make_export_plan(store))
        with pytest.raises(LabError,match="no edits"):export_checkpoint(store,filename,tmp_path/"out")

def test_cli_status_plan_validation(store,tmp_path):
    assert main(["status","--run",str(store.root)])==0
    assert main(["plan","--run",str(store.root)])==0
    assert main(["doctor"])==0

def test_resume_interrupted_stage_from_generation_cache(store,monkeypatch):
    p=Pipeline(store)
    with store.lock():
        p.run_stage("inspect")
        original=p.engine._generate_retry
        calls=0
        def flaky(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise KeyboardInterrupt()
            return original(*args,**kwargs)
        monkeypatch.setattr(p.engine,"_generate_retry",flaky)
        with pytest.raises(KeyboardInterrupt):p.run_stage("baseline")
        assert len(store.cache)>0
        cached_keys=set(store.cache)
        monkeypatch.setattr(p.engine,"_generate_retry",original)
        p.run_stage("baseline")
        assert store.completed("baseline")
        assert cached_keys<=set(store.cache)

@pytest.mark.integration
def test_moe_export_commutes_with_scalar_mixing(tmp_path,toy_config,dataset_file):
    toy_config["model"]["id"]="toy:moe"
    data,audit=load_dataset(dataset_file,2)
    store=RunStore.create(tmp_path/"moe_run",toy_config,data,audit)
    p=Pipeline(store)
    with store.lock():
        p.run_stage("directions",with_deps=True)
        plan=make_export_plan(store);plan["edits"]=[{"layer":1,"site":"mlp","strength":.5}]
        filename=tmp_path/"plan.json";write_json(filename,plan)
        result=export_checkpoint(store,filename,tmp_path/"moe_export")
        assert len(result["changes"])==3
        assert result["runtime_vs_weight_max_logit_delta"]<1e-5
        assert result["original_model_restored"]

def test_cached_reference_nll_intervention_is_applied(toy_config):
    from ablationlab.directions import DirectionBundle
    from ablationlab.interventions import Intervention
    b=Backend(toy_config)
    q=torch.eye(24)[0].expand(4,1,24).clone()
    bundle=DirectionBundle(q,torch.zeros(4,24),torch.zeros(4,24),torch.ones(4),{})
    record={"neutral":[{"role":"user","content":"test"}]}
    tokens=[10,11,12]
    baseline=b.reference_nll(record,tokens)
    zero=b.reference_nll(record,tokens,bundle,Intervention((1,),operation="steer",strength=0))
    changed=b.reference_nll(record,tokens,bundle,Intervention((1,),operation="steer",strength=8))
    assert baseline==zero
    assert abs(changed-baseline)>1e-5

@pytest.mark.integration
def test_checkpoint_comparison_same_model_zero_change(store,tmp_path):
    from ablationlab.compare import evaluate_checkpoint
    with store.lock():
        result=evaluate_checkpoint(store,"toy:dense",tmp_path/"comparison")
    assert all(r["mean_gain"]==0 for r in result["summary"])
    assert (tmp_path/"comparison/comparison.json").exists()
