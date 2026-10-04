"""Frozen replication, paired evidence, provenance, and promotion contracts."""
from copy import deepcopy

import pytest

from ablationlab.bundle import package_runtime_bundle
from ablationlab.cli import main
from ablationlab.data import load_replication_dataset
from ablationlab.engine import Engine
from ablationlab.interventions import Intervention
from ablationlab.metrics import binary_paired_transitions
from ablationlab.pipeline import Pipeline
from ablationlab.replication import (_decide, _pair_summary, ReplicationStore,
                                    dataset_from_historical_outputs, open_or_create,
                                    run_replication, write_recipe)
from ablationlab.toy import TinyLM
from ablationlab.util import LabError, digest, file_hash, read_json, write_json


def _outputs(scores, split="test", *, capped=False, review=False):
    return [{"id":f"{split}{i}","split":split,"score":float(score),"tokens":2,
             "token_ids":[3,4],"hit_limit":capped,"repeat4":0.,"empty":False,
             "required_term_fraction":None,
             "score_details":{"review_required":review,
                              "review_reasons":["classifier_input_truncated"] if review else []}}
            for i,score in enumerate(scores)]


def _evidence(config, *, capped=False, review=False, control_regression=False):
    baseline=[1]*160+[0]*32
    changed=[0]*59+[1]*101+[1]*2+[0]*30
    records=[{"id":f"test{i}","split":"test"} for i in range(192)]
    test=_pair_summary(records,_outputs(baseline,capped=capped),
                       _outputs(changed,capped=capped,review=review),config)
    cb=[1]*4+[0]*96
    ca=[0]*2+[1]*2+[1 if control_regression else 0]+[0]*95
    control=_pair_summary([{"id":f"control{i}","split":"control"} for i in range(100)],
                          _outputs(cb,"control"),_outputs(ca,"control"),config)
    control["absolute_score_drift_fraction"]=.02 if not control_regression else .03
    return test,control


def test_binary_paired_evidence_uses_discordant_pairs(toy_config):
    test,control=_evidence(toy_config)
    binary=test["binary_transitions"]
    assert test["win_rate"]<.5
    assert (binary["baseline_positive_count"],binary["intervention_positive_count"])==(160,103)
    assert (binary["favorable_transitions"],binary["unfavorable_transitions"],binary["ties"])==(59,2,131)
    assert binary["conditional_improvement_fraction"]==59/160
    assert binary["paired_absolute_effect"]==57/192
    assert binary["binary_transition_test"]["p_value"]<1e-12
    meta={"evaluation_policy":{"binary_min_samples":20,"binary_min_discordant":5,
                                  "binary_min_favorable_fraction":.75,
                                  "binary_max_regression_fraction":.05,
                                  "binary_max_control_regression_fraction":0.,
                                  "max_scorer_review_fraction":1.},
          "source_evidence":{"validation_specificity_pass":True,
                             "selection_quality_confirmed":True}}
    assert _decide(test,control,.1,meta,toy_config)["passed"]
    capped_test,_=_evidence(toy_config,capped=True)
    censored=_decide(capped_test,control,.1,meta,toy_config)
    assert censored["status"]=="promising_censored"
    assert not censored["passed"] and not censored["promotion_gates"]["test_censoring_pass"]
    capped_control=deepcopy(control);capped_control["cap_rate"]=1.
    assert _decide(test,capped_control,.1,meta,toy_config)["status"]=="promising_censored"
    _,bad_control=_evidence(toy_config,control_regression=True)
    assert not _decide(test,bad_control,.1,meta,toy_config)["promotion_gates"]["control_regression"]
    reviewed,_=_evidence(toy_config,review=True)
    assert reviewed["intervention_review"]["reason_counts"]["classifier_input_truncated"]==192


def test_increase_and_continuous_domains():
    result=binary_paired_transitions([0,0,1,1],[1,0,0,1],"increase")
    assert result["favorable_transitions"]==1
    assert result["unfavorable_transitions"]==1
    assert result["opportunity_count"]==2
    assert binary_paired_transitions([3,4],[2,3]) is None


def test_continuous_scores_keep_win_rate_gate(toy_config):
    test,control=_evidence(toy_config)
    records=[{"id":f"test{i}","split":"test"} for i in range(20)]
    test=_pair_summary(records,_outputs([2]*20),_outputs([1.5]*10+[2]*10),toy_config)
    assert test["binary_transitions"] is None
    meta={"evaluation_policy":{"binary_min_samples":20,"binary_min_discordant":5,
                                  "binary_min_favorable_fraction":.75,
                                  "binary_max_regression_fraction":.05,
                                  "binary_max_control_regression_fraction":0.,
                                  "max_scorer_review_fraction":1.},
          "source_evidence":{"validation_specificity_pass":True,
                             "selection_quality_confirmed":True}}
    decision=_decide(test,control,.1,meta,toy_config)
    assert not decision["passed"]
    assert not decision["promotion_gates"]["continuous_win_rate"]


@pytest.fixture
def frozen_source(tmp_path,toy_config,dataset_file):
    from ablationlab.data import load_dataset
    from ablationlab.store import RunStore
    model=tmp_path/"model"
    TinyLM().save_pretrained(model)
    config=deepcopy(toy_config)
    config["model"]["id"]=str(model)
    scorer=tmp_path/"scorer.py"
    scorer.write_text("def score(*, text, token_count, record):\n"
                      "    return {'score': float('a' in text.lower()), "
                      "'details': {'review_required': True, "
                      "'review_reasons': ['classifier_input_truncated']}}\n")
    config["behavior"].update(metric="custom",plugin=f"{scorer}:score")
    config["evaluation"]["reference_tokens"]=0
    config["generation"]["confirm_max_new_tokens"]=12
    rows,audit=load_dataset(dataset_file,2)
    source=RunStore.create(tmp_path/"source",config,rows,audit)
    with source.lock():Pipeline(source).run_stage("directions",with_deps=True)
    selected=Intervention((1,),site="residual",operation="ablate",strength=1.,
                          reference="negative",scope="last",phase="both").serial()
    path=source.stage_dir("evaluate")/"result.json"
    write_json(path,{"selected_intervention":selected,"passed":False,
                     "evaluation_max_new_tokens":6,"validation_specificity_pass":True,
                     "selection_quality_confirmed":True,
                     "candidate_evidence_status":"promising"})
    state=source.state()
    state["stages"]["evaluate"]={"status":"complete","artifacts":{
        str(path.relative_to(source.root)):file_hash(path)}}
    write_json(source.root/"state.json",state)
    dataset=tmp_path/"replication-data.json"
    write_json(dataset,[{"id":f"new-test-{i}","split":"test","neutral":f"fresh test {i}"}
                        for i in range(20)]+[
                        {"id":f"new-control-{i}","split":"control","neutral":f"fresh control {i}"}
                        for i in range(4)])
    return source,dataset


def test_replication_freezes_source_and_detects_mutation(frozen_source,tmp_path,monkeypatch):
    source,dataset=frozen_source
    fingerprints={name:file_hash(source.stage_dir("directions")/name)
                  for name in ("directions.json","directions.safetensors")}
    replica=open_or_create(source.root,dataset,tmp_path/"replica")
    assert replica.meta["intervention"]==source.result("evaluate")["selected_intervention"]
    assert replica.meta["directions_fingerprint"]
    assert [r["split"] for r in replica.dataset()]==["test"]*20+["control"]*4
    monkeypatch.setattr(Pipeline,"run_stage",lambda *a,**kw:pytest.fail("discovery called"))
    monkeypatch.setattr("ablationlab.replication._decide",lambda *a,**kw:{
        "passed":False,"status":"failed_replication","promotion_gates":{},
        "failed_gates":[],"semantic_confirmation":None})
    result=run_replication(replica)
    assert result["intervention"]==replica.meta["intervention"]
    assert result["generation_cap"]==6
    assert result["test"]["intervention_review"]["requiring_review"]==20
    assert "Classifier input truncation occurred" in (replica.root/"report.md").read_text()
    assert all(file_hash(source.stage_dir("directions")/name)==value
               for name,value in fingerprints.items())
    with pytest.raises(LabError,match="passing frozen replication"):
        write_recipe(replica,tmp_path/"recipe.json")
    with pytest.raises(LabError,match="passing frozen replication"):
        package_runtime_bundle(source,tmp_path/"blocked-bundle",replication=replica)
    assert main(["replication-status","--run",str(replica.root)])==0
    assert main(["replication-report","--run",str(replica.root)])==0
    rows=read_json(dataset);rows[0]["neutral"]="changed";write_json(dataset,rows)
    with pytest.raises(LabError,match="input dataset changed"):
        open_or_create(source.root,dataset,replica.root)
    changed=replica.root/"directions"/"directions.json"
    changed.write_text(changed.read_text()+" ")
    with pytest.raises(LabError,match="Frozen direction file changed"):
        ReplicationStore(replica.root)


def test_replication_resume_and_higher_cap_retry(frozen_source,tmp_path,monkeypatch):
    source,dataset=frozen_source
    replica=open_or_create(source.root,dataset,tmp_path/"retry")
    original=Engine._generate_retry
    calls=0
    def interrupt_once(self,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise KeyboardInterrupt()
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Engine,"_generate_retry",interrupt_once)
    with pytest.raises(KeyboardInterrupt):run_replication(replica)
    cached=len(replica.cache)
    assert cached>0
    monkeypatch.setattr(Engine,"_generate_retry",original)
    # Force a censored initial decision to exercise the actual matched retry.
    from ablationlab import replication as module
    decide=module._decide
    def censored_once(test,control,nll,meta,config):
        result=decide(test,control,nll,meta,config)
        if test["baseline_cap_rate"]>=0 and test["cap_rate"]>=0 and not (replica.root/"attempts"/"initial"/"summary.json").exists():
            result.update(status="promising_censored",passed=False)
        return result
    monkeypatch.setattr(module,"_decide",censored_once)
    summary=run_replication(ReplicationStore(replica.root))
    assert summary["higher_cap_retry_performed"]
    assert [x["generation_cap"] for x in summary["attempts"]]==[6,12]
    assert len(read_json(replica.root/"attempts"/"initial"/"baseline_outputs.json"))==20
    assert read_json(replica.root/"state.json")["generation_count"]>=cached
    assert "initial result" in (replica.root/"report.md").read_text()


def test_replication_provenance_rejects_changed_source(frozen_source,tmp_path):
    source,dataset=frozen_source
    replica=open_or_create(source.root,dataset,tmp_path/"replica")
    path=source.stage_dir("evaluate")/"result.json"
    value=read_json(path);value["selected_intervention"]["strength"]=.5
    write_json(path,value)
    with pytest.raises(LabError):ReplicationStore(replica.root)


def test_replication_rejects_scorer_change(frozen_source,tmp_path):
    source,dataset=frozen_source
    replica=open_or_create(source.root,dataset,tmp_path/"replica")
    scorer=tmp_path/"scorer.py"
    scorer.write_text(scorer.read_text()+"\n# changed\n")
    with pytest.raises(LabError,match="scorer source changed"):
        ReplicationStore(replica.root)


def test_replication_rejects_model_identity_change(frozen_source,tmp_path,monkeypatch):
    source,dataset=frozen_source
    replica=open_or_create(source.root,dataset,tmp_path/"replica")
    from ablationlab import replication as module
    class WrongEngine:
        def __init__(self,store):self.backend=type("WrongBackend",(),{"identity":{"other":"model"}})()
    monkeypatch.setattr(module,"Engine",WrongEngine)
    with pytest.raises(LabError,match="identity differs from source"):
        run_replication(replica)


def test_direction_fingerprint_mismatch_fails(frozen_source,tmp_path):
    source,dataset=frozen_source
    replica=open_or_create(source.root,dataset,tmp_path/"replica")
    meta=read_json(replica.root/"replication.json")
    meta["directions_fingerprint"]="different"
    write_json(replica.root/"replication.json",meta)
    state=replica.state();state["manifest_hash"]=digest(meta)
    write_json(replica.root/"state.json",state)
    with pytest.raises(LabError,match="Frozen direction fingerprint changed"):
        ReplicationStore(replica.root)


def test_passing_replication_authorizes_recipe_and_bundle(frozen_source,tmp_path,monkeypatch):
    source,dataset=frozen_source
    replica=open_or_create(source.root,dataset,tmp_path/"replica")
    monkeypatch.setattr("ablationlab.replication._decide",lambda *a,**kw:{
        "passed":True,"status":"validated_replication",
        "promotion_gates":{"test_censoring_pass":True},
        "failed_gates":[],"semantic_confirmation":False})
    summary=run_replication(replica)
    assert summary["passed"]
    recipe=write_recipe(replica,tmp_path/"recipe.json")
    assert recipe["validated_by"]=="external_frozen_intervention_replication"
    assert recipe["directions_fingerprint"]==replica.meta["directions_fingerprint"]
    result=package_runtime_bundle(source,tmp_path/"runtime",replication=replica)
    assert result["exact_token_matches"]==24
    manifest=read_json(tmp_path/"runtime"/"bundle.json")
    assert manifest["replication"]["dataset_hash"]==replica.meta["replication_dataset_hash"]
    assert manifest["intervention"]==replica.meta["intervention"]


def test_historical_output_converter_only_copies_prompts(tmp_path):
    outputs=tmp_path/"outputs";outputs.mkdir()
    for split in ("test","control"):
        row={"id":split,"split":split,"messages":[{"role":"user","content":split}],
             "score":1,"text":"old answer","category":"example"}
        for arm in ("baseline","intervention"):
            write_json(outputs/f"{split}_{arm}.json",[row])
    result=dataset_from_historical_outputs(outputs,tmp_path/"new.json")
    rows,audit=load_replication_dataset(result["destination"])
    assert audit["counts"]=={"test":1,"control":1}
    assert all("score" not in row and "text" not in row for row in rows)
