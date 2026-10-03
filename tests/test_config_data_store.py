import json
from pathlib import Path
import pytest
from ablationlab.util import LabError,BudgetReached,write_json,append_jsonl,read_journal
from ablationlab.config import load_config,merge,DEFAULTS
from ablationlab.data import load_dataset
from ablationlab.store import RunStore

@pytest.mark.parametrize("bad",[{"unknown":1},{"generation":{"batch_siz":3}},{"search":{"typo":.1}}])
def test_unknown_config_rejected(bad):
    with pytest.raises(LabError):merge(DEFAULTS,bad)

def test_dotted_override_and_relative_dataset(tmp_path,dataset_file):
    p=tmp_path/"config.json";write_json(p,{"dataset":dataset_file.name})
    c=load_config(p,["generation.batch_size=2","model.device=cpu"])
    assert c["generation"]["batch_size"]==2
    assert c["dataset"]==str(dataset_file)

def test_group_leakage(dataset_file):
    rows=json.loads(dataset_file.read_text());rows[0]["group"]="duplicate";rows[-1]["group"]="duplicate"
    write_json(dataset_file,rows)
    with pytest.raises(LabError,match="Group leakage"):load_dataset(dataset_file,2)

def test_exact_prompt_leakage(dataset_file):
    rows=json.loads(dataset_file.read_text());rows[-1]["neutral"]=rows[0]["neutral"]
    write_json(dataset_file,rows)
    with pytest.raises(LabError,match="crosses splits"):load_dataset(dataset_file,2)

def test_inplace_config_changes_rejected(store):
    c=store.config;c["generation"]["batch_size"]=3;write_json(store.root/"config.json",c)
    with pytest.raises(LabError,match="edited in place"):RunStore(store.root)

def test_stage_resume_and_artifact_integrity(store):
    with store.stage("inspect") as p:write_json(p/"result.json",{"ok":True})
    assert store.completed("inspect")
    write_json(store.stage_dir("inspect")/"result.json",{"ok":False})
    with pytest.raises(LabError,match="Artifact missing/changed"):store.completed("inspect")

def test_failure_marked_no_fake_complete(store):
    with pytest.raises(RuntimeError):
        with store.stage("inspect"):raise RuntimeError("failure")
    assert not store.completed("inspect")
    assert store.state()["stages"]["inspect"]["status"]=="failed"

def test_stage_dependencies(store):
    with pytest.raises(LabError,match="requires"):
        with store.stage("sweep"):pass

def test_journal_truncated_tail_repair(tmp_path):
    p=tmp_path/"journal.jsonl";append_jsonl(p,{"x":1})
    with p.open("ab") as f:f.write(b'{"x":')
    assert read_journal(p,ignore_tail=True)==[{"x":1}]
    assert p.read_bytes().endswith(b'{"x":')
    assert read_journal(p,repair_tail=True)==[{"x":1}]
    append_jsonl(p,{"x":2});assert len(read_journal(p))==2

def test_journal_valid_no_newline(tmp_path):
    p=tmp_path/"journal";p.write_text('{"x":1}')
    read_journal(p,repair_tail=True);append_jsonl(p,{"x":2})
    assert len(read_journal(p))==2

def test_journal_does_not_ignore_middle_corruption(tmp_path):
    p=tmp_path/"journal";p.write_text('{"x":1}\nBROKEN\n{"x":2}\n')
    with pytest.raises(LabError):read_journal(p,repair_tail=True)

def test_lock_exclusive(store):
    with store.lock():
        with pytest.raises(LabError):
            with store.lock():pass
    assert not (store.root/".run.lock").exists()

def test_budget_blocks_without_deleting_cache(store):
    store.config["budget"]["max_generations"]=1
    store.put_generation({"job_key":"test","tokens":1})
    with pytest.raises(BudgetReached):store.check_budget(1)
    assert "test" in store.cache

def test_reset_archives_and_invalidates(store):
    with store.stage("inspect") as p:write_json(p/"result.json",{"ok":True})
    with store.stage("baseline") as p:write_json(p/"result.json",{"ok":True})
    store.reset("inspect")
    assert not store.completed("baseline") and list((store.root/"history").rglob("result.json"))
