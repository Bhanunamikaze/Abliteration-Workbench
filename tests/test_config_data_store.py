import json
from pathlib import Path
import pytest
from ablationlab.util import LabError,BudgetReached,write_json,append_jsonl,read_journal
from ablationlab.config import load_config,merge,DEFAULTS
from ablationlab.data import load_dataset
from ablationlab.store import RunStore
from ablationlab.metrics import Scorer

@pytest.mark.parametrize("bad",[{"unknown":1},{"generation":{"batch_siz":3}},{"search":{"typo":.1}}])
def test_unknown_config_rejected(bad):
    with pytest.raises(LabError):merge(DEFAULTS,bad)

def test_dotted_override_and_relative_dataset(tmp_path,dataset_file):
    p=tmp_path/"config.json";write_json(p,{"dataset":dataset_file.name})
    c=load_config(p,["generation.batch_size=2","model.device=cpu"])
    assert c["generation"]["batch_size"]==2
    assert c["dataset"]==str(dataset_file)

def test_local_scorer_path_resolves_from_config_and_loads_without_pythonpath(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    plugin = source / "scorer.py"
    plugin.write_text("def score(*, text, token_count, record):\n    return 3.0 if text else 0.0\n")
    config = source / "experiment.json"
    write_json(config, {"behavior": {"metric": "custom", "plugin": "./scorer.py:score"}})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    resolved = load_config(config)
    assert resolved["behavior"]["plugin"] == f"{plugin}:score"
    assert Scorer(resolved["behavior"])("answer", 1, {}) == 3.0

def test_local_dotted_scorer_resolves_without_pythonpath(tmp_path, monkeypatch):
    root = tmp_path / "project"
    module = root / "examples" / "scorer.py"
    module.parent.mkdir(parents=True)
    module.write_text("def score(*, text, token_count, record):\n    return 2.0\n")
    config = root / "examples" / "experiment.json"
    write_json(config, {"behavior": {"metric": "custom", "plugin": "examples.scorer:score"}})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    resolved = load_config(config)
    assert resolved["behavior"]["plugin"] == f"{module}:score"
    assert Scorer(resolved["behavior"])("answer", 1, {}) == 2.0

def test_immutable_run_detects_changed_scorer(tmp_path, toy_config, dataset_file):
    plugin = tmp_path / "scorer.py"
    plugin.write_text("def score(*, text, token_count, record):\n    return 1.0\n")
    toy_config["behavior"].update(metric="custom", plugin=f"{plugin}:score")
    rows, audit = load_dataset(dataset_file, 2)
    run = RunStore.create(tmp_path / "run", toy_config, rows, audit)
    assert run.meta["scorer_provenance"]["source_path"] == str(plugin)
    assert len(run.meta["scorer_provenance"]["sha256"]) == 64
    plugin.write_text("def score(*, text, token_count, record):\n    return 2.0\n")
    with pytest.raises(LabError, match="Custom scorer implementation changed"):
        RunStore(run.root)

def test_local_scorer_load_uses_current_fingerprinted_source(tmp_path):
    plugin = tmp_path / "scorer.py"
    behavior = dict(DEFAULTS["behavior"], metric="custom", plugin=f"{plugin}:score")
    plugin.write_text("def score(*, text, token_count, record):\n    return 1.0\n")
    assert Scorer(behavior)("answer", 1, {}) == 1.0
    plugin.write_text("def score(*, text, token_count, record):\n    return 2.0\n")
    assert Scorer(behavior)("answer", 1, {}) == 2.0

@pytest.mark.parametrize("update", [
    {"search": {"mode": "unbounded"}},
    {"search": {"reference": "bogus"}},
    {"search": {"persistent_max_span_layers": 0}},
    {"search": {"persistent_cluster_gap": 0}},
    {"search": {"max_cap_confirmations": 3}},
    {"generation": {"confirm_max_new_tokens": 0}},
])
def test_new_search_limits_are_validated(update):
    from ablationlab.config import validate_config
    with pytest.raises(LabError):
        validate_config(merge(DEFAULTS, update))

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
