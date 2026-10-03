from pathlib import Path
import pytest
from ablationlab.legacy import audit_legacy
from ablationlab.util import file_hash

FIXTURE=Path(__file__).parent/"fixtures/legacy"

def test_replay_user_sweep_no_hardcoded_layers(tmp_path):
    hashes={p.name:file_hash(p) for p in FIXTURE.iterdir() if p.suffix==".csv"}
    audit=audit_legacy(FIXTURE,tmp_path/"audit")
    assert audit["sweep_rows"]==560
    winner=max(audit["sweep_summary"],key=lambda r:r["legacy_candidate_score"])
    assert winner["layer"]==16
    assert winner["legacy_candidate_score"]==pytest.approx(.8155359033143397)
    assert winner["avg_symmetric_effect_tokens"]==pytest.approx(516.875)
    assert {p.name:file_hash(p) for p in FIXTURE.iterdir() if p.suffix==".csv"}==hashes

def test_actual_ratio_artifact_corrected_without_overwriting(tmp_path):
    audit=audit_legacy(FIXTURE,tmp_path/"audit",False)
    value=audit["source13_layer15_full"]
    assert value["legacy_mean_individual_abs_ratio"]==pytest.approx(14.317313824952969)
    assert value["ratio_of_mean_abs"]==pytest.approx(2.9824034316499923)
    assert value["sign_agreement_eligible"]<1

def test_per_sample_and_aggregate_percentage_retained(tmp_path):
    audit=audit_legacy(FIXTURE,tmp_path/"audit",False)
    row=next(r for r in audit["behavior_summary"] if r["layer"]==13 and r["strength"]==1)
    assert row["reduction_fraction_of_means"]!=pytest.approx(row["mean_per_example_reduction_fraction"])
