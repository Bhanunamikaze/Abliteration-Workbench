from ablationlab.report import build_report
from ablationlab.util import read_json


def _result(reference, gain, status, behaviorally_promising):
    return {
        "label": "L3/residual", "intervention": {"layers": [3], "site": "residual",
                                             "operation": "ablate", "reference": reference,
                                             "strength": 1.0},
        "mean_baseline": .75, "mean_intervention": .75 - gain,
        "gain_fraction_of_baseline_mean": gain, "win_rate": .75,
        "ci95_low": .25, "ci95_high": 1.0,
        "random_control_max_gain_fraction": 0.0,
        "cap_rate": 1.0 if status == "promising_censored" else 0.0,
        "uncensored_pair_fraction": 0.0 if status == "promising_censored" else 1.0,
        "evidence_status": status, "behaviorally_promising": behaviorally_promising,
        "generation_cap": 512,
        "confirmation": {"attempted": True, "final_max_new_tokens": 512, "final_status": status},
    }


class _Store:
    def __init__(self, root, metric="custom"):
        self.root = root
        self.config = {"behavior": {"name": "contrast", "metric": metric},
                       "model": {"id": "toy"}, "search": {"mode": "explore", "max_cap_rate": .25}}
        zero = _result("zero", 0, "rejected", False)
        negative = _result("negative", .75, "promising_censored", metric != "tokens")
        self.results = {"inspect": {}, "baseline": {}, "capture": {},
                        "directions": {"eligible_layers": [3], "calibrated": True},
                        "sweep": {"summary": [], "candidates": []},
                        "trace": {"summary": [zero, negative],
                                  "reference_comparison_performed": True},
                        "evaluate": {"passed": False,
                                     "selected_intervention": negative["intervention"],
                                     "candidate_source_stage": "trace",
                                     "candidate_evidence_status": "promising_censored",
                                     "selection_reason": "Strongest causal validation effect",
                                     "evaluation_status": "promising_censored"}}

    def completed(self, stage):
        return stage in self.results

    def result(self, stage):
        return self.results[stage]


def test_report_surfaces_reference_censoring_confirmation_and_selection(tmp_path, monkeypatch):
    monkeypatch.setattr("ablationlab.plotting.plot_run", lambda store, destination: [])
    store = _Store(tmp_path)
    destination = tmp_path / "report"
    destination.mkdir()
    build_report(store, destination)
    markdown = (destination / "report.md").read_text()
    html = (destination / "report.html").read_text()
    assert "Strong behavioral effect detected, but generation censoring prevents final promotion" in markdown
    assert "Automatic residual reference comparison performed: **yes**" in markdown
    assert "zero" in markdown and "negative" in markdown
    assert "attempted at 512 tokens" in markdown
    assert "Selected from: `trace`" in markdown
    assert "Strongest causal validation effect" in html


def test_report_does_not_claim_censored_length_effect(tmp_path, monkeypatch):
    monkeypatch.setattr("ablationlab.plotting.plot_run", lambda store, destination: [])
    store = _Store(tmp_path, metric="tokens")
    destination = tmp_path / "report"
    destination.mkdir()
    build_report(store, destination)
    markdown = (destination / "report.md").read_text()
    assert "Censored length evidence requires uncensored confirmation" in markdown
    assert "Strong behavioral effect detected" not in markdown


def test_residual_reference_plot_marks_censoring(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    from ablationlab.plotting import plot_run

    store = _Store(tmp_path)
    store.results = {"ablate": {"summary": [
        _result("zero", 0, "rejected", False),
        _result("negative", .75, "promising_censored", True)]}}
    files = plot_run(store, tmp_path / "figures")
    assert any(path.endswith("ablate_reference_comparison.png") for path in files)
    metadata = read_json(tmp_path / "figures" / "plot_manifest.json")["candidate_metadata"]
    assert {row["reference"] for row in metadata} == {"zero", "negative"}
    assert any(row["cap_rate"] == 1.0 and row["evidence_status"] == "promising_censored"
               for row in metadata)
