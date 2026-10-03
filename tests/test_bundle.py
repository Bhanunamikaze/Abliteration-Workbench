import pytest

from ablationlab.bundle import RuntimeBundle, package_runtime_bundle
from ablationlab.util import LabError, file_hash, write_json


def test_bundle_requires_passing_held_out_evaluation(store, tmp_path, monkeypatch):
    with pytest.raises(LabError, match="passing held-out evaluation"):
        package_runtime_bundle(store, tmp_path / "unvalidated")
    assert not (tmp_path / "unvalidated").exists()

    monkeypatch.setattr(store, "completed", lambda stage: stage == "evaluate")
    monkeypatch.setattr(store, "result", lambda stage: {"passed": False})
    with pytest.raises(LabError, match="passing held-out evaluation"):
        package_runtime_bundle(store, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_runtime_bundle_rejects_changed_file_before_model_load(tmp_path):
    protected = tmp_path / "directions" / "directions.json"
    protected.parent.mkdir()
    protected.write_text("original")
    write_json(tmp_path / "bundle.json", {
        "schema_version": 1,
        "kind": "activation_runtime_bundle",
        "files": {"directions/directions.json": file_hash(protected)},
    })
    protected.write_text("modified")
    with pytest.raises(LabError, match="Bundled file missing or changed"):
        RuntimeBundle(tmp_path)
