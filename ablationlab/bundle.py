"""Portable local model bundle for a held-out-tested runtime intervention.

The model weights are copied unchanged. The bundled runner applies the selected
activation hook on every generation, so it does not depend on the original run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .backend import Backend
from .directions import DirectionBundle
from .interventions import Intervention
from .util import LabError, digest, file_hash, read_json, write_json


def _model_source(model: dict) -> tuple[Path, str | None]:
    local = Path(model["id"])
    if local.is_dir():
        return local.resolve(), None
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(model["id"], "config.json", revision=model["revision"])
    if not isinstance(cached, str):
        raise LabError("Model snapshot is not cached locally; load the model first before bundling")
    snapshot = Path(cached).parent
    commit = snapshot.name if snapshot.parent.name == "snapshots" else None
    return snapshot, commit


def _copy_model(source: Path, destination: Path) -> dict[str, str]:
    hashes = {}
    for item in sorted(source.rglob("*")):
        if item.is_dir():
            continue
        if not item.is_file():
            raise LabError(f"Model snapshot contains a non-file entry: {item}")
        relative = item.relative_to(source)
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        with item.open("rb") as src, output.open("wb") as dst:
            for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
                hasher.update(block)
                dst.write(block)
        hashes[str(Path("model") / relative)] = hasher.hexdigest()
    if not any(name.endswith((".safetensors", ".bin")) for name in hashes):
        raise LabError("Model snapshot has no supported weight file")
    return hashes


def _runtime_files(destination: Path) -> dict[str, str]:
    package = Path(__file__).parent
    runtime = destination / "runtime" / "ablationlab"
    runtime.mkdir(parents=True)
    for item in package.glob("*.py"):
        shutil.copy2(item, runtime / item.name)
    source_license = package.parent / "LICENSE"
    if source_license.exists():
        shutil.copy2(source_license, destination / "LICENSE")
    (destination / "requirements.txt").write_text(
        "numpy>=1.24\ntorch>=2.5\nsafetensors>=0.4.5\n"
        "transformers>=4.57,<6\naccelerate>=1.0\nsentencepiece>=0.2\n"
    )
    (destination / "run_bundle.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "root = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(root / 'runtime'))\n"
        "from ablationlab.bundle import bundle_main\n"
        "raise SystemExit(bundle_main(root))\n"
    )
    return {str(item.relative_to(destination)): file_hash(item)
            for item in (destination / "runtime").rglob("*") if item.is_file()}


class RuntimeBundle:
    """Load a packaged checkpoint and its validated hook without the source run."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        manifest = read_json(self.root / "bundle.json")
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "activation_runtime_bundle":
            raise LabError("Unsupported runtime bundle manifest")
        for relative, expected in manifest["files"].items():
            target = (self.root / relative).resolve()
            if self.root not in target.parents or not target.is_file() or file_hash(target) != expected:
                raise LabError(f"Bundled file missing or changed: {relative}")
        config = deepcopy(manifest["backend_config"])
        config["model"]["id"] = str(self.root / "model")
        config["model"]["local_files_only"] = True
        self.backend = Backend(config)
        self.directions = DirectionBundle.load(self.root / "directions")
        if self.directions.fingerprint() != manifest["directions_fingerprint"]:
            raise LabError("Bundled direction fingerprint mismatch")
        # A copied checkpoint has a new local-path identity. The file hashes
        # above and package-time generation replay establish identical inputs.
        self.directions.metadata["model_identity"] = digest(self.backend.identity)
        raw = manifest["intervention"]
        self.intervention = Intervention(**{**raw, "layers": tuple(raw["layers"])})
        self.manifest = manifest

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> dict:
        record = {"neutral": [{"role": "user", "content": prompt}]}
        return self.backend.generate([record], self.directions, self.intervention, max_new_tokens)[0]


def _replay_evaluation(store, bundle: RuntimeBundle) -> dict:
    records = [r for split in ("test", "control") for r in store.dataset() if r["split"] == split]
    expected = read_json(store.stage_dir("evaluate") / "outputs.json")
    if [r["id"] for r in records] != [r["id"] for r in expected]:
        raise LabError("Stored held-out outputs do not align with dataset snapshot")
    batch_size = store.config["generation"]["batch_size"]
    observed = []
    for split in ("test", "control"):
        group = [r for r in records if r["split"] == split]
        for start in range(0, len(group), batch_size):
            chunk = group[start:start + batch_size]
            observed.extend(bundle.backend.generate(chunk, bundle.directions, bundle.intervention))
    mismatches = [r["id"] for r, old, new in zip(records, expected, observed)
                  if old["token_ids"] != new["token_ids"]]
    if mismatches:
        raise LabError("Packaged model failed exact held-out generation replay: " + ", ".join(mismatches))
    return {"replayed_test_prompts": sum(r["split"] == "test" for r in records),
            "replayed_control_prompts": sum(r["split"] == "control" for r in records),
            "exact_token_matches": len(records)}


def package_runtime_bundle(store, destination: str | Path) -> dict:
    if not store.completed("evaluate") or not store.result("evaluate").get("passed"):
        raise LabError("A passing held-out evaluation is required before packaging a runtime model")
    config = store.config
    model = config["model"]
    if model.get("backend_plugin") or model["quantization"] != "none" or model["allow_offload"]:
        raise LabError("Runtime bundle currently requires the standard nonquantized, nonoffloaded backend")
    selected = store.result("evaluate")["selected_intervention"]
    if selected.get("control_seed") is not None:
        raise LabError("A random-direction control cannot be packaged as a model")
    source, commit = _model_source(model)
    dest = Path(destination).resolve()
    if dest.exists():
        raise LabError("Bundle destination already exists")
    if dest == source or source in dest.parents or dest in source.parents:
        raise LabError("Bundle destination must be separate from source model files")
    if dest == store.root or dest in store.root.parents:
        raise LabError("Bundle cannot replace the run directory")
    original = DirectionBundle.load(store.stage_dir("directions"))
    if original.metadata.get("model_identity") != store.result("inspect")["model_identity"]:
        raise LabError("Direction and inspected model identities disagree")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".ablab-bundle-", dir=dest.parent))
    committed = False
    try:
        files = _copy_model(source, temp / "model")
        direction_source = store.stage_dir("directions")
        (temp / "directions").mkdir()
        for name in ("directions.safetensors", "directions.json"):
            shutil.copy2(direction_source / name, temp / "directions" / name)
            files[str(Path("directions") / name)] = file_hash(temp / "directions" / name)
        files.update(_runtime_files(temp))
        files["run_bundle.py"] = file_hash(temp / "run_bundle.py")
        backend_config = {"model": deepcopy(config["model"]), "generation": deepcopy(config["generation"]),
                          "identity_schema": config.get("identity_schema", 1)}
        manifest = {"schema_version": 1, "kind": "activation_runtime_bundle",
                    "weights_modified": False, "source_run": str(store.root),
                    "source_model_id": model["id"], "source_snapshot_commit": commit,
                    "backend_config": backend_config, "intervention": selected,
                    "directions_fingerprint": original.fingerprint(),
                    "evaluation": {"passed": True, "test": store.result("evaluate")["test"],
                                   "control_absolute_drift_fraction": store.result("evaluate")["control_absolute_drift_fraction"]},
                    "files": files,
                    "note": "Model weights are unchanged; run_bundle.py applies the validated activation hook during generation."}
        write_json(temp / "bundle.json", manifest)
        temp.rename(dest)
        committed = True
        replay = _replay_evaluation(store, RuntimeBundle(dest))
        manifest["package_validation"] = replay
        write_json(dest / "bundle.json", manifest)
        store.event("runtime_bundle_created", destination=str(dest), snapshot_commit=commit, replay=replay)
        return {"destination": str(dest), "snapshot_commit": commit, **replay}
    except BaseException:
        if committed:
            shutil.rmtree(dest, ignore_errors=True)
        raise
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def bundle_main(root: str | Path, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate with a packaged AblationLab runtime model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    result = RuntimeBundle(root).generate(args.prompt, args.max_new_tokens)
    print(result["text"])
    if args.out:
        write_json(args.out, result)
    return 0
