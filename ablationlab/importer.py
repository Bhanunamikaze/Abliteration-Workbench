"""Explicit import of legacy .pt direction tensors, with provenance limitations preserved."""
from pathlib import Path
import torch
from .util import LabError, file_hash, write_json, digest
from .directions import DirectionBundle
from .pipeline import Pipeline

def import_directions(store,filename,raw_blocks=False):
    if store.completed("capture") or store.completed("directions"):
        raise LabError("Import into a fresh run, or reset capture explicitly")
    saved=torch.load(filename,map_location="cpu",weights_only=True)
    if not isinstance(saved,dict) or "directions" not in saved:raise LabError("Expected legacy dict with directions tensor")
    model_id=saved.get("model_id")
    if model_id!=store.config["model"]["id"]:raise LabError("Legacy model_id does not match run model")
    raw=saved["directions"].float()
    if raw.ndim!=2 or not torch.isfinite(raw).all() or (raw.norm(dim=-1)<1e-8).any():
        raise LabError("Invalid legacy direction tensor")
    pipe=Pipeline(store);pipe.run_stage("inspect")
    backend=pipe.engine.backend
    if raw.shape!=(len(backend.adapter.blocks),backend.adapter.hidden_size):raise LabError("Legacy directions have incompatible architecture")
    margins={int(l):abs(float(m)) for l,m in saved.get("layer_results",[])}
    if set(margins)!=set(range(raw.shape[0])):raise LabError("Legacy import requires a margin for every layer")
    metadata={"model_identity":digest(backend.identity),"model_id":model_id,"capture_site":"raw_block_output" if raw_blocks else "legacy_hidden_states",
              "coordinate":"residual","excluded_layers":[] if raw_blocks else [raw.shape[0]-1],"calibrated":False,"legacy_source_hash":file_hash(filename),
              "identity_note":"Bound to current backend by model ID and shape only. Original revision/dtype/tokenization was not recorded.",
              "note":"No class centroids or standardized held-out validation in old artifact. Automatic selection is blocked until recapture; manual exploratory stages are possible."}
    bundle=DirectionBundle((raw/raw.norm(dim=-1,keepdim=True))[:,None],torch.zeros_like(raw),torch.zeros_like(raw),
                            torch.tensor([margins[l] for l in range(raw.shape[0])]),metadata).validate()
    excluded=[] if raw_blocks else [raw.shape[0]-1]
    with store.stage("capture") as p:
        write_json(p/"result.json",{"imported":True,"raw_activations_available":False,"model_identity":digest(backend.identity),
                                   "source":str(Path(filename).resolve()),"excluded_last_layer":excluded})
    with store.stage("directions") as p:
        bundle.save(p)
        write_json(p/"result.json",{"imported":True,"calibrated":False,"eligible_layers":[l for l in range(raw.shape[0]) if l not in excluded],
                                    "metrics":[{"layer":l,"standardized_separation":0.0,"legacy_margin":margins[l]} for l in range(raw.shape[0])],
                                    "excluded_layers":excluded,"note":metadata["note"]})
    return metadata
