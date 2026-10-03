"""Fresh original-versus-copied-checkpoint evaluation, loading one model at a time."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import gc
import numpy as np
import torch
from .backend import Backend
from .metrics import Scorer, lexical_checks, paired_stats
from .util import LabError, write_json, write_csv

def evaluate_checkpoint(store,candidate_id,output):
    destination=Path(output).resolve()
    if destination.exists() and any(destination.iterdir()):raise LabError("Comparison output directory must be empty")
    destination.mkdir(parents=True,exist_ok=True)
    records=[r for r in store.dataset() if r["split"] in {"test","control"}]
    scorer=Scorer(store.config["behavior"])
    versions=[]
    for label,model_id in [("original",store.config["model"]["id"]),("candidate",candidate_id)]:
        c=deepcopy(store.config);c["model"]["id"]=model_id
        backend=Backend(c);rows=[]
        size=c["generation"]["batch_size"]
        for start in range(0,len(records),size):
            chunk=records[start:start+size]
            results=backend.generate(chunk)
            for r,value in zip(chunk,results):
                rows.append({"id":r["id"],"split":r["split"],"model":label,"messages":r["neutral"],
                             **value,"score":scorer(value["text"],value["tokens"],r),**lexical_checks(value["text"],r)})
        versions.append({"identity":backend.identity,"outputs":rows})
        write_json(destination/(label+"_outputs.json"),rows)
        del backend;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    summary=[]
    for split in ["test","control"]:
        original=[x for x in versions[0]["outputs"] if x["split"]==split]
        candidate=[x for x in versions[1]["outputs"] if x["split"]==split]
        stats=paired_stats([x["score"] for x in original],[x["score"] for x in candidate],
                           store.config["behavior"]["goal"],scale_floor=store.config["behavior"]["scale_floor"])
        summary.append({"split":split,**stats,
                        "original_cap_rate":float(np.mean([x["hit_limit"] for x in original])),
                        "candidate_cap_rate":float(np.mean([x["hit_limit"] for x in candidate])),
                        "mean_repeat_increase":float(np.mean([x["repeat4"]-y["repeat4"] for x,y in zip(candidate,original)]))})
    value={"original":versions[0]["identity"],"candidate":versions[1]["identity"],"summary":summary,
           "human_quality_review_required":True,"note":"Fresh held-out/control generations. Not a universal quality or safety certification. Token-count comparisons require equivalent tokenizers."}
    write_json(destination/"comparison.json",value);write_csv(destination/"comparison.csv",summary)
    return value
