"""Explicit copied-checkpoint projection; deliberately separate from the automatic planner."""
from __future__ import annotations
from pathlib import Path
import shutil
import tempfile
import torch
from .util import LabError, Unsupported, read_json, write_json, digest, file_hash
from .directions import DirectionBundle
from .adapters import export_writers
from .interventions import Intervention, intervention_hooks

@torch.no_grad()
def project_linear(module: torch.nn.Linear, basis: torch.Tensor, strength: float) -> dict:
    if type(module) is not torch.nn.Linear:
        raise Unsupported("Only plain torch.nn.Linear residual writers can be exported, not quantized/fused/Conv1D/aggregate MoE modules")
    if module.weight.device.type=="meta":raise Unsupported("Offloaded/meta parameters cannot be edited by this exporter")
    if not 0<=strength<=1:raise LabError("Export strength must be in [0,1]")
    q=basis.to(module.weight.device,dtype=torch.float32)
    if module.out_features!=q.shape[-1]:raise LabError("Writer output and direction dimensions disagree")
    w=module.weight.float()
    new=w-float(strength)*q.T@(q@w)
    stats={"weight_relative_change":float((new-w).norm()/w.norm().clamp_min(1e-12)),"shape":list(w.shape)}
    module.weight.copy_(new.to(module.weight.dtype))
    if module.bias is not None:
        b=module.bias.float()
        module.bias.copy_((b-float(strength)*q.T@(q@b)).to(module.bias.dtype))
    return stats

def make_export_plan(store) -> dict:
    bundle=DirectionBundle.load(store.stage_dir("directions"))
    plan={"schema_version":1,"model_identity":bundle.metadata.get("model_identity"),
          "directions_fingerprint":bundle.fingerprint(),"edits":[],"logit_tolerance":0.05,
          "note":"No automatic export. Each edit needs layer, site=attention|mlp, strength in [0,1]. Plain dense writers or registered unfused scalar-mixture expert output linears only. Runtime scope must be all tokens for equivalence."}
    if store.completed("evaluate"):
        ev=store.result("evaluate");spec=ev.get("selected_intervention",{})
        if ev.get("passed") and spec.get("operation")=="ablate" and spec.get("reference")=="zero" and spec.get("scope")=="all" and spec.get("phase")=="both" and spec.get("site") in {"attention","mlp"}:
            plan["edits"]=[{"layer":l,"site":spec["site"],"strength":spec["strength"]} for l in spec["layers"]]
            plan["note"]+=" Populated from held-out evaluation; human quality review remains necessary."
    return plan

def export_checkpoint(store,plan_path,destination):
    from .backend import Backend
    from contextlib import ExitStack
    plan=read_json(plan_path)
    if plan.get("schema_version")!=1 or not plan.get("edits"):raise LabError("Export plan has no edits. A rank winner alone does not create an edit plan.")
    import math
    if not isinstance(plan["edits"], list): raise LabError("Export edits must be a list")
    tolerance = plan.get("logit_tolerance", .05)
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise LabError("logit_tolerance must be a finite nonnegative number")
    for item in plan["edits"]:
        if not isinstance(item, dict) or not {"layer", "site", "strength"} <= item.keys():
            raise LabError("Each export edit needs layer, site and strength")
        if type(item["layer"]) is not int or item["layer"] < 0:
            raise LabError("Export layer must be a nonnegative integer")
        strength = item["strength"]
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise LabError("Export strength must be a finite number in [0,1]")
    if store.config["model"].get("backend_plugin"):raise Unsupported("Custom backends must supply their own export implementation")
    b=Backend(store.config);bundle=DirectionBundle.load(store.stage_dir("directions"))
    b._validate_bundle(bundle)
    if plan.get("model_identity")!=digest(b.identity) or plan.get("directions_fingerprint")!=bundle.fingerprint():
        raise LabError("Export plan does not match model/directions provenance")
    if store.config["model"]["quantization"]!="none" or store.config["model"]["allow_offload"]:
        raise Unsupported("Reload a floating, nonoffloaded model for export; quantized/offloaded exports are not implemented")
    dest=Path(destination).resolve();model_path=Path(store.config["model"]["id"]).resolve()
    if dest.exists():raise LabError("Export destination must not already exist")
    if model_path.exists() and (dest==model_path or model_path in dest.parents or dest in model_path.parents):
        raise LabError("Export must be separate from the source checkpoint directory")
    if dest==store.root or dest in store.root.parents:raise LabError("Export cannot replace the run directory")
    edits=[];seen=set()
    aliases = {}
    for name, parameter in b.model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for item in plan["edits"]:
        l=int(item["layer"]);site=str(item["site"]);s=float(item["strength"])
        if site not in {"attention","mlp"}:raise Unsupported("Export accepts only named residual-writing linear modules")
        mod=b.adapter.get(l,site)
        expanded=export_writers(mod)
        for path,module in expanded:
            if id(module.weight) in seen:raise LabError("Duplicate/shared selected weights need a dedicated tied-parameter adapter")
            for parameter in (module.weight, module.bias):
                if parameter is not None and len(aliases.get(id(parameter), [])) > 1:
                    raise Unsupported("Selected parameter has aliases; export needs a dedicated tied-parameter adapter: " + ", ".join(aliases[id(parameter)]))
            seen.add(id(module.weight))
        edits.append((l,site,s,mod,expanded))
    # Backup only selected parameters, not another full model; restore in finally.
    backups=[(module,module.weight.detach().cpu().clone(),None if module.bias is None else module.bias.detach().cpu().clone()) for _,_,_,_,expanded in edits for path,module in expanded]
    record=next(r for r in store.dataset() if r["split"]=="control")
    inputs=b.build_inputs([record["neutral"]])
    specs=[Intervention((l,),site,"ablate",s,"zero","all","both") for l,site,s,mod,expanded in edits]
    tmp=None
    try:
        with torch.inference_mode(), ExitStack() as stack:
            for spec in specs:stack.enter_context(intervention_hooks(b.adapter,bundle,spec,inputs["attention_mask"]))
            reference=b.model(**inputs,use_cache=False).logits.detach().float().cpu()
        changes=[]
        for l,site,s,mod,expanded in edits:
            for path,module in expanded:
                changes.append({"layer":l,"site":site,"path":path,"strength":s,**project_linear(module,bundle.basis[l],s)})
        with torch.inference_mode():edited=b.model(**inputs,use_cache=False).logits.detach().float().cpu()
        difference=float((reference-edited).abs().max())
        if difference>float(plan.get("logit_tolerance",.05)):
            raise LabError(f"Runtime/weight logit equivalence check failed: max delta {difference}; nothing exported")
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=Path(tempfile.mkdtemp(prefix=".ablab-export-",dir=dest.parent))
        b.model.save_pretrained(tmp,safe_serialization=True,max_shard_size="2GB")
        b.tokenizer.save_pretrained(tmp)
        write_json(tmp/"ablationlab_edit_manifest.json",{"model_identity":b.identity,"plan":plan,"changes":changes,
                   "runtime_vs_weight_max_logit_delta":difference,"validation_scope":"one fixed control prompt forward; not whole-model quality certification",
                   "note":"Export changes every token at these writer modules. This is not equivalent to last-token-only runtime ablation."})
        tmp.rename(dest);tmp=None
    finally:
        with torch.no_grad():
            for module,w,bias in backups:
                module.weight.copy_(w.to(module.weight.device))
                if bias is not None:module.bias.copy_(bias.to(module.bias.device))
        if tmp is not None:shutil.rmtree(tmp,ignore_errors=True)
    store.event("copied_checkpoint_exported",destination=str(dest),plan_hash=file_hash(plan_path))
    return {"destination":str(dest),"runtime_vs_weight_max_logit_delta":difference,"changes":changes,
            "original_model_restored":all(torch.equal(m.weight.detach().cpu(),w) and (bias is None or torch.equal(m.bias.detach().cpu(),bias)) for m,w,bias in backups)}
