"""Read-only migration/audit of scripts 05-09. Original files remain intact."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import shutil
import numpy as np
from .util import LabError, read_rows, write_json, write_csv, atomic_text, file_hash, as_bool
from .metrics import projection_diagnostics, corr

NAMES = ["layer_sweep_results.csv","layer_sweep_outputs.json",
         "06_layer_analysis_summary.csv","06_layer_analysis_summary(1).csv",
         "07_writer_attribution_summary.csv","07_writer_attribution_results.csv","07_writer_attribution_outputs.json",
         "08_runtime_ablation_summary.csv","08_runtime_ablation_results.csv","08_runtime_ablation_outputs.json",
         "09_residual_ablation_summary.csv","09_residual_ablation_results.csv","09_residual_ablation_outputs.json",
         "09_regrowth_trace.csv","09_regrowth_summary.csv","09_regrowth_source_summary.csv"]

def audit_legacy(source: str | Path, destination: str | Path, copy_originals=True) -> dict:
    source,dest=Path(source).resolve(),Path(destination).resolve()
    if not source.is_dir():raise LabError(f"Legacy source folder not found: {source}")
    if dest.exists() and any(dest.iterdir()):raise LabError("Legacy destination must be empty; originals are never overwritten")
    dest.mkdir(parents=True,exist_ok=True)
    files={name:source/name for name in NAMES if (source/name).exists()}
    if not files:raise LabError("No recognized 05-09 artifact filenames in source folder")
    provenance={n:file_hash(p) for n,p in files.items()}
    if copy_originals:
        original=dest/"originals";original.mkdir()
        for n,p in files.items():shutil.copy2(p,original/n)
    warnings=["Imported values are historical observations, not fresh model runs.",
              "Legacy stages have different precision, batching, EOS counting and generation ceilings; do not combine baseline counts across stages.",
              "Old Code 03 final hidden state may have included final normalization; recapture for aligned raw-block directions.",
              "No actual direction tensor was reconstructed from scalar margins. Resume model experiments requires a direction file or new capture."]
    result={"files":provenance,"warnings":warnings}
    if "layer_sweep_results.csv" in files:
        rows=read_rows(files["layer_sweep_results.csv"])
        groups=defaultdict(list)
        for r in rows:groups[int(r["layer"])].append(r)
        summarized=[]
        for l,g in sorted(groups.items()):
            by=defaultdict(list)
            for r in g:by[float(r["strength"])].append(r)
            means={s:float(np.mean([float(r["tokens"]) for r in rr])) for s,rr in by.items()}
            pairs=[s for s in means if s>0 and -s in means]
            effects=[means[s]-means[-s] for s in pairs]
            wins=[]
            for s in pairs:
                pos={str(r["prompt_id"]):float(r["tokens"]) for r in by[s]}
                neg={str(r["prompt_id"]):float(r["tokens"]) for r in by[-s]}
                wins.extend(pos[i]>neg[i] for i in pos.keys()&neg.keys())
            effect=float(np.mean(effects)) if effects else 0
            cc=corr(sorted(means),[means[s] for s in sorted(means)])
            cap=float(np.mean([as_bool(r["hit_limit"]) for r in g]))
            consistency=float(np.mean(wins)) if wins else 0
            summarized.append({"layer":l,"legacy_margin":float(g[0]["margin"]),"avg_symmetric_effect_tokens":effect,
                               "avg_direction_consistency":consistency,"correlation_mean_curve":cc,"cap_rate":cap,
                               "baseline_mean":means.get(0),"means":means})
        maximum=max(max(x["avg_symmetric_effect_tokens"],0) for x in summarized) or 1
        for row in summarized:
            row["legacy_candidate_score"]=max(row["avg_symmetric_effect_tokens"],0)/maximum*row["avg_direction_consistency"]*max(row["correlation_mean_curve"],0)*(1-.35*row["cap_rate"])
        result["sweep_rows"]=len(rows);result["sweep_summary"]=summarized
        write_csv(dest/"recalculated_05_summary.csv",summarized)
    if "09_regrowth_trace.csv" in files:
        groups=defaultdict(list)
        for r in read_rows(files["09_regrowth_trace.csv"]):
            if int(r["observed_layer"])>=int(r["source_layer"]):
                groups[int(r["source_layer"]),float(r["ablation_strength"]),int(r["observed_layer"])].append(r)
        fixed=[]
        for (source_layer,strength,observed),g in sorted(groups.items()):
            baseline=[float(r["baseline_projection"]) for r in g]
            ablated=[float(r["ablated_projection"]) for r in g]
            legacy=np.mean([abs(a)/abs(b) for a,b in zip(ablated,baseline) if abs(b)>1e-6])
            fixed.append({"source_layer":source_layer,"strength":strength,"observed_layer":observed,
                          "legacy_mean_individual_abs_ratio":float(legacy),
                          **projection_diagnostics(baseline,ablated,.1)})
        write_csv(dest/"recalculated_09_projection_diagnostics.csv",fixed)
        result["projection_groups"]=len(fixed)
        result["source13_layer15_full"]=next((r for r in fixed if r["source_layer"]==13 and r["strength"]==1 and r["observed_layer"]==15),None)
    if "09_residual_ablation_results.csv" in files:
        groups=defaultdict(list)
        for r in read_rows(files["09_residual_ablation_results.csv"]):
            groups[int(r["source_layer"]),float(r["ablation_strength"])].append(r)
        recal=[]
        for (l,s),g in sorted(groups.items()):
            b=np.array([float(r["baseline_tokens"]) for r in g]);a=np.array([float(r["tokens"]) for r in g])
            recal.append({"layer":l,"strength":s,"baseline_mean":float(b.mean()),"ablated_mean":float(a.mean()),
                          "mean_reduction_tokens":float((b-a).mean()),
                          "reduction_fraction_of_means":float((b.mean()-a.mean())/b.mean()),
                          "mean_per_example_reduction_fraction":float(((b-a)/np.maximum(b,1)).mean()),
                          "cap_rate":float(np.mean([as_bool(r["hit_limit"]) for r in g]))})
        write_csv(dest/"recalculated_09_behavior.csv",recal);result["behavior_summary"]=recal
    # Prompts recovered without pretending they constitute new held-out data.
    for name in ["09_residual_ablation_outputs.json","08_runtime_ablation_outputs.json","layer_sweep_outputs.json"]:
        if name in files:
            found={}
            for row in read_rows(files[name]):
                pid=str(row["prompt_id"])
                if pid in found and found[pid]!=row["prompt"]:raise LabError("Legacy prompt id maps to conflicting prompt text")
                found[pid]=row["prompt"]
            write_json(dest/"recovered_prompts.json",[{"id":pid,"prompt":text,"status":"previously_used_in_selection_not_new_holdout"} for pid,text in found.items()])
            result["prompts_recovered"]=len(found);break
    write_json(dest/"audit.json",result)
    text=["# Legacy experiment audit","",*warnings,"","## Key findings",""]
    if result.get("sweep_summary"):
        top=max(result["sweep_summary"],key=lambda x:x["legacy_candidate_score"])
        text.append(f"Recomputed legacy winner: layer {top['layer']}, score {top['legacy_candidate_score']:.6f}. This reproduces the old heuristic, not an export recommendation.")
    spike=result.get("source13_layer15_full")
    if spike:
        text.append(f"L13 source, observed L15: old mean-of-ratios {spike['legacy_mean_individual_abs_ratio']:.6f}; ratio of mean magnitudes {spike['ratio_of_mean_abs']:.6f}. Neither proves semantic reconstruction.")
    text += ["","## Continuing research","",
             "Use a new run with explicit train/validation/test/control groups. The recovered prompts are already selection data; do not relabel them as untouched test data.",
             "Existing scalar results can be reviewed without the model. To continue live interventions, provide or recapture actual directions; this importer deliberately does not invent hidden activations."]
    atomic_text(dest/"audit.md","\n".join(text)+"\n")
    return result
