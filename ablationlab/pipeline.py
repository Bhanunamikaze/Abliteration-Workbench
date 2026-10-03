"""Executable stages for local representation-engineering research."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from .util import LabError, BudgetReached, digest, read_json, write_json, write_csv, file_hash, save_tensors
from .store import STAGES, DEPS
from .data import split_rows
from .directions import DirectionBundle, estimate, validation_metrics
from .interventions import Intervention
from .metrics import paired_stats, corr, projection_diagnostics
from .engine import Engine
from .planner import choose_next, apply_override
from .evidence import classify_stats, candidate_priority, representation_removal_signal, build_persistent_regions

class Pipeline:
    def __init__(self,store):
        self.store=store; self.c=store.config; self.data=store.dataset(); self.engine=Engine(store)
        self._bundle=None
    def bundle(self):
        if self._bundle is None:self._bundle=DirectionBundle.load(self.store.stage_dir("directions"))
        return self._bundle
    def finish(self,path,result):
        write_json(path/"result.json",result)
        return result
    def run_stage(self,name,with_deps=False,reset=False):
        if reset:
            self.store.reset(name)
            self._bundle = None
        if self.store.completed(name):return self.store.result(name)
        if with_deps:
            for dep in DEPS[name]:self.run_stage(dep,True)
        print(f"\n=== {name.upper()} ===",flush=True)
        with self.store.stage(name) as p:
            return getattr(self,"stage_"+name)(p)
    def auto(self,start=None,until=None,max_steps=20):
        decisions=[]
        if start:self.run_stage(start)
        for _ in range(max_steps):
            results={s:self.store.result(s) for s in STAGES if self.store.completed(s)}
            order=sorted(results,key=lambda s:self.store.state()["stages"][s].get("ended_at",""))
            decision=apply_override(choose_next(results, mode=self.c["search"].get("mode","strict")),order,self.c["search"]["overrides"])
            self.store.event("planner_decision",**decision.serial());decisions.append(decision.serial())
            print(f"NEXT: {decision.next_stage} | {decision.reason}",flush=True)
            if decision.next_stage is None:break
            self.run_stage(decision.next_stage)
            if decision.next_stage==until:break
        else:raise LabError("Planner step budget exhausted; check manual route overrides")
        return decisions
    def _eval_rows(self,split="validation"):
        return split_rows(self.data,split)
    def _base(self,records,limit=None):
        return self.engine.generate(records,max_new_tokens=limit)
    def _spec(self,layers,site="residual",operation="ablate",strength=1.0,reference=None,control_seed=None):
        g=self.c["generation"]
        selected=reference or (self.c["search"]["reference"] if site=="residual" else "zero")
        if selected=="auto":selected="zero"
        if site!="residual":selected="zero"
        return Intervention(tuple(layers),site,operation,float(strength),
            selected,g["scope"],g["phase"],control_seed)
    def _aggregate(self,records,base,changed):
        b=self.c["behavior"];s=self.c["search"]
        if [x["id"] for x in base] != [x["id"] for x in changed]:
            raise LabError("Baseline/intervention prompt IDs are not aligned")
        r=paired_stats([x["score"] for x in base],[x["score"] for x in changed],b["goal"],
                       self.c["generation"]["seed"],b["scale_floor"])
        r["baseline_cap_rate"]=float(np.mean([x["hit_limit"] for x in base]))
        r["cap_rate"]=float(np.mean([x["hit_limit"] for x in changed]))
        r["uncensored_pair_fraction"]=float(np.mean([not (x["hit_limit"] or y["hit_limit"]) for x,y in zip(base,changed)]))
        r["repeat_increase"]=float(np.mean([x["repeat4"]-y["repeat4"] for x,y in zip(changed,base)]))
        r["empty_fraction"]=float(np.mean([x["empty"] for x in changed]))
        termdiff=[x["required_term_fraction"]-y["required_term_fraction"] for x,y in zip(changed,base)
                  if x["required_term_fraction"] is not None and y["required_term_fraction"] is not None]
        r["required_term_change"]=float(np.mean(termdiff)) if termdiff else None
        r.update(classify_stats(r,b["metric"],s))
        return r
    def _records(self,path,rows):
        write_json(path/"outputs.json",rows)
        write_csv(path/"results.csv",[{k:v for k,v in row.items() if k not in {"text","token_ids"}} for row in rows])
    def _run_variant(self,records,spec,base,limit=None):
        changed=self.engine.generate(records,self.bundle(),spec,limit)
        cap=limit or self.c["generation"]["max_new_tokens"]
        tagged=[{**x,"intervention":spec.serial(),"baseline_score":b["score"],"baseline_tokens":b["tokens"],
                 "generation_cap":cap} for x,b in zip(changed,base)]
        return tagged,{"intervention":spec.serial(),"generation_cap":cap,
                       "layer_count":len(spec.layers),"region_width":max(spec.layers)-min(spec.layers)+1 if spec.layers else 0,
                       **self._aggregate(records,base,changed)}
    def _candidates(self):
        for stage in ["refine","sweep"]:
            if self.store.completed(stage):
                result=self.store.result(stage)
                return result.get("candidates",[]) or result.get("exploratory_candidates",[])
        raise LabError("Run sweep before selecting candidates")

    def stage_inspect(self,p):
        b=self.engine.backend
        write_json(p/"capabilities.json",b.capabilities)
        write_csv(p/"parameters.csv",[{"name":n,"shape":list(v.shape),"dtype":str(v.dtype),"numel":v.numel()} for n,v in b.model.named_parameters()])
        return self.finish(p,{"capabilities":b.capabilities,"model_identity":digest(b.identity)})

    def stage_baseline(self,p):
        validation=self._eval_rows()
        control=self._eval_rows("control")
        rows=self._base(validation)+self._base(control)
        # An explicit Stage 0 option retains fixed test baselines without using
        # their scores in validation summaries or selecting an intervention.
        include_test=self.c["evaluation"].get("baseline_include_test",False)
        summary_rows=list(rows)
        if include_test:rows.extend(self._base(self._eval_rows("test")))
        contrast=[]
        for condition in ["positive","negative"]:
            records=[dict(r,neutral=r[condition],id=r["id"]+":"+condition) for r in validation]
            contrast.extend([{**x,"condition":condition} for x in self._base(records)])
        # Zero hook run is mathematically exact and should be operationally exact.
        repeated=self.engine.generate(validation,spec=Intervention((),strength=0))
        ref=rows[:len(validation)]
        self._records(p,rows)
        write_json(p/"contrast_outputs.json",contrast)
        pos=[x["score"] for x in contrast if x["condition"]=="positive"]
        neg=[x["score"] for x in contrast if x["condition"]=="negative"]
        return self.finish(p,{"mean_score":float(np.mean([x["score"] for x in summary_rows])),
                             "cap_rate":float(np.mean([x["hit_limit"] for x in summary_rows])),
                             "test_baseline_included":include_test,
                             "positive_minus_negative_score":float(np.mean(pos)-np.mean(neg)),
                             "note":"Instruction labels do not prove observed behavior; inspect contrast_outputs.json",
                             "no_op_exact":all(x["token_ids"]==y["token_ids"] for x,y in zip(repeated,ref))})

    def stage_capture(self,p):
        tensors={};ids={};batch_size=self.c["generation"]["batch_size"]
        for split in ["train","validation"]:
            records=self._eval_rows(split);ids[split]=[r["id"] for r in records]
            for condition in ["positive","negative"]:
                parts=[]
                for start in range(0,len(records),batch_size):
                    self.store.check_budget()
                    key=f"{split}_{condition}_{start}"
                    checkpoint=p/(key+".safetensors")
                    if checkpoint.exists():part=load_file(str(checkpoint))["states"]
                    else:
                        part=self.engine.backend.capture([r[condition] for r in records[start:start+batch_size]])
                        save_tensors(checkpoint,{"states":part.contiguous()})
                    parts.append(part)
                tensors[split+"_"+condition]=torch.cat(parts)
        save_tensors(p/"activations.safetensors",{k:v.contiguous() for k,v in tensors.items()})
        # Batched checkpoints retained for interruption recovery and audit.
        return self.finish(p,{"ids":ids,"capture_site":"raw_block_output","position":"last_prompt_token",
                             "shapes":{k:list(v.shape) for k,v in tensors.items()},
                             "model_identity":digest(self.engine.backend.identity)})

    def stage_directions(self,p):
        raw=load_file(str(self.store.stage_dir("capture")/"activations.safetensors"))
        d=self.c["discovery"]
        bundle=estimate(raw["train_positive"],raw["train_negative"],d["rank"],d["method"],
            {"model_identity":self.store.result("capture")["model_identity"],"behavior":self.c["behavior"],
             "dataset_hash":self.store.meta["dataset_hash"],"token_scope":"last_prompt_token"})
        bundle.save(p);self._bundle=bundle
        rows=validation_metrics(bundle,raw["validation_positive"],raw["validation_negative"])
        eligible=[x["layer"] for x in rows if x["standardized_separation"]>=self.c["search"]["min_separation"] and x["paired_positive_fraction"]>=.6]
        write_csv(p/"validation.csv",rows)
        return self.finish(p,{"eligible_layers":eligible,"metrics":rows,"calibrated":True,
                             "note":"Held-out instruction-contrast separation; not automatic causal proof"})

    def _layer_grid(self):
        dr=self.store.result("directions");s=self.c["search"]
        total=self.bundle().basis.shape[0]
        if s["layers"] is not None:
            layers=list(dict.fromkeys(s["layers"]))
            if max(layers)>=total:raise LabError("Explicit search layer outside model")
            return layers
        eligible=dr["eligible_layers"]
        count=s["max_layer_tests"]
        if not count or len(eligible)<=count:return eligible
        ranked=sorted(dr["metrics"],key=lambda x:x["standardized_separation"],reverse=True)
        # Coverage, not just maximum-margin layers, because detectability != controllability.
        coverage=[eligible[i] for i in np.linspace(0,len(eligible)-1,max(1,count//2),dtype=int)]
        extra=[x["layer"] for x in ranked if x["layer"] in eligible and x["layer"] not in coverage]
        return sorted((coverage+extra)[:count])

    def _steering_sweep(self,p,layers,strengths,limit):
        records=self._eval_rows();base=self._base(records,limit)
        all_rows=[];summaries=[];s=self.c["search"]
        for layer in layers:
            print(f"Testing layer {layer}",flush=True)
            runs={}
            for strength in strengths:
                spec=self._spec([layer],operation="steer",strength=strength,reference="zero")
                rows,stats=self._run_variant(records,spec,base,limit)
                runs[strength]=rows;all_rows.extend([{**x,"layer":layer,"strength":strength} for x in rows])
            positives=sorted(x for x in strengths if x>0)
            differences=np.array([[a["score"]-b["score"] for a,b in zip(runs[x],runs[-x])] for x in positives])
            per_prompt=differences.mean(0)
            orientation=1 if per_prompt.mean()>=0 else -1
            effect=float(abs(per_prompt.mean()));baseline_scale=max(float(np.mean([abs(x["score"]) for x in base])),self.c["behavior"]["scale_floor"])
            cap=float(np.mean([r["hit_limit"] for rows in runs.values() for r in rows]))
            wins=float((orientation*per_prompt>0).mean())
            mean_curve=[np.mean([x["score"] for x in runs[v]]) for v in sorted(strengths)]
            coherence=float(np.mean([x["repeat4"] for rows in runs.values() for x in rows])-np.mean([x["repeat4"] for x in base]))
            # Matched-magnitude random directions are not assumed inert.
            random_effects=[]
            for seed in range(s["random_controls"]):
                random_runs=[]
                for strength in [-min(positives),min(positives)]:
                    spec=self._spec([layer],operation="steer",strength=strength,reference="zero",control_seed=1000+seed)
                    rr,_=self._run_variant(records,spec,base,limit)
                    random_runs.append(rr)
                    all_rows.extend([{**x,"layer":layer,"strength":strength,"random_control":True} for x in rr])
                random_effects.append(abs(float(np.mean([a["score"]-b["score"] for a,b in zip(random_runs[1],random_runs[0])]))))
            target_small=abs(float(differences[0].mean()))
            random_max=max(random_effects,default=0)
            ci=paired_stats(np.zeros(len(per_prompt)),orientation*per_prompt,"increase",
                            self.c["generation"]["seed"],self.c["behavior"]["scale_floor"])
            baseline_cap=float(np.mean([r["hit_limit"] for r in base]))
            effect_pass=effect/baseline_scale>=s["min_effect_fraction"]
            win_pass=wins>=s["min_win_rate"]
            confidence_pass=ci["ci95_low"]>0
            repeat_pass=coherence<=s["max_repeat_increase"]
            specificity_pass=bool(random_effects) and target_small>random_max
            content_pass=not any(r["empty"] for rows in runs.values() for r in rows)
            censoring_pass=max(cap,baseline_cap)<=s["max_cap_rate"]
            signal=effect_pass and win_pass and confidence_pass and repeat_pass and specificity_pass and content_pass
            status=("promising" if signal and censoring_pass else
                    "promising_censored" if signal else "exploratory" if effect>0 else "rejected")
            summaries.append({"layer":layer,"effect":effect,"signed_effect":float(per_prompt.mean()),"effect_fraction":effect/baseline_scale,
                              "orientation":orientation,"win_rate":wins,"strength_score_correlation":corr(sorted(strengths),mean_curve),
                              "cap_rate":cap,"baseline_cap_rate":baseline_cap,"repeat_increase":coherence,
                              "random_control_max_effect":random_max,
                              "random_control_seeds":[1000+seed for seed in range(s["random_controls"])],
                              "random_control_status":"measured" if random_effects else "not_run",
                              "effect_pass":effect_pass,"win_rate_pass":win_pass,"confidence_pass":confidence_pass,
                              "specificity_pass":specificity_pass,"repeat_pass":repeat_pass,
                              "content_check_pass":content_pass,"censoring_pass":censoring_pass,
                              "behaviorally_promising":bool(signal and (censoring_pass or self.c["behavior"]["metric"] not in {"tokens","words"})),
                              "quality_confirmed":bool(censoring_pass and repeat_pass and content_pass),
                              "evidence_status":status,"ci95_low":ci["ci95_low"],"ci95_high":ci["ci95_high"],
                              "eligible":signal,"ranking_score":effect/baseline_scale*wins,
                              "max_new_tokens":limit})
        ranked=sorted(summaries,key=lambda x:x["ranking_score"],reverse=True)
        candidates=[x for x in ranked if x["eligible"]][:s["top_k"]]
        self._records(p,all_rows);write_csv(p/"summary.csv",ranked)
        return self.finish(p,{"candidates":candidates,"exploratory_candidates":ranked[:s["top_k"]],
                             "cap_heavy":any(not x["censoring_pass"] for x in candidates),
                             "summary":ranked,"strengths":strengths,"max_new_tokens":limit,
                             "note":"Candidate thresholds are configurable heuristics; test split not used here"})
    def stage_sweep(self,p):
        return self._steering_sweep(p,self._layer_grid(),self.c["search"]["strengths"],self.c["generation"]["max_new_tokens"])
    def stage_refine(self,p):
        previous=self.store.result("sweep")
        layers=[x["layer"] for x in previous["exploratory_candidates"]]
        if not layers:return self.finish(p,{"candidates":[],"summary":[],"cap_heavy":False})
        strengths=sorted(set(float(x)*self.c["search"]["refine_strength_factor"] for x in previous["strengths"]))
        limit=max(self.c["generation"]["max_new_tokens"],self.c["generation"]["refine_max_new_tokens"])
        return self._steering_sweep(p,layers,strengths,limit)

    def stage_writers(self,p):
        records=self._eval_rows();base=self._base(records);b=self.engine.backend
        all_rows=[];summary=[];selected=[]
        strength=min(abs(x) for x in self.c["search"]["strengths"])*.6
        for candidate in self._candidates():
            l=candidate["layer"];entries=[]
            for site in ["residual","attention","mlp","both"]:
                if site=="both":
                    if not all((l,n) in b.adapter.validated for n in ["attention","mlp"]):continue
                elif (l,site) not in b.adapter.validated:continue
                runs=[]
                for v in [-strength,strength]:
                    spec=self._spec([l],site,"steer",v,"zero")
                    rows,_=self._run_variant(records,spec,base);runs.append(rows);all_rows.extend(rows)
                diff=np.array([a["score"]-n["score"] for a,n in zip(runs[1],runs[0])])
                sign=candidate["orientation"]
                entry={"layer":l,"site":site,"effect":float(sign*diff.mean()),"win_rate":float((sign*diff>0).mean()),
                       "cap_rate":float(np.mean([r["hit_limit"] for x in runs for r in x])),
                       "path":b.adapter.get(l,site).path if site!="both" else "split attention + MLP output injection",
                       "kind":b.adapter.get(l,site).kind if site!="both" else "composite",
                       "note":"Intervention-site sensitivity, NOT natural component responsibility"}
                summary.append(entry)
                if site in {"attention","mlp"}:entries.append(entry)
            maximum=max([x["effect"] for x in entries],default=0)
            for x in entries:
                if x["effect"]>0 and x["effect"]>=self.c["search"]["writer_fraction"]*maximum and x["win_rate"]>=self.c["search"]["min_win_rate"]:
                    selected.append(x)
        self._records(p,all_rows);write_csv(p/"summary.csv",summary)
        return self.finish(p,{"selected_sites":selected,"summary":summary,
                             "note":"For sequential pre-norm blocks, MLP-output addition and block-output addition can be algebraically equivalent; rounding is not natural attribution"})

    def _residual_references(self,site):
        if site!="residual":return ["zero"]
        strategy=self.c["search"]["reference"]
        if strategy!="auto":return [strategy]
        bundle=self.bundle()
        return ["zero","negative"] if bundle.metadata.get("calibrated",True) and bundle.negative is not None else ["zero"]

    def _ablation_variants(self,p,sites,source_stage):
        records=self._eval_rows();base=self._base(records);all_rows=[];summaries=[]
        s=self.c["search"];g=self.c["generation"]
        confirmation_cap=g.get("confirm_max_new_tokens",g["refine_max_new_tokens"])
        max_confirmations=s.get("max_cap_confirmations",1)
        for layers,site,label in sites:
            for reference in self._residual_references(site):
                group=[]
                for strength in s["ablations"]:
                    print(f"{label} reference={reference} ablation={strength}",flush=True)
                    spec=self._spec(layers,site,strength=strength,reference=reference)
                    rows,stats=self._run_variant(records,spec,base)
                    all_rows.extend([{**row,"generation_phase":"initial"} for row in rows])
                    group.append(dict(stats,label=label,candidate_source_stage=source_stage,generation_phase="initial",
                                      candidate_source=label,confirmation={"attempted":False,"reason":"not_required"}))
                # Same-rank random projection controls are matched to site,
                # region, reference, strength and generation ceiling.
                controls=[]
                for seed in range(s["random_controls"]):
                    spec=self._spec(layers,site,strength=max(s["ablations"]),reference=reference,control_seed=2000+seed)
                    rows,stats=self._run_variant(records,spec,base)
                    all_rows.extend([{**row,"generation_phase":"initial"} for row in rows])
                    controls.append(stats["gain_fraction_of_baseline_mean"])
                for item in group:
                    item["random_control_max_gain_fraction"]=max(controls) if controls else None
                    item["random_control_seeds"]=[2000+seed for seed in range(s["random_controls"])]
                    item["random_control_status"]="measured" if controls else "not_run"
                    item.update(classify_stats(item,self.c["behavior"]["metric"],s))
                censored=[item for item in group if item["evidence_status"]=="promising_censored"]
                if censored and max_confirmations and confirmation_cap>g["max_new_tokens"]:
                    high_base=self._base(records,confirmation_cap)
                    high_controls=[]
                    for seed in range(s["random_controls"]):
                        control_spec=self._spec(layers,site,strength=max(s["ablations"]),reference=reference,control_seed=2000+seed)
                        rows,stats=self._run_variant(records,control_spec,high_base,confirmation_cap)
                        all_rows.extend([{**row,"generation_phase":"confirmation"} for row in rows])
                        high_controls.append(stats["gain_fraction_of_baseline_mean"])
                    selected_confirmations=sorted(censored,key=candidate_priority,reverse=True)[:max_confirmations]
                    for item in censored:
                        if item not in selected_confirmations:
                            item["confirmation"]={"attempted":False,"reason":"bounded_candidate_limit_reached"}
                    for item in selected_confirmations:
                        initial={k:v for k,v in item.items() if k!="confirmation"}
                        spec=Intervention(**{**item["intervention"],"layers":tuple(item["intervention"]["layers"])})
                        rows,stats=self._run_variant(records,spec,high_base,confirmation_cap)
                        all_rows.extend([{**row,"generation_phase":"confirmation"} for row in rows])
                        stats["random_control_max_gain_fraction"]=max(high_controls) if high_controls else None
                        stats["random_control_seeds"]=[2000+seed for seed in range(s["random_controls"])]
                        stats["random_control_status"]="measured" if high_controls else "not_run"
                        stats["generation_phase"]="confirmation"
                        stats.update(classify_stats(stats,self.c["behavior"]["metric"],s))
                        item.update(stats)
                        item["confirmation"]={"attempted":True,"initial_max_new_tokens":g["max_new_tokens"],
                                              "final_max_new_tokens":confirmation_cap,
                                              "initial":initial,"initial_status":initial["evidence_status"],
                                              "final_status":item["evidence_status"],"bounded_attempts":1}
                elif censored:
                    for item in censored:
                        item["confirmation"]={"attempted":False,"reason":"No larger configured confirmation ceiling or confirmation disabled"}
                summaries.extend(group)
        self._records(p,all_rows);write_csv(p/"summary.csv",summaries)
        promising=[x for x in summaries if x["evidence_status"] in {"promising","promising_censored"}]
        return {"summary":summaries,"promising":promising,
                "confirmed_promising":[x for x in promising if x["evidence_status"]=="promising"],
                "all_tested":len(summaries),
                "reference_comparison_performed":any(x["intervention"]["reference"]=="negative" for x in summaries)
                                                 and any(x["intervention"]["reference"]=="zero" for x in summaries),
                "note":"Zero and measured negative-centroid residual targets are distinct; censoring remains a quality limitation"}
    def stage_ablate(self,p):
        selected=self.store.result("writers")["selected_sites"]
        result=self._ablation_variants(p,[([x["layer"]],x["site"],f"L{x['layer']}/{x['site']}") for x in selected],"ablate")
        return self.finish(p,result)

    def stage_trace(self,p):
        layers=[x["layer"] for x in self._candidates()]
        result=self._ablation_variants(p,[([l],"residual",f"L{l}/residual") for l in layers],"trace")
        records=self._eval_rows();base=self._base(records);b=self.engine.backend;bundle=self.bundle()
        traces=[];summary=[]
        max_strength=max(self.c["search"]["ablations"])
        for prefix in self.c["trace"]["prefix_tokens"]:
            baseline={}
            for r,gen in zip(records,base):
                if prefix>len(gen["token_ids"]):continue
                baseline[r["id"]]=b.trace(r,bundle,prefix_ids=gen["token_ids"][:prefix])
            for source in layers:
                spec=self._spec([source],strength=max_strength)
                for r,gen in zip(records,base):
                    if r["id"] not in baseline:continue
                    self.store.check_budget()
                    changed=b.trace(r,bundle,spec,gen["token_ids"][:prefix])
                    normal=baseline[r["id"]]
                    for observed in range(source,len(b.adapter.blocks)):
                        traces.append({"source_layer":source,"prefix_tokens":prefix,"id":r["id"],"observed_layer":observed,
                                       "baseline_projection":float(normal[observed,0]),"changed_projection":float(changed[observed,0]),
                                       "basis_overlap_with_source":float(torch.dot(bundle.basis[source,0],bundle.basis[observed,0]))})
        for source in layers:
            for prefix in self.c["trace"]["prefix_tokens"]:
                for observed in range(source,len(b.adapter.blocks)):
                    group=[x for x in traces if x["source_layer"]==source and x["prefix_tokens"]==prefix and x["observed_layer"]==observed]
                    if not group:continue
                    stats=projection_diagnostics([x["baseline_projection"] for x in group],[x["changed_projection"] for x in group],self.c["trace"]["denominator_floor"])
                    summary.append({"source_layer":source,"prefix_tokens":prefix,"observed_layer":observed,**stats})
        write_csv(p/"projection_trace.csv",traces);write_csv(p/"projection_summary.csv",summary)
        signal=representation_removal_signal(summary,result["summary"],self.c["search"])
        result.update(projection_summary=summary,representation_removal_behavior_weak=signal,
                      trace_caveat="Matched fixed-prefix, single-forward diagnostic. Later layer probes use different directions; coordinate return is not proof of information reconstruction. Not a recording of divergent autoregressive generations.")
        return self.finish(p,result)

    def stage_persistent(self,p):
        peaks=sorted(x["layer"] for x in self._candidates())
        custom=self.c["search"].get("custom_regions",[])
        if not peaks and not custom:return self.finish(p,{"summary":[],"promising":[],"note":"No valid source layers"})
        count=self.bundle().basis.shape[0]
        regions,omitted=build_persistent_regions(peaks,count,self.c["search"])
        seen=set();sites=[]
        for name,layers in regions.items():
            if tuple(layers) not in seen:sites.append((layers,"residual",name));seen.add(tuple(layers))
        for entry in custom:
            layers=entry["layers"]
            if max(layers)>=count:raise LabError("Custom persistent region layer outside model")
            if tuple(layers) not in seen:sites.append((layers,"residual",entry["label"]));seen.add(tuple(layers))
        result=self._ablation_variants(p,sites,"persistent")
        result["regions"]={label:layers for layers,site,label in sites}
        result["omitted_regions"]=omitted
        result["note"]="Each layer uses its own calibrated basis. More layers means a larger intervention. Wide automatic spans are omitted; custom regions remain explicit."
        return self.finish(p,result)

    def stage_evaluate(self,p):
        # Candidate selection used validation only. Test and control data are now independent evaluations.
        options=[];considered=[]
        for name in ["ablate","trace","persistent"]:
            if not self.store.completed(name):continue
            stage_result=self.store.result(name)
            # Old runs placed censored effects in summary but omitted them from
            # promising. Reclassify read-only rather than changing old artifacts.
            for original in stage_result.get("summary",stage_result.get("promising",[])):
                if "intervention" not in original or original["intervention"].get("control_seed") is not None:continue
                row=dict(original,candidate_source_stage=name)
                if "evidence_status" not in row:
                    row.update(classify_stats(row,self.c["behavior"]["metric"],self.c["search"]))
                considered.append(row)
                if row["evidence_status"]=="promising" or (
                    row["evidence_status"]=="promising_censored" and row.get("behaviorally_promising",False)) or (
                    row["evidence_status"]=="exploratory" and row.get("mean_gain",0)>0 and row.get("specificity_pass",False)):
                    options.append(row)
        steering=self._candidates()
        steering_effect=max((x.get("effect_fraction",0) for x in steering),default=0)
        eligible=[x for x in options if x["evidence_status"]=="promising" or
                  (x["evidence_status"]=="promising_censored" and x.get("behaviorally_promising",False))
                  or x.get("gain_fraction_of_baseline_mean",0)>steering_effect]
        if eligible:
            best=max(eligible,key=candidate_priority)
            spec=Intervention(**{**best["intervention"],"layers":tuple(best["intervention"]["layers"])})
            selected_source=best["candidate_source_stage"]
            selected_status=best["evidence_status"]
            selection_reason=(f"Selected {selected_source} {selected_status} candidate with validation gain "
                              f"{best['gain_fraction_of_baseline_mean']:.4g}, win rate {best.get('win_rate',0):.4g}, "
                              f"cap rate {best.get('cap_rate',0):.4g}, and {best.get('layer_count',len(spec.layers))} layers; "
                              "held-out quality and controls remain required")
            limit=best.get("generation_cap",self.c["generation"]["max_new_tokens"])
            validation_specificity_pass=best.get("specificity_pass",False)
        else:
            if not steering:return self.finish(p,{"passed":False,"reason":"No candidate exists",
                                                  "selection_reason":"No causal or steering candidate was available",
                                                  "candidate_source_stage":None,"candidate_evidence_status":None,
                                                  "evaluation_status":"not_evaluated"})
            candidate=steering[0]
            goal_sign=1 if self.c["behavior"]["goal"]=="increase" else -1
            spec=self._spec([candidate["layer"]],operation="steer",strength=min(abs(v) for v in self.c["search"]["strengths"])*candidate["orientation"]*goal_sign,reference="zero")
            selected_source="refine" if self.store.completed("refine") else "sweep"
            selected_status=candidate.get("evidence_status","exploratory" if not candidate.get("eligible",False) else "promising")
            selection_reason="No stronger causal candidate met evidence criteria; selected top steering candidate for held-out measurement"
            deferred=[{"stage":row["candidate_source_stage"],"layers":row["intervention"]["layers"],
                       "reference":row["intervention"].get("reference"),"evidence_status":row["evidence_status"],
                       "reason":("directly capped length metric" if row["evidence_status"]=="promising_censored"
                                 and not row.get("behaviorally_promising",False) else
                                 "higher-cap confirmation did not retain a promising effect" if
                                 row.get("confirmation",{}).get("initial_status")=="promising_censored" else
                                 "effect or specificity did not exceed the steering alternative")}
                      for row in considered if row.get("gain_fraction_of_baseline_mean",0)>steering_effect]
            if deferred:
                selection_reason+=f"; {len(deferred)} stronger observed causal result(s) were deferred: "+", ".join(
                    f"{x['stage']} {x['reference']} {x['evidence_status']} ({x['reason']})" for x in deferred[:3])
            limit=self.c["generation"]["max_new_tokens"]
            validation_specificity_pass=candidate.get("specificity_pass",False)
        records=self._eval_rows("test");controls=self._eval_rows("control")
        base=self._base(records,limit);cb=self._base(controls,limit)
        changed,stats=self._run_variant(records,spec,base,limit)
        cc,cstats=self._run_variant(controls,spec,cb,limit)
        drift=float(np.mean([abs(x["score"]-y["score"]) for x,y in zip(cc,cb)])) / max(float(np.mean([abs(x["score"]) for x in cb])),self.c["behavior"]["scale_floor"])
        nll_rows=[]
        for record,normal in zip(controls,cb):
            self.store.check_budget()
            tokens=normal["token_ids"][:self.c["evaluation"]["reference_tokens"]]
            if not tokens:continue
            baseline_nll=self.engine.backend.reference_nll(record,tokens)
            changed_nll=self.engine.backend.reference_nll(record,tokens,self.bundle(),spec)
            nll_rows.append({"id":record["id"],"reference_tokens":len(tokens),"baseline_nll":baseline_nll,
                             "intervention_nll":changed_nll,"nll_increase":changed_nll-baseline_nll})
        nll_delta=float(np.mean([r["nll_increase"] for r in nll_rows])) if nll_rows else None
        write_csv(p/"reference_nll.csv",nll_rows)
        self._records(p,changed+cc)
        write_json(p/"baseline_outputs.json",base+cb)
        heldout_behavior_signal=all(stats.get(key,False) for key in
                                    ("effect_pass","win_rate_pass","confidence_pass",
                                     "repeat_pass","content_check_pass"))
        heldout_effect_pass=heldout_behavior_signal and stats.get("censoring_pass",False)
        selection_quality_confirmed=selected_status=="promising"
        passed=(heldout_effect_pass and validation_specificity_pass and selection_quality_confirmed and
                drift<=self.c["search"]["max_control_drift"]
                and (nll_delta is None or nll_delta<=self.c["evaluation"]["max_reference_nll_increase"]))
        heldout_status=("validated_candidate" if passed else
                        "promising_censored" if heldout_behavior_signal and
                        (selected_status=="promising_censored" or not stats.get("censoring_pass",False)) else
                        "exploratory" if heldout_behavior_signal else "rejected")
        return self.finish(p,{"passed":passed,"selected_intervention":spec.serial(),"test":stats,"control":cstats,
                             "selection_reason":selection_reason,"candidate_source_stage":selected_source,
                             "candidate_evidence_status":selected_status,"evaluation_status":heldout_status,
                             "deferred_candidates":deferred if selected_source in {"sweep","refine"} else [],
                             "evaluation_max_new_tokens":limit,
                             "heldout_effect_pass":heldout_effect_pass,
                             "heldout_behavior_signal":heldout_behavior_signal,
                             "validation_specificity_pass":validation_specificity_pass,
                             "selection_quality_confirmed":selection_quality_confirmed,
                             "control_absolute_drift_fraction":drift,"control_reference_nll_increase":nll_delta,
                             "needs_human_quality_review":True,
                             "note":"Lexical/repetition checks do not establish factual accuracy, and a small held-out set does not establish general performance"})

    def stage_report(self,p):
        from .report import build_report
        paths=build_report(self.store,p)
        return self.finish(p,{"files":paths,"weights_modified":False})
