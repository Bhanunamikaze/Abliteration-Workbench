"""Immutable, resumable evaluation of a frozen intervention on new prompts."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import html
import math
import os
import shutil
import time

import numpy as np

from . import __version__
from .config import DEFAULTS
from .data import load_replication_dataset, messages, split_rows
from .directions import DirectionBundle
from .engine import Engine
from .interventions import Intervention
from .metrics import binary_paired_transitions, paired_stats, scorer_provenance
from .store import RunStore
from .util import (BudgetReached, LabError, append_jsonl, digest, file_hash, now,
                   read_journal, read_json, write_csv, write_json, atomic_text)


OUTPUT_FILES = ("baseline_outputs.json", "intervention_outputs.json",
                "control_baseline.json", "control_intervention.json",
                "results.csv", "manual_review.csv", "reference_nll.csv",
                "runtime_identity.json", "summary.json", "report.md", "report.html")


def _policy(config: dict) -> dict:
    defaults = DEFAULTS["evaluation"]
    evaluation = config["evaluation"]
    keys = ("binary_min_samples", "binary_min_discordant",
            "binary_min_favorable_fraction", "binary_max_regression_fraction",
            "binary_max_control_regression_fraction", "max_scorer_review_fraction")
    return {key: evaluation.get(key, defaults[key]) for key in keys}


def _source_selection(source: RunStore) -> tuple[dict, dict]:
    if not source.completed("inspect") or not source.completed("directions") or not source.completed("evaluate"):
        raise LabError("Replication requires completed inspect, directions and evaluate stages")
    evaluation = source.result("evaluate")
    raw = evaluation.get("selected_intervention")
    if not isinstance(raw, dict):
        raise LabError("Source evaluation has no selected intervention to freeze")
    try:
        spec = Intervention(**{**raw, "layers": tuple(raw["layers"])})
    except (TypeError, KeyError) as error:
        raise LabError("Source selected_intervention is invalid") from error
    if not spec.layers or spec.control_seed is not None:
        raise LabError("Replication requires a nonempty, nonrandom selected intervention")
    bundle = DirectionBundle.load(source.stage_dir("directions"))
    if max(spec.layers) >= bundle.basis.shape[0] or min(spec.layers) < 0:
        raise LabError("Selected intervention has a layer outside the direction bundle")
    model_identity = source.result("inspect")["model_identity"]
    if bundle.metadata.get("model_identity") != model_identity:
        raise LabError("Source direction and inspected model identities disagree")
    if spec.reference == "negative" and (spec.site != "residual" or not bundle.metadata.get("calibrated", True)):
        raise LabError("Negative reference requires calibrated residual directions")
    source_evidence = {
        "candidate_source_stage": evaluation.get("candidate_source_stage"),
        "candidate_evidence_status": evaluation.get("candidate_evidence_status"),
        "validation_specificity_pass": evaluation.get("validation_specificity_pass", False),
        "selection_quality_confirmed": evaluation.get("selection_quality_confirmed", False),
        "original_evaluation_status": evaluation.get("evaluation_status"),
        "original_evaluation_passed": evaluation.get("passed", False),
    }
    return spec.serial(), source_evidence


class ReplicationStore:
    """Separate run type; discovery STAGES and source artifacts stay untouched."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.meta = read_json(self.root / "replication.json")
        if self.meta.get("schema_version") != 1 or self.meta.get("kind") != "frozen_intervention_replication":
            raise LabError("Unsupported replication manifest")
        self.config = read_json(self.root / "config.json")
        self.data = read_json(self.root / "dataset.json")
        state = self.state()
        if state.get("manifest_hash") != digest(self.meta):
            raise LabError("Replication manifest changed after initialization")
        if digest(self.config) != self.meta["source_config_hash"]:
            raise LabError("Frozen replication config changed")
        if digest(self.data) != self.meta["replication_dataset_hash"]:
            raise LabError("Replication dataset snapshot changed")
        if self.meta.get("scorer_provenance") is not None:
            actual = scorer_provenance(self.config["behavior"]["plugin"])
            if actual != self.meta["scorer_provenance"]:
                raise LabError("Frozen scorer source changed since replication initialization")
        for name, expected in self.meta["direction_file_hashes"].items():
            if file_hash(self.root / "directions" / name) != expected:
                raise LabError("Frozen direction file changed")
        if DirectionBundle.load(self.root / "directions").fingerprint() != self.meta["directions_fingerprint"]:
            raise LabError("Frozen direction fingerprint changed")
        self.source = RunStore(self.meta["source_run"])
        self.source.dataset()
        if self.source.meta["config_hash"] != self.meta["source_config_hash"] or self.source.meta["dataset_hash"] != self.meta["source_dataset_hash"]:
            raise LabError("Source run identity changed since replication initialization")
        if not self.source.completed("directions") or not self.source.completed("evaluate"):
            raise LabError("Source directions/evaluation are no longer complete")
        if self.source.result("inspect")["model_identity"] != self.meta["model_identity"]:
            raise LabError("Source model identity changed")
        if DirectionBundle.load(self.source.stage_dir("directions")).fingerprint() != self.meta["directions_fingerprint"]:
            raise LabError("Source direction fingerprint changed")
        if file_hash(self.source.stage_dir("evaluate") / "result.json") != self.meta["source_evaluation_hash"]:
            raise LabError("Source selected evaluation changed")
        if self.source.result("evaluate").get("selected_intervention") != self.meta["intervention"]:
            raise LabError("Source selected intervention changed")
        if state.get("status") == "complete":
            for relative, expected in state.get("artifacts", {}).items():
                if file_hash(self.root / relative) != expected:
                    raise LabError(f"Completed replication artifact changed: {relative}")
        self.started = time.monotonic()
        self.prior_seconds = float(state.get("seconds_spent", 0.0))
        self.cache_path = self.root / "cache" / "generations.jsonl"
        self.cache = {r["job_key"]: r for r in read_journal(self.cache_path, ignore_tail=True)}

    @classmethod
    def create(cls, source: RunStore, dataset_path: str | Path, root: str | Path,
               *, max_generations: int | None = None, max_seconds: float | None = None,
               retry_max_new_tokens: int | None = None):
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise LabError(f"Replication destination is not empty: {root}")
        rows, audit = load_replication_dataset(dataset_path)
        original_prompts = {digest(r["neutral"]) for r in source.dataset()}
        repeated = [r["id"] for r in rows if digest(r["neutral"]) in original_prompts]
        if repeated:
            raise LabError("Replication prompts overlap source discovery/evaluation prompts: " + ", ".join(repeated[:5]))
        intervention, source_evidence = _source_selection(source)
        bundle = DirectionBundle.load(source.stage_dir("directions"))
        source_evaluation = source.result("evaluate")
        cap = source_evaluation.get("evaluation_max_new_tokens", source.config["generation"]["max_new_tokens"])
        if type(cap) is not int or cap < 1:
            raise LabError("Source evaluation generation cap is invalid")
        retry_cap=(retry_max_new_tokens if retry_max_new_tokens is not None else
                   max(cap*2,source.config["generation"].get("confirm_max_new_tokens",cap)))
        if type(retry_cap) is not int or retry_cap<=cap:
            raise LabError("Replication retry ceiling must be an integer above the source evaluation ceiling")
        n = len(rows)
        # Reserve both arms for one automatic higher-ceiling confirmation.
        generations = max_generations if max_generations is not None else max(source.config["budget"]["max_generations"], 4*n)
        seconds = max_seconds if max_seconds is not None else max(source.config["budget"]["max_seconds"], 86400)
        if (type(generations) is not int or generations < 4*n or isinstance(seconds, bool)
                or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0):
            raise LabError("Replication budget must allow both arms and one higher-cap retry for every prompt, with positive finite execution time")
        scorer = source.meta.get("scorer_provenance")
        if source.config["behavior"]["metric"] == "custom" and scorer is None:
            raise LabError("Source custom scorer lacks recorded provenance; an external replication cannot prove scorer continuity")
        direction_files = {name: file_hash(source.stage_dir("directions") / name)
                           for name in ("directions.safetensors", "directions.json")}
        meta = {
            "schema_version":1,"kind":"frozen_intervention_replication",
            "created_at":now(),"tool_version":__version__,
            "source_run":str(source.root),"source_config_hash":source.meta["config_hash"],
            "source_dataset_hash":source.meta["dataset_hash"],
            "source_evaluation_hash":file_hash(source.stage_dir("evaluate") / "result.json"),
            "model_identity":source.result("inspect")["model_identity"],
            "source_model_capabilities":source.result("inspect").get("capabilities",{}),
            "directions_fingerprint":bundle.fingerprint(),
            "direction_file_hashes":direction_files,
            "intervention":intervention,"source_evidence":source_evidence,
            "replication_dataset_hash":digest(rows),
            "input_dataset_path":str(Path(dataset_path).resolve()),
            "input_dataset_file_hash":file_hash(dataset_path),
            "behavior":deepcopy(source.config["behavior"]),
            "scorer_provenance":scorer,
            "generation":deepcopy(source.config["generation"]),
            "generation_cap":cap,
            "retry_generation_cap":retry_cap,
            "evaluation_policy":_policy(source.config),
            "budget":{"max_generations":generations,"max_seconds":seconds},
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "cache").mkdir()
        (root / "directions").mkdir()
        for name in direction_files:
            shutil.copy2(source.stage_dir("directions") / name, root / "directions" / name)
        write_json(root / "config.json", source.config)
        write_json(root / "dataset.json", rows)
        write_json(root / "dataset_audit.json", audit)
        write_json(root / "replication.json", meta)
        write_json(root / "state.json", {"status":"ready","manifest_hash":digest(meta),
                                          "generation_count":0,"seconds_spent":0.0,"artifacts":{}})
        return cls(root)

    def state(self) -> dict:
        return read_json(self.root / "state.json")

    def dataset(self) -> list[dict]:
        return self.data

    def completed(self, stage: str, verify: bool = True) -> bool:
        return stage == "inspect"

    def result(self, stage: str) -> dict:
        if stage != "inspect":
            raise LabError(f"Replication has no discovery stage {stage}")
        return self.source.result("inspect")

    def event(self, kind: str, **data):
        append_jsonl(self.root / "events.jsonl", {"time":now(),"kind":kind,**data})

    @contextmanager
    def lock(self):
        path = self.root / ".replication.lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise LabError(f"Replication is locked: {path}") from error
        with os.fdopen(fd, "w") as output:
            output.write(str(os.getpid()))
        self.prior_seconds=float(self.state().get("seconds_spent",0.0))
        self.started=time.monotonic()
        try:
            self.cache = {r["job_key"]:r for r in read_journal(self.cache_path, repair_tail=True)}
            yield
        finally:
            state=self.state()
            if state.get("status")!="complete":
                state["seconds_spent"]=self.prior_seconds+time.monotonic()-self.started
                write_json(self.root/"state.json",state)
            path.unlink(missing_ok=True)

    def check_budget(self, new_generations: int = 0):
        if self.state()["generation_count"] + new_generations > self.meta["budget"]["max_generations"]:
            raise BudgetReached("Replication generation budget reached; cached batches remain available")
        if self.prior_seconds + time.monotonic()-self.started > self.meta["budget"]["max_seconds"]:
            raise BudgetReached("Replication time budget reached; cached batches remain available")

    def put_generation(self, value: dict):
        key=value["job_key"]
        if key not in self.cache:
            append_jsonl(self.cache_path,value)
            self.cache[key]=value
            state=self.state();state["generation_count"]+=1
            write_json(self.root / "state.json",state)

    def verify_input(self, source_path: str | Path, dataset_path: str | Path,
                     max_generations: int | None = None, max_seconds: float | None = None,
                     retry_max_new_tokens: int | None = None):
        if Path(source_path).resolve() != Path(self.meta["source_run"]):
            raise LabError("Existing replication belongs to a different source run")
        rows,_=load_replication_dataset(dataset_path)
        if digest(rows) != self.meta["replication_dataset_hash"] or file_hash(dataset_path) != self.meta["input_dataset_file_hash"]:
            raise LabError("Replication input dataset changed; use a new replication directory")
        if max_generations is not None and max_generations != self.meta["budget"]["max_generations"]:
            raise LabError("Replication generation budget is immutable")
        if max_seconds is not None and max_seconds != self.meta["budget"]["max_seconds"]:
            raise LabError("Replication time budget is immutable")
        if retry_max_new_tokens is not None and retry_max_new_tokens != self.meta["retry_generation_cap"]:
            raise LabError("Replication retry ceiling is immutable")

    def finish(self, summary: dict):
        state=self.state()
        artifacts={name:file_hash(self.root / name) for name in OUTPUT_FILES}
        for path in (self.root / "attempts").rglob("*") if (self.root / "attempts").exists() else []:
            if path.is_file():artifacts[str(path.relative_to(self.root))]=file_hash(path)
        state.update(status="complete",seconds_spent=state.get("seconds_spent",0.0)+time.monotonic()-self.started,
                     completed_at=now(),artifacts=artifacts)
        write_json(self.root / "state.json",state)
        self.event("replication_completed",status=summary["status"],passed=summary["passed"])


def _review(rows: list[dict]) -> dict:
    reasons=Counter()
    required=0
    for row in rows:
        details=row.get("score_details") or {}
        required+=bool(details.get("review_required",False))
        reasons.update(details.get("review_reasons") or [])
    return {"requiring_review":required,"fraction":required/len(rows) if rows else 0.0,
            "reason_counts":dict(sorted(reasons.items()))}


def _pair_summary(records: list[dict], base: list[dict], changed: list[dict], config: dict) -> dict:
    ids=[r["id"] for r in records]
    if ids != [r["id"] for r in base] or ids != [r["id"] for r in changed]:
        raise LabError("Replication baseline/intervention output IDs are not aligned")
    behavior=config["behavior"]
    paired=paired_stats([r["score"] for r in base],[r["score"] for r in changed],
                        behavior["goal"],config["generation"]["seed"],behavior["scale_floor"])
    transitions=binary_paired_transitions([r["score"] for r in base],
                                           [r["score"] for r in changed],behavior["goal"])
    repeat=float(np.mean([x["repeat4"]-y["repeat4"] for x,y in zip(changed,base)]))
    terms=[x["required_term_fraction"]-y["required_term_fraction"] for x,y in zip(changed,base)
           if x["required_term_fraction"] is not None and y["required_term_fraction"] is not None]
    return {**paired,"score_domain":"binary" if transitions is not None else "continuous",
            "binary_transitions":transitions,
            "baseline_cap_rate":float(np.mean([r["hit_limit"] for r in base])),
            "cap_rate":float(np.mean([r["hit_limit"] for r in changed])),
            "uncensored_pair_fraction":float(np.mean([not (x["hit_limit"] or y["hit_limit"]) for x,y in zip(base,changed)])),
            "repeat_increase":repeat,
            "empty_fraction":float(np.mean([r["empty"] for r in changed])),
            "required_term_change":float(np.mean(terms)) if terms else None,
            "baseline_review":_review(base),"intervention_review":_review(changed)}


def _quality(summary: dict, search: dict) -> dict:
    return {"censoring_pass":max(summary["baseline_cap_rate"],summary["cap_rate"])<=search["max_cap_rate"],
            "repeat_pass":summary["repeat_increase"]<=search["max_repeat_increase"],
            "content_check_pass":summary["empty_fraction"]==0 and
                (summary["required_term_change"] is None or summary["required_term_change"]>=-0.1)}


def _decide(test: dict, control: dict, nll_increase: float | None, meta: dict, config: dict) -> dict:
    policy=meta["evaluation_policy"];search=config["search"];evaluation=config["evaluation"]
    binary=test["binary_transitions"]
    gates={"positive_effect":test["gain_fraction_of_baseline_mean"]>=search["min_effect_fraction"] and test["mean_gain"]>0,
           "positive_confidence":test["ci95_low"]>0,
           "source_specificity":bool(meta["source_evidence"]["validation_specificity_pass"]),
           "source_quality_confirmed":bool(meta["source_evidence"]["selection_quality_confirmed"]),
           "control_drift":control["absolute_score_drift_fraction"]<=search["max_control_drift"],
           "reference_nll":(evaluation["reference_tokens"]==0 or
                            (nll_increase is not None and nll_increase<=evaluation["max_reference_nll_increase"])),
           "review_fraction":max(test["intervention_review"]["fraction"],
                                 control["intervention_review"]["fraction"])<=policy["max_scorer_review_fraction"]}
    if binary is not None:
        gates.update(binary_sample=test["n"]>=policy["binary_min_samples"],
                     binary_discordant=binary["discordant"]>=policy["binary_min_discordant"],
                     binary_favorable_fraction=(binary["favorable_fraction_of_discordant"] is not None and
                           binary["favorable_fraction_of_discordant"]>=policy["binary_min_favorable_fraction"]),
                     binary_regression=binary["regression_fraction"]<=policy["binary_max_regression_fraction"])
        control_binary=control["binary_transitions"]
        gates["control_regression"]=(control_binary is not None and
            control_binary["regression_fraction"]<=policy["binary_max_control_regression_fraction"])
    else:
        gates["continuous_win_rate"]=test["win_rate"]>=search["min_win_rate"]
        gates["control_regression"]=True
    for split, values in (("test",test),("control",control)):
        for key, value in _quality(values,search).items():
            gates[f"{split}_{key}"]=value
    passed=all(gates.values())
    promising=(gates["positive_effect"] and gates["positive_confidence"] and
               gates["source_specificity"] and gates["source_quality_confirmed"])
    if passed:status="validated_replication"
    elif promising and not (gates["test_censoring_pass"] and gates["control_censoring_pass"]):
        status="promising_censored"
    elif promising and not gates["review_fraction"]:
        status="needs_human_review"
    elif promising:
        status="promising_replication"
    else:status="failed_replication"
    return {"passed":passed,"status":status,"promotion_gates":gates,
            "failed_gates":[key for key,value in gates.items() if not value],
            "semantic_confirmation":False if any((test["baseline_review"]["requiring_review"],
                                                      test["intervention_review"]["requiring_review"],
                                                      control["baseline_review"]["requiring_review"],
                                                      control["intervention_review"]["requiring_review"])) else None}


def _review_manifest(records: list[dict], base: list[dict], changed: list[dict], goal: str) -> list[dict]:
    rows=[]
    for index,(record,b,a) in enumerate(zip(records,base,changed)):
        if b["score"]==a["score"] and not b.get("score_details",{}).get("review_required") and not a.get("score_details",{}).get("review_required"):
            continue
        favorable=(b["score"]>a["score"]) if goal=="decrease" else (b["score"]<a["score"])
        transition="favorable" if favorable else "unfavorable" if b["score"]!=a["score"] else "unchanged_review_flag"
        bd=b.get("score_details") or {};ad=a.get("score_details") or {}
        rows.append({"id":record["id"],"split":record["split"],"category":record.get("category"),
                     "baseline_score":b["score"],"intervention_score":a["score"],"transition":transition,
                     "baseline_probability":bd.get("refusal_probability",bd.get("probability")),
                     "intervention_probability":ad.get("refusal_probability",ad.get("probability")),
                     "baseline_tokens":b["tokens"],"intervention_tokens":a["tokens"],
                     "baseline_review_required":bd.get("review_required",False),
                     "intervention_review_required":ad.get("review_required",False),
                     "review_reasons":sorted(set((bd.get("review_reasons") or [])+(ad.get("review_reasons") or []))),
                     "output_index":index})
    return rows


def _report(store: ReplicationStore, summary: dict):
    test=summary["test"];control=summary["control"]
    binary=test["binary_transitions"];control_binary=control["binary_transitions"]
    spec=store.meta["intervention"]
    lines=["# Frozen-intervention replication","",
           f"Status: **{summary['status']}** (`passed={str(summary['passed']).lower()}`).",
           f"Source run: `{store.meta['source_run']}`",
           f"Direction fingerprint: `{store.meta['directions_fingerprint']}`",
           f"Frozen intervention: `{spec}`","",
           "The direction and intervention were fixed before this dataset was scored. This report describes the configured metric on these prompts; it does not establish universal layer meaning, factual quality, or full behavior removal.","",
           "## Generation ceiling and confirmation","",
           f"Source evaluation ceiling: {store.meta['generation_cap']} tokens; final replication ceiling: {summary['generation_cap']} tokens."]
    if summary.get("higher_cap_retry_performed"):
        first=summary["attempts"][0]
        lines += [f"The initial result was **{first['status']}** at {first['generation_cap']} tokens, with test baseline/intervention cap rates {first['test']['baseline_cap_rate']:.3g} / {first['test']['cap_rate']:.3g}. It remains saved in `attempts/initial/`. A matched higher-ceiling retry ran automatically before the promotion decision."]
    lines += ["",
           "## Held-out test","",
           f"N: {test['n']}; baseline mean: {test['mean_baseline']:.6g}; intervention mean: {test['mean_intervention']:.6g}; paired gain: {test['mean_gain']:.6g}.",
           f"Gain fraction of baseline scale: {test['gain_fraction_of_baseline_mean']:.6g}; CI95: [{test['ci95_low']:.6g}, {test['ci95_high']:.6g}].",
           f"Baseline/intervention cap rates: {test['baseline_cap_rate']:.3g} / {test['cap_rate']:.3g}."]
    if binary:
        lines += [f"Favorable/unfavorable transitions: {binary['favorable_transitions']} / {binary['unfavorable_transitions']}; ties: {binary['ties']}.",
                  f"Favorable among discordant: {binary['favorable_fraction_of_discordant']}; conditional improvement: {binary['conditional_improvement_fraction']}.",
                  f"Exact paired binomial p-value: {binary['binary_transition_test']['p_value']:.6g} (supporting evidence, not the sole decision rule)."]
    lines += ["","## Controls","",
              f"N: {control['n']}; baseline mean: {control['mean_baseline']:.6g}; intervention mean: {control['mean_intervention']:.6g}.",
              f"Absolute score drift: {control['absolute_score_drift']:.6g}; normalized drift: {control['absolute_score_drift_fraction']:.6g}.",
              f"Baseline/intervention cap rates: {control['baseline_cap_rate']:.3g} / {control['cap_rate']:.3g}."]
    if control_binary:
        lines += [f"Control favorable/unfavorable transitions: {control_binary['favorable_transitions']} / {control_binary['unfavorable_transitions']}."]
    lines += [f"Mean fixed-reference NLL increase: {summary['control_reference_nll_increase']}.",
              "","## Scorer review","",]
    for split, data in (("test",test),("control",control)):
        for arm in ("baseline","intervention"):
            review=data[f"{arm}_review"]
            lines += [f"{split} {arm}: {review['requiring_review']}/{data['n']} responses require review; reasons: {review['reason_counts']}."]
    if any("classifier_input_truncated" in data[f"{arm}_review"]["reason_counts"]
           for data in (test,control) for arm in ("baseline","intervention")):
        lines += ["Classifier input truncation occurred. Its binary scores remain provisional until responses are reviewed."]
    lines += ["","## Promotion gates","",]
    lines += [f"- {name}: {'pass' if passed else 'fail'}" for name,passed in summary["promotion_gates"].items()]
    lines += ["","The runtime recipe applies an activation hook; it does not edit checkpoint weights.",
              "Multi-layer effects are attributed to the complete intervention, not an individual layer.",""]
    markdown="\n".join(lines)
    atomic_text(store.root / "report.md",markdown)
    # Keep the HTML offline and escape all provenance/labels. Full generation
    # text remains in JSON artifacts, not embedded in this summary page.
    body="<br>\n".join(html.escape(line) for line in lines)
    document=("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Frozen replication</title>"
              "<style>body{font:16px/1.5 system-ui;max-width:1000px;margin:40px auto;padding:0 20px}"
              "br{line-height:1.7}</style><h1>Frozen-intervention replication</h1><div>"+body+"</div></html>")
    atomic_text(store.root / "report.html",document)


def run_replication(store: ReplicationStore) -> dict:
    """Use only frozen model, directions, intervention and scorer; never search."""
    if store.state().get("status") == "complete":
        return read_json(store.root / "summary.json")
    with store.lock():
        if store.state().get("status") == "complete":
            return read_json(store.root / "summary.json")
        store.event("replication_started",intervention=store.meta["intervention"])
        records=store.dataset();test_records=split_rows(records,"test");control_records=split_rows(records,"control")
        spec=Intervention(**{**store.meta["intervention"],"layers":tuple(store.meta["intervention"]["layers"])})
        bundle=DirectionBundle.load(store.root / "directions")
        engine=Engine(store)
        if store.meta["scorer_provenance"] is not None and scorer_provenance(
                store.config["behavior"]["plugin"]) != store.meta["scorer_provenance"]:
            raise LabError("Frozen scorer source changed before generation")
        backend=engine.backend
        if digest(backend.identity) != store.meta["model_identity"]:
            raise LabError("Replication backend model/revision/dtype/chat template identity differs from source")
        backend._validate_bundle(bundle)
        def evaluate_at(limit: int):
            test_base=engine.generate(test_records,max_new_tokens=limit)
            test_changed=engine.generate(test_records,bundle,spec,limit)
            control_base=engine.generate(control_records,max_new_tokens=limit)
            control_changed=engine.generate(control_records,bundle,spec,limit)
            test=_pair_summary(test_records,test_base,test_changed,store.config)
            control=_pair_summary(control_records,control_base,control_changed,store.config)
            absolute=float(np.mean([abs(a["score"]-b["score"]) for a,b in zip(control_changed,control_base)]))
            scale=max(float(np.mean([abs(x["score"]) for x in control_base])),store.config["behavior"]["scale_floor"])
            control["absolute_score_drift"]=absolute
            control["absolute_score_drift_fraction"]=absolute/scale
            nll=[]
            for record,normal in zip(control_records,control_base):
                store.check_budget()
                tokens=normal["token_ids"][:store.config["evaluation"]["reference_tokens"]]
                if not tokens:continue
                before=backend.reference_nll(record,tokens)
                after=backend.reference_nll(record,tokens,bundle,spec)
                nll.append({"id":record["id"],"reference_tokens":len(tokens),"baseline_nll":before,
                            "intervention_nll":after,"nll_increase":after-before})
            nll_increase=float(np.mean([r["nll_increase"] for r in nll])) if nll else None
            decision=_decide(test,control,nll_increase,store.meta,store.config)
            summary={"schema_version":1,"kind":"frozen_intervention_replication_result",
                 "source_run":store.meta["source_run"],
                 "source_config_hash":store.meta["source_config_hash"],
                 "source_dataset_hash":store.meta["source_dataset_hash"],
                 "replication_dataset_hash":store.meta["replication_dataset_hash"],
                 "model_identity":store.meta["model_identity"],
                 "directions_fingerprint":store.meta["directions_fingerprint"],
                 "intervention":store.meta["intervention"],
                 "generation_cap":limit,"behavior":store.meta["behavior"],
                 "scorer_provenance":store.meta["scorer_provenance"],
                 "source_evidence":store.meta["source_evidence"],
                 "evaluation_policy":store.meta["evaluation_policy"],
                 "test":test,"control":control,
                 "control_reference_nll_increase":nll_increase,
                 **decision,
                 "note":"Validation applies to the frozen intervention and configured metric on this dataset; classifier review flags are not semantic ground truth."}
            return summary,(test_base,test_changed,control_base,control_changed,nll)

        limit=store.meta["generation_cap"]
        summary,outputs=evaluate_at(limit)
        initial={"generation_cap":limit,"status":summary["status"],"passed":summary["passed"],
                 "test":summary["test"],"control":summary["control"],
                 "promotion_gates":summary["promotion_gates"],"failed_gates":summary["failed_gates"]}
        attempts=[initial]
        if summary["status"] == "promising_censored":
            # Preserve the censored finding before starting a resumable,
            # matched baseline/intervention retry at the frozen higher cap.
            initial_dir=store.root / "attempts" / "initial"
            initial_dir.mkdir(parents=True,exist_ok=True)
            for name,values in zip(("baseline_outputs.json","intervention_outputs.json",
                                    "control_baseline.json","control_intervention.json"),outputs[:4]):
                write_json(initial_dir / name,values)
            write_json(initial_dir / "summary.json",initial)
            store.event("censored_replication_retry",initial_cap=limit,
                        retry_cap=store.meta["retry_generation_cap"])
            limit=store.meta["retry_generation_cap"]
            summary,outputs=evaluate_at(limit)
            attempts.append({"generation_cap":limit,"status":summary["status"],
                             "passed":summary["passed"],"test":summary["test"],
                             "control":summary["control"],
                             "promotion_gates":summary["promotion_gates"],
                             "failed_gates":summary["failed_gates"]})
        summary["attempts"]=attempts
        summary["higher_cap_retry_performed"]=len(attempts)>1
        test_base,test_changed,control_base,control_changed,nll=outputs
        write_json(store.root / "baseline_outputs.json",test_base)
        write_json(store.root / "intervention_outputs.json",test_changed)
        write_json(store.root / "control_baseline.json",control_base)
        write_json(store.root / "control_intervention.json",control_changed)
        write_json(store.root / "runtime_identity.json",backend.identity)
        write_csv(store.root / "reference_nll.csv",nll)
        write_csv(store.root / "results.csv",[
            {"id":record["id"],"split":record["split"],"category":record.get("category"),
             "baseline_score":before["score"],"intervention_score":after["score"],
             "baseline_tokens":before["tokens"],"intervention_tokens":after["tokens"],
             "baseline_hit_limit":before["hit_limit"],"intervention_hit_limit":after["hit_limit"]}
            for group,original,modified in ((test_records,test_base,test_changed),(control_records,control_base,control_changed))
            for record,before,after in zip(group,original,modified)])
        write_csv(store.root / "manual_review.csv",
                  _review_manifest(test_records,test_base,test_changed,store.config["behavior"]["goal"])+
                  _review_manifest(control_records,control_base,control_changed,store.config["behavior"]["goal"]))
        write_json(store.root / "summary.json",summary)
        _report(store,summary)
        store.finish(summary)
        return summary


def open_or_create(source_path: str | Path, dataset_path: str | Path, output: str | Path,
                   *, max_generations: int | None = None, max_seconds: float | None = None,
                   retry_max_new_tokens: int | None = None) -> ReplicationStore:
    root=Path(output).resolve()
    if (root / "replication.json").exists():
        replica=ReplicationStore(root)
        replica.verify_input(source_path,dataset_path,max_generations,max_seconds,retry_max_new_tokens)
        return replica
    if root.exists() and any(root.iterdir()):
        raise LabError(f"Replication destination is not empty: {root}")
    return ReplicationStore.create(RunStore(source_path),dataset_path,root,
                                   max_generations=max_generations,max_seconds=max_seconds,
                                   retry_max_new_tokens=retry_max_new_tokens)


def write_recipe(replica: ReplicationStore, destination: str | Path):
    replica=ReplicationStore(replica.root)
    summary=read_json(replica.root / "summary.json") if replica.state().get("status")=="complete" else None
    if summary is None or not summary.get("passed") or summary.get("status")!="validated_replication":
        raise LabError("A passing frozen replication is required before writing a recipe")
    value={"intervention":replica.meta["intervention"],
           "directions_fingerprint":replica.meta["directions_fingerprint"],
           "source_run":replica.meta["source_run"],"replication_run":str(replica.root),
           "replication_dataset_hash":replica.meta["replication_dataset_hash"],
           "replication_summary_hash":file_hash(replica.root / "summary.json"),
           "validated_by":"external_frozen_intervention_replication",
           "note":"Runtime intervention recipe. Validated by external frozen-intervention replication on the configured metric. Not an edited checkpoint; classifier review flags remain visible in the report."}
    write_json(destination,value)
    return value


def dataset_from_historical_outputs(outputs_dir: str | Path, destination: str | Path) -> dict:
    """Recover prompts only from external baseline files; never import scores as evidence."""
    source=Path(outputs_dir).resolve();out=Path(destination).resolve()
    if out.exists():raise LabError(f"Replication dataset destination already exists: {out}")
    rows=[];seen=set()
    for split in ("test","control"):
        base=read_json(source / f"{split}_baseline.json")
        changed=read_json(source / f"{split}_intervention.json")
        if len(base)!=len(changed) or not base:
            raise LabError(f"Historical {split} output arms are empty or misaligned")
        for before,after in zip(base,changed):
            key=before.get("id")
            if key in seen or key!=after.get("id") or before.get("split")!=split or after.get("split")!=split:
                raise LabError(f"Historical {split} output IDs/splits are missing, duplicated, or misaligned")
            prompt=messages(before.get("messages"),f"{key}.messages")
            if prompt!=messages(after.get("messages"),f"{key}.messages"):
                raise LabError(f"Historical output prompts differ between arms at {key}")
            seen.add(key)
            rows.append({"id":key,"split":split,"neutral":prompt,
                         **{k:before[k] for k in ("category","prompt_label") if k in before}})
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,rows)
    try:
        _,audit=load_replication_dataset(out)
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    return {"destination":str(out),"dataset_hash":audit["dataset_hash"],
            "counts":audit["counts"],
            "note":"Only prompts and labels were copied. Historical scores are not first-class replication evidence; rerun replicate."}
