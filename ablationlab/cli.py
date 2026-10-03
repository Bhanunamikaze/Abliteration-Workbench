"""Command-line entrypoint; all expensive execution is explicit and logged."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from . import __version__
from .util import LabError, BudgetReached, read_json, write_json, dumps, environment
from .config import load_config, DEFAULTS
from .data import load_dataset
from .store import RunStore, STAGES

def parser():
    p=argparse.ArgumentParser(prog="abliteration",description="Local, file-driven representation-engineering experiments")
    p.add_argument("--version",action="version",version=__version__)
    sub=p.add_subparsers(dest="command",required=True)
    init=sub.add_parser("init",help="Validate files and initialize an immutable run snapshot")
    init.add_argument("--config",required=True);init.add_argument("--run",required=True);init.add_argument("--set",action="append",default=[])
    val=sub.add_parser("validate",help="Validate config/dataset without loading a model")
    val.add_argument("--config",required=True);val.add_argument("--set",action="append",default=[])
    for name in ["status","plan","report","unlock"]:
        cmd=sub.add_parser(name);cmd.add_argument("--run",required=True)
    plots=sub.add_parser("plot",help="Plot completed-run measurements without loading the model")
    plots.add_argument("--run",required=True);plots.add_argument("--out")
    run=sub.add_parser("run",help="Automatic pipeline or an individual explicit stage")
    run.add_argument("--run",required=True);run.add_argument("--stage",choices=STAGES)
    run.add_argument("--from-stage",choices=STAGES);run.add_argument("--until",choices=STAGES)
    run.add_argument("--with-deps",action="store_true");run.add_argument("--reset",action="store_true")
    run.add_argument("--dry-run",action="store_true")
    fork=sub.add_parser("fork",help="Fork config/history; preserve compatible upstream stages")
    fork.add_argument("--run",required=True);fork.add_argument("--new-run",required=True);fork.add_argument("--set",action="append",default=[])
    reset=sub.add_parser("reset",help="Archive a stage and invalidate dependent stages")
    reset.add_argument("--run",required=True);reset.add_argument("--stage",choices=STAGES,required=True)
    legacy=sub.add_parser("import-legacy",help="Audit old 05-09 artifacts without running the model")
    legacy.add_argument("--source",required=True);legacy.add_argument("--out",required=True)
    legacy.add_argument("--no-copy",action="store_true")
    imp=sub.add_parser("import-directions",help="Import actual legacy .pt vectors for manual exploratory reuse")
    imp.add_argument("--run",required=True);imp.add_argument("--file",required=True)
    imp.add_argument("--raw-blocks",action="store_true",help="Only for vectors known to come from raw block hooks; old Code03 should omit this")
    ep=sub.add_parser("export-plan",help="Write a separate dense-writer checkpoint-edit plan")
    ep.add_argument("--run",required=True);ep.add_argument("--out",required=True)
    ex=sub.add_parser("export",help="Explicit copied-checkpoint projection and fixed-input equivalence check")
    ex.add_argument("--run",required=True);ex.add_argument("--plan",required=True);ex.add_argument("--out",required=True)
    bundled=sub.add_parser("bundle",help="Package a passing runtime intervention with unchanged model weights and a local runner")
    bundled.add_argument("--run",required=True);bundled.add_argument("--out",required=True)
    gen=sub.add_parser("generate",help="Generate with a saved runtime recipe, without changing weights")
    gen.add_argument("--run",required=True);gen.add_argument("--prompt",required=True);gen.add_argument("--recipe")
    gen.add_argument("--out",default=None)
    recipe=sub.add_parser("recipe",help="Save the held-out-tested runtime intervention as JSON")
    recipe.add_argument("--run",required=True);recipe.add_argument("--out",required=True)
    doc=sub.add_parser("doctor",help="Show environment and accelerator availability")
    bench=sub.add_parser("benchmark",help="Measure actual local throughput on one prompt")
    bench.add_argument("--run",required=True);bench.add_argument("--tokens",type=int,default=32)
    bench.add_argument("--prompt",default="Explain what a database index does.")
    ev=sub.add_parser("evaluate-checkpoint",help="Fresh original vs exported checkpoint held-out/control comparison")
    ev.add_argument("--run",required=True);ev.add_argument("--model",required=True);ev.add_argument("--out",required=True)
    compare=sub.add_parser("compare",help="Compare held-out reports from two runs")
    compare.add_argument("runs",nargs=2)
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    try:
        return execute(args)
    except KeyboardInterrupt:
        print("Interrupted. Completed generation batches remain cached; rerun the same command to resume.",file=sys.stderr)
        return 130
    except (LabError,FileNotFoundError,ValueError) as e:
        print(f"ERROR: {e}",file=sys.stderr)
        return 2

def execute(a):
    if a.command=="doctor":
        import torch
        info=environment();info.update(cuda_available=torch.cuda.is_available(),cuda_version=torch.version.cuda,
            gpu=[{"name":torch.cuda.get_device_name(i),"vram_gib":torch.cuda.get_device_properties(i).total_memory/1024**3} for i in range(torch.cuda.device_count())],
            tpu_backend=False)
        print(dumps(info,True));return 0
    if a.command in {"init","validate"}:
        c=load_config(a.config,a.set);rows,audit=load_dataset(c["dataset"],c["discovery"]["min_pairs"])
        if a.command=="init":
            store=RunStore.create(a.run,c,rows,audit);print(f"Initialized {store.root}")
        print(dumps(audit,True));return 0
    if a.command=="import-legacy":
        from .legacy import audit_legacy
        result=audit_legacy(a.source,a.out,not a.no_copy)
        print(f"Imported {len(result['files'])} artifacts; audit: {Path(a.out).resolve()/'audit.md'}");return 0
    if a.command=="compare":
        values=[]
        for path in a.runs:
            s=RunStore(path);values.append({"run":str(s.root),"config_hash":s.meta["config_hash"],
                                          "evaluate":s.result("evaluate") if s.completed("evaluate") else None})
        print(dumps(values,True));return 0
    store=RunStore(a.run)
    if a.command=="status":print(dumps(store.state(),True));return 0
    if a.command=="plan" or (a.command=="run" and a.dry_run):
        from .planner import choose_next,apply_override
        results={s:store.result(s) for s in STAGES if store.completed(s)}
        order=sorted(results,key=lambda s:store.state()["stages"][s].get("ended_at",""))
        decision=apply_override(choose_next(results),order,store.config["search"]["overrides"])
        print(dumps({"next":decision.serial(),"stages":STAGES,"budget":store.config["budget"],
                    "note":"Future stages depend on measurements; this is not a fabricated complete run plan"},True));return 0
    if a.command=="unlock":
        lock=store.root/".run.lock"
        if lock.exists():
            try:
                pid=int(lock.read_text());os.kill(pid,0)
            except ProcessLookupError:lock.unlink();print("Removed stale lock");return 0
            except ValueError:raise LabError(f"Unrecognized lock; inspect manually: {lock}")
            raise LabError(f"Process {pid} is still alive; stop it before removing lock")
        print("No lock exists");return 0
    with store.lock():
        if a.command=="fork":
            c=load_config(store.root/"config.json",a.set)
            # Older snapshots predate identity_schema. Keep their identity hash
            # and reusable foundation when no identity migration was requested.
            if "identity_schema" not in store.config and not any(x.startswith("identity_schema=") for x in a.set):
                c.pop("identity_schema",None)
            child=store.fork(a.new_run,c);print(f"Forked to {child.root}");return 0
        if a.command=="reset":store.reset(a.stage);print(f"Archived {a.stage} and dependent artifacts");return 0
        if a.command=="import-directions":
            from .importer import import_directions
            print(dumps(import_directions(store,a.file,a.raw_blocks),True));return 0
        if a.command=="evaluate-checkpoint":
            from .compare import evaluate_checkpoint
            print(dumps(evaluate_checkpoint(store,a.model,a.out),True));return 0
        if a.command=="plot":
            from .plotting import plot_run
            print(dumps({"plots":plot_run(store,a.out)},True));return 0
        if a.command=="export-plan":
            from .export import make_export_plan
            value=make_export_plan(store);write_json(a.out,value)
            print(f"Wrote plan with {len(value['edits'])} edits: {a.out}");return 0
        if a.command=="export":
            from .export import export_checkpoint
            print(dumps(export_checkpoint(store,a.plan,a.out),True));return 0
        if a.command=="bundle":
            from .bundle import package_runtime_bundle
            print(dumps(package_runtime_bundle(store,a.out),True));return 0
        if a.command=="recipe":
            value=store.result("evaluate")
            if not value.get("passed"):raise LabError("No held-out-tested passing recipe; inspect evaluation, do not promote a winner blindly")
            from .directions import DirectionBundle
            bundle=DirectionBundle.load(store.stage_dir("directions"))
            write_json(a.out,{"intervention":value["selected_intervention"],"directions_fingerprint":bundle.fingerprint(),
                              "note":"Runtime hook recipe. It is not an edited checkpoint."});print(a.out);return 0
        from .pipeline import Pipeline
        pipeline=Pipeline(store)
        if a.command=="report":
            pipeline.run_stage("report",reset=store.completed("report"));print(store.stage_dir("report")/"report.html");return 0
        if a.command in {"generate","benchmark"}:
            record={"id":"interactive","split":"test","neutral":[{"role":"user","content":a.prompt}]}
            spec=None;bundle=None
            if a.command=="generate" and a.recipe:
                from .interventions import Intervention
                recipe=read_json(a.recipe);bundle=pipeline.bundle()
                if recipe.get("directions_fingerprint")!=bundle.fingerprint():raise LabError("Recipe/direction mismatch")
                raw=recipe["intervention"];spec=Intervention(**{**raw,"layers":tuple(raw["layers"])})
            # Interactive output need not have scorer-specific expected answers.
            backend=pipeline.engine.backend
            result=backend.generate([record],bundle,spec,a.tokens if a.command=="benchmark" else None)[0]
            if a.command=="generate":
                print(result["text"])
                if a.out:write_json(a.out,result)
            else:print(dumps({"device":str(backend.device),"dtype":str(backend.dtype),"tokens":result["tokens"],
                              "seconds":result["batch_seconds"],"tokens_per_second":result["tokens"]/max(result["batch_seconds"],1e-9),
                              "note":"Single measured run, not a hardware-wide performance guarantee"},True))
            return 0
        if a.command=="run":
            if a.stage:pipeline.run_stage(a.stage,a.with_deps,a.reset)
            else:
                if a.reset:raise LabError("--reset requires --stage")
                pipeline.auto(a.from_stage,a.until)
            return 0
    raise LabError("Unknown command")
