"""Single-writer run state, incremental journals, stage completion and resume."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import time
from . import __version__
from .util import LabError, BudgetReached, now, digest, file_hash, write_json, read_json, append_jsonl, read_journal, environment

STAGES = ["inspect", "baseline", "capture", "directions", "sweep", "refine", "writers", "ablate", "trace", "persistent", "evaluate", "report"]
DEPS = {"inspect": [], "baseline": ["inspect"], "capture": ["inspect"],
        "directions": ["capture"], "sweep": ["directions", "baseline"],
        "refine": ["sweep"], "writers": ["sweep"], "ablate": ["writers"], "trace": ["sweep"],
        "persistent": ["trace"], "evaluate": ["sweep"], "report": []}

# Optional analysis stages influence later selection without being mandatory prerequisites.
INVALIDATES = {"refine": {"writers", "trace", "evaluate"},
               "ablate": {"trace", "evaluate"}, "persistent": {"evaluate"},
               "trace": {"evaluate"}, "writers": {"trace", "evaluate"}}

def descendants(stage: str) -> set[str]:
    selected = {stage}
    while True:
        new = selected | {s for s, deps in DEPS.items() if any(d in selected for d in deps)}
        new |= {s for parent in selected for s in INVALIDATES.get(parent, set())}
        if new == selected: return selected | {"report"}
        selected = new

class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not (self.root / "run.json").exists():
            raise LabError(f"Not an initialized run: {self.root}; use ablab init")
        self.meta = read_json(self.root / "run.json")
        self.config = read_json(self.root / "config.json")
        if digest(self.config) != self.meta["config_hash"]:
            raise LabError("config.json was edited in place. Use ablab fork with --set overrides instead.")
        self.started = time.monotonic()
        self.prior_seconds = float(self.state().get("seconds_spent", 0.0))
        self.cache_path = self.root / "cache/generations.jsonl"
        self.cache = {r["job_key"]: r for r in read_journal(self.cache_path, ignore_tail=True)}

    @classmethod
    def create(cls, root: str | Path, config: dict, dataset: list[dict], audit: dict):
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise LabError(f"Run directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        (root / "cache").mkdir()
        write_json(root / "dataset.json", dataset)
        write_json(root / "dataset_audit.json", audit)
        config = dict(config, dataset="dataset.json")
        write_json(root / "config.json", config)
        write_json(root / "run.json", {"schema_version": 1, "tool_version": __version__,
                   "created_at": now(), "config_hash": digest(config), "dataset_hash": digest(dataset),
                   "environment": environment()})
        write_json(root / "state.json", {"stages": {}, "seconds_spent": 0.0, "generation_count": 0})
        return cls(root)

    def dataset(self) -> list[dict]:
        value = read_json(self.root / "dataset.json")
        if digest(value) != self.meta["dataset_hash"]:
            raise LabError("Snapshot dataset was changed; create a new run")
        return value

    @contextmanager
    def lock(self):
        p = self.root / ".run.lock"
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise LabError(f"Run is locked: {p}. Stop the other process; use ablab unlock for stale lock.") from e
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        try:
            self.cache = {r["job_key"]: r for r in read_journal(self.cache_path, repair_tail=True)}
            yield
        finally:
            p.unlink(missing_ok=True)

    def state(self) -> dict:
        return read_json(self.root / "state.json")

    def stage_dir(self, stage: str) -> Path:
        if stage not in STAGES: raise LabError(f"Unknown stage {stage}")
        p = self.root / "stages" / f"{STAGES.index(stage)+1:02d}_{stage}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def completed(self, stage: str, verify: bool = True) -> bool:
        value = self.state()["stages"].get(stage, {})
        if value.get("status") != "complete": return False
        if verify:
            for rel, h in value.get("artifacts", {}).items():
                p = self.root / rel
                if not p.exists() or file_hash(p) != h:
                    raise LabError(f"Artifact missing/changed: {p}; rerun stage with --reset")
        return True

    def result(self, stage: str) -> dict:
        if not self.completed(stage): raise LabError(f"Missing completed prerequisite stage: {stage}")
        return read_json(self.stage_dir(stage) / "result.json")

    def event(self, kind: str, **data):
        append_jsonl(self.root / "events.jsonl", {"time": now(), "kind": kind, **data})

    @contextmanager
    def stage(self, stage: str):
        for dep in DEPS[stage]:
            if not self.completed(dep): raise LabError(f"{stage} requires {dep}; run it first")
        state = self.state()
        state["stages"][stage] = {"status": "running", "started_at": now()}
        write_json(self.root / "state.json", state)
        start = time.monotonic()
        self.event("stage_started", stage=stage)
        try:
            yield self.stage_dir(stage)
        except BaseException as e:
            state = self.state()
            state["stages"][stage].update(status="interrupted" if isinstance(e, (KeyboardInterrupt, BudgetReached)) else "failed",
                                           error=f"{type(e).__name__}: {e}")
            state["seconds_spent"] += time.monotonic()-start
            write_json(self.root / "state.json", state)
            self.event("stage_stopped", stage=stage, error=str(e))
            raise
        else:
            p = self.stage_dir(stage)
            if not (p / "result.json").exists(): raise LabError(f"Stage {stage} did not produce result.json")
            state = self.state()
            state["stages"][stage].update(status="complete", ended_at=now(),
                artifacts={str(x.relative_to(self.root)): file_hash(x) for x in p.rglob("*") if x.is_file() and not x.name.startswith(".")})
            state["seconds_spent"] += time.monotonic()-start
            write_json(self.root / "state.json", state)
            self.event("stage_completed", stage=stage)

    def reset(self, stage: str):
        state = self.state()
        archive = self.root / "history" / str(time.time_ns())
        for s in descendants(stage):
            state["stages"].pop(s, None)
            p = self.stage_dir(s)
            if any(p.iterdir()):
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), archive / p.name)
        write_json(self.root / "state.json", state)
        self.event("reset", stage=stage, archived=str(archive))

    def check_budget(self, new_generations: int = 0):
        if self.state().get("generation_count",0) + new_generations > self.config["budget"]["max_generations"]:
            raise BudgetReached("Generation budget reached; cached jobs kept. Fork with larger budget to continue.")
        elapsed = self.prior_seconds + time.monotonic()-self.started
        if elapsed > self.config["budget"]["max_seconds"]:
            raise BudgetReached("Time budget reached between batches; completed work is cached")

    def put_generation(self, value: dict):
        key = value["job_key"]
        if key not in self.cache:
            append_jsonl(self.cache_path, value)
            self.cache[key] = value
            state=self.state(); state["generation_count"]=state.get("generation_count",0)+1
            write_json(self.root / "state.json",state)

    def fork(self, destination: str | Path, config: dict) -> "RunStore":
        """Fresh stage statuses; content-addressed generation cache is reusable where compatible."""
        dest = RunStore.create(destination, config, self.dataset(), read_json(self.root / "dataset_audit.json"))
        if self.cache_path.exists(): shutil.copy2(self.cache_path, dest.cache_path)
        old,new=self.config,dest.config
        stable_model=old["model"]==new["model"] and old.get("identity_schema",1)==new.get("identity_schema",1)
        stable_gen=old["generation"]==new["generation"]
        stable_discovery=old["discovery"]==new["discovery"]
        preserve=[]
        if stable_model: preserve.append("inspect")
        if stable_model and stable_gen and old["behavior"]==new["behavior"]:preserve.append("baseline")
        if stable_model and stable_gen:preserve.append("capture")
        if stable_model and stable_gen and stable_discovery and old["behavior"]==new["behavior"]:preserve.append("directions")
        state=dest.state()
        for name in preserve:
            if self.completed(name):
                source=self.stage_dir(name);target=dest.stage_dir(name)
                shutil.copytree(source,target,dirs_exist_ok=True)
                state["stages"][name]=self.state()["stages"][name]
        write_json(dest.root / "state.json",state)
        dest.event("forked", preserved_stages=list(state["stages"]), parent=str(self.root), parent_config_hash=self.meta["config_hash"],
                   notes="Only compatible foundational stages retained; downstream stages must be rerun, matching generation cache entries may be reused")
        return RunStore(dest.root)
