"""Offline plots of recorded measurements; no model load and no synthesized results."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from statistics import mean
from .util import LabError, read_json, write_json


def plot_run(store, destination=None) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise LabError("Install the optional plots extra: python -m pip install -e '.[plots]'") from error
    output = Path(destination) if destination else store.root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(name, xlabel, ylabel, title):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        path = output / name
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(str(path))

    if store.completed("directions"):
        metrics = store.result("directions").get("metrics", [])
        if metrics:
            plt.figure(figsize=(10, 5))
            metrics = sorted(metrics, key=lambda row: row["layer"])
            plt.plot([r["layer"] for r in metrics], [r["standardized_separation"] for r in metrics], marker="o")
            save("heldout_separation.png", "Raw transformer block index", "Held-out standardized separation",
                 "Instruction-contrast separation, not causal control")

    if store.completed("evaluate"):
        file = store.stage_dir("evaluate") / "outputs.json"
        if file.exists():
            evaluated = read_json(file)
            for split, name in (("test", "heldout_response.png"), ("control", "control_response.png")):
                rows = [row for row in evaluated if row.get("split") == split]
                if not rows:
                    continue
                plt.figure(figsize=(9, max(3.5, .48 * len(rows) + 1.6)))
                for index, row in enumerate(rows):
                    before, after = row["baseline_score"], row["score"]
                    plt.plot([before, after], [index + .08, index - .08], color="#9aa7b2", linewidth=2)
                    plt.scatter(before, index + .08, color="#2563a6", s=50,
                                label="Original" if index == 0 else None, zorder=3)
                    plt.scatter(after, index - .08, color="#d97728", s=50,
                                label="Intervention" if index == 0 else None, zorder=3)
                plt.yticks(range(len(rows)), [str(row["id"])[:28] for row in rows])
                plt.gca().invert_yaxis()
                plt.legend(loc="best")
                save(name, f"{store.config['behavior']['metric']} score", "Example ID",
                     f"{split.title()}: paired original and intervention scores")

    for stage in ("sweep", "refine"):
        if not store.completed(stage):
            continue
        file = store.stage_dir(stage) / "outputs.json"
        if not file.exists():
            continue
        rows = read_json(file)
        groups = defaultdict(lambda: defaultdict(list))
        baselines = {}
        for row in rows:
            if row.get("random_control") or row.get("intervention", {}).get("control_seed") is not None:
                continue
            layer = row["layer"]
            groups[layer][row["strength"]].append(row["score"])
            baselines[layer] = row["baseline_score"]  # overwritten below with all paired values
        for layer in groups:
            neutral = [r["baseline_score"] for r in rows if r.get("layer") == layer and not r.get("random_control")]
            groups[layer][0.0] = neutral
        if groups:
            plt.figure(figsize=(11, 6))
            for layer, series in sorted(groups.items()):
                x = sorted(series)
                plt.plot(x, [mean(series[v]) for v in x], marker="o", label=f"Block {layer}")
            plt.legend(ncol=2)
            save(f"{stage}_response.png", "Relative steering strength", f"Mean {store.config['behavior']['metric']} score",
                 f"{stage.title()}: observed scores (capped generations remain censored)")

    for stage in ("ablate", "trace", "persistent"):
        if not store.completed(stage):
            continue
        entries = store.result(stage).get("summary", [])
        groups = defaultdict(list)
        for row in entries:
            groups[row["label"]].append(row)
        if groups:
            plt.figure(figsize=(11, 6))
            for label, series in sorted(groups.items()):
                series = sorted(series, key=lambda r: r["intervention"]["strength"])
                x = [0.0] + [r["intervention"]["strength"] for r in series]
                y = [series[0]["mean_baseline"]] + [r["mean_intervention"] for r in series]
                plt.plot(x, y, marker="o", label=label)
            plt.legend(ncol=2)
            save(f"{stage}_response.png", "Fractional projection removal", f"Mean {store.config['behavior']['metric']} score",
                 f"{stage.title()}: behavior measurement, not quality certification")

    if store.completed("trace"):
        rows = store.result("trace").get("projection_summary", [])
        for prefix in sorted({r["prefix_tokens"] for r in rows}):
            for field, ylabel, suffix in (
                ("ratio_of_mean_abs", "Ratio of mean absolute coordinates", "magnitude_ratio"),
                ("mean_signed_delta", "Mean signed coordinate change from baseline", "signed_delta"),
                ("mean_abs_paired_error", "Mean absolute paired coordinate error", "paired_error"),
            ):
                plt.figure(figsize=(11, 6))
                plotted = False
                for source in sorted({r["source_layer"] for r in rows}):
                    group = sorted([r for r in rows if r["source_layer"] == source and r["prefix_tokens"] == prefix
                                    and r.get(field) is not None], key=lambda r: r["observed_layer"])
                    if not group:
                        continue
                    plt.plot([r["observed_layer"] for r in group], [r[field] for r in group], marker="o", label=f"Intervene at {source}")
                    plotted = True
                if plotted:
                    plt.axhline(1.0 if field == "ratio_of_mean_abs" else 0.0, linestyle="--", linewidth=1)
                    plt.legend()
                    save(f"trace_prefix{prefix}_{suffix}.png", "Observed raw block index", ylabel,
                         f"Matched baseline prefix ({prefix} response tokens): coordinate diagnostic")
                else:
                    plt.close()
    write_json(output / "plot_manifest.json", {"files": [Path(p).name for p in paths],
              "note": "Derived only from stored measurements; connecting lines are visual interpolation. Ratios do not prove representation reconstruction."})
    return paths
