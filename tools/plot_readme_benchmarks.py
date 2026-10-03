"""Render README verbosity figures from tracked Qwen measurements."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE = "#2563a6"
HOOK = "#d97728"


def verbosity_plot(data: dict, destination: Path) -> None:
    rows = data["verbosity"]["test"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    positions = list(reversed(range(len(rows))))
    for index, (row, y) in enumerate(zip(rows, positions)):
        ax.plot([row["baseline_tokens"], row["hook_tokens"]], [y, y], color="#9aa7b2", lw=2, zorder=1)
        ax.scatter(row["baseline_tokens"], y, color=BASELINE, s=65, marker="^" if row["baseline_capped"] else "o",
                   label="Original" if index == 0 else None, zorder=3)
        ax.scatter(row["hook_tokens"], y, color=HOOK, s=65,
                   label="Verbosity hook" if index == 0 else None, zorder=3)
    ax.set_yticks(positions, [f"Test {i + 1}" for i in range(len(rows))])
    ax.set_xlabel("Generated tokens (lower is shorter)")
    ax.set_xlim(0, 410)
    ax.axvline(384, ls="--", color="#89939e", lw=1)
    ax.text(383, positions[0] + .42, "384-token cap", ha="right", fontsize=9, color="#52606d")
    ax.grid(axis="x", color="#e5e8eb")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.56, .11), frameon=False, ncol=2)
    ax.set_title("Qwen verbosity: paired held-out responses", loc="left", fontweight="bold", pad=13)
    ax.text(0, 1.02, "Mean 207 → 120 tokens (42.4% shorter); 8 test prompts", transform=ax.transAxes,
            fontsize=10, color="#3f4b57")
    fig.text(.11, .015, "One original response reached the token cap. Four control prompts showed no absolute drift.",
             fontsize=8.5, color="#52606d")
    fig.subplots_adjust(left=.14, right=.98, top=.81, bottom=.31)
    fig.savefig(destination, dpi=190, facecolor="white")
    plt.close(fig)


def layer_screen_plot(source: Path, destination: Path) -> None:
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    layers = [int(row["layer"]) for row in rows]
    separation = [float(row["standardized_separation"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(layers, separation, marker="o", color=BASELINE, linewidth=2)
    ax.set(xlabel="Raw transformer block index", ylabel="Held-out standardized separation")
    ax.set_title("Qwen instruction contrast across 28 blocks", loc="left", fontweight="bold", pad=13)
    ax.grid(color="#e5e8eb")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.12, .015, "Representation screen only; behavioral tests selected the five-layer runtime hook.",
             fontsize=8.5, color="#52606d")
    fig.subplots_adjust(left=.12, right=.98, top=.85, bottom=.23)
    fig.savefig(destination, dpi=190, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="validation/benchmark-data.json")
    parser.add_argument("--layer-data", default="validation/qwen-verbosity-layer-screen.csv")
    parser.add_argument("--out-dir", default="docs/assets")
    args = parser.parse_args()
    data = json.loads(Path(args.data).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    verbosity_plot(data, out / "verbosity-qwen.png")
    layer_screen_plot(Path(args.layer_data), out / "qwen-heldout-separation.png")
    print(out / "verbosity-qwen.png")
    print(out / "qwen-heldout-separation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
