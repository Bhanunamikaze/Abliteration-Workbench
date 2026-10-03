"""Offline, escaped HTML/Markdown reports. No server, accounts or API required."""
from __future__ import annotations
import html
from pathlib import Path
from .util import read_json, write_json, atomic_text, dumps
from .store import STAGES

EXPLANATIONS = {
 "inspect":"What architecture and output sites were actually observed? Unsupported sites are not guessed.",
 "baseline":"How does the untouched model respond? The positive/negative instructions may not produce the behavior you expected.",
 "capture":"Capture raw block outputs on paired training/validation examples, before final model normalization.",
 "directions":"Estimate a direction or subspace and check held-out separation. This is not proof of causality.",
 "sweep":"Intervene at individual layers and compare behavioral scores. Random directions test generic disruption.",
 "refine":"One bounded confirmation with gentler interventions and a higher token ceiling.",
 "writers":"Compare places to inject an activation. This does not identify which matrix naturally owns the behavior.",
 "ablate":"Remove an existing component at writer outputs. Zero and negative-class coordinates are not equivalent.",
 "trace":"Compare raw residual ablation with fixed-prefix, signed-coordinate measurements. Coordinate return alone is not reconstruction.",
 "persistent":"Repeat a defined intervention over selected layers. Compare a matched random projection, not just raw output length.",
 "evaluate":"Apply the chosen intervention to held-out test/control examples. Lexical checks are not factual verification.",
}

def build_report(store, destination: Path):
    from .planner import choose_next
    results={s:store.result(s) for s in STAGES if s!="report" and store.completed(s)}
    decision=choose_next(results)
    title=f"Abliteration | {store.config['behavior']['name']}"
    final_status=decision.status
    screened=len(results.get("directions",{}).get("eligible_layers",[]))
    sampled=len(results.get("sweep",{}).get("summary",[]))
    confirmed=len(results.get("refine",results.get("sweep",{})).get("candidates",[]))
    revision=results.get("inspect",{}).get("capabilities",{}).get("resolved_revision")
    revision_note=(f"Resolved checkpoint commit: `{revision}`." if revision else
                   "Checkpoint commit was not resolved for this run; do not assume `main` is immutable.")
    lines=[f"# {title}","",f"Model: `{store.config['model']['id']}`",f"Run: `{store.root.name}`",
           "",f"**Outcome:** {final_status}",f"**Reason:** {decision.reason}",
           f"**Layer funnel:** {screened} representation-screen passes → {sampled} behaviorally sampled → {confirmed} refined steering candidates.",
           revision_note,"",
           "Automatic research stages do not edit model parameters. A separate explicit export command writes a copied checkpoint.",
           "Results describe this dataset, metric, precision, token scope, and the recorded model identity. They are not universal layer labels.","",
           "## Stage record",""]
    sections=[]
    figures=[]
    try:
        import matplotlib
        from .plotting import plot_run
        figures=plot_run(store,destination/"figures")
    except ImportError:
        pass
    figure_html="".join(f'<figure><img style="max-width:100%" src="figures/{html.escape(Path(f).name)}" alt="{html.escape(Path(f).stem)}"></figure>' for f in figures)
    if figures:
        lines += ["## Measurement plots", ""] + [f"![{Path(f).stem}](figures/{Path(f).name})" for f in figures] + [""]
    for name in STAGES:
        if name not in results:continue
        value=results[name]
        lines += [f"### {name}",EXPLANATIONS.get(name,""),"","```json",dumps(value,True),"```",""]
        sections.append(f"<details><summary>{html.escape(name)}: {html.escape(EXPLANATIONS.get(name,''))}</summary><pre>{html.escape(dumps(value,True))}</pre></details>")
    lines += ["## Interpretation limits","",
              "A larger raw activation margin is not automatically better separability; use standardized held-out metrics.",
              "Signed deltas and paired errors matter. Ratios with small denominators are unstable.",
              "Generation limits censor lengths. Mean per-example percentages and ratios of means are different statistics.",
              "No-op/cache checks test software plumbing, not semantic quality. Random directions are controls, not guaranteed inert features.",
              "A selected winner can fail held-out/control evaluation. No export is triggered by candidate rank alone."]
    atomic_text(destination/"report.md","\n".join(lines)+"\n")
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:48px auto;padding:0 24px}}h1{{font-size:32px}}summary{{cursor:pointer;font-weight:600;padding:14px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;padding:18px;background:#f5f5f5;border-radius:8px}}details{{border-bottom:1px solid #ddd}}.note{{padding:18px;background:#eef4f7}}</style>
<h1>{html.escape(title)}</h1><p>Model: <code>{html.escape(store.config['model']['id'])}</code></p>
<p class="note">Outcome: <strong>{html.escape(final_status)}</strong>. {html.escape(decision.reason)}<br>
Layer funnel: {screened} representation-screen passes → {sampled} behaviorally sampled → {confirmed} refined steering candidates.<br>{html.escape(revision_note)}</p>
<p>This is a local research report. No scientific claim follows from a heuristic label alone. Runtime stages leave checkpoint weights unchanged.</p>
{figure_html}{''.join(sections)}<h2>Review the actual responses</h2><p>Each completed stage has outputs.json and results.csv. Compare answer content, stopping reasons, objective scores, random controls and held-out performance before interpreting any effect.</p></html>'''
    atomic_text(destination/"report.html",doc)
    return ["report.md","report.html"]
