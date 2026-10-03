"""Offline, escaped HTML/Markdown reports. No server, accounts or API required."""
from __future__ import annotations
import html
from pathlib import Path
from .util import read_journal, atomic_text, dumps
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

_COLUMNS = ("candidate", "stage", "layers", "layer count", "region width", "site", "operation", "reference", "strength", "generation cap",
            "baseline", "intervention", "gain", "win rate", "CI95", "random control",
            "cap rate", "uncensored pairs", "evidence", "confirmation", "evaluation")


def _number(value):
    return "—" if value is None else f"{value:.3g}" if isinstance(value, (int, float)) else str(value)


def _candidate_rows(results):
    """Flatten stored summaries without upgrading legacy results to new evidence states."""
    selected = results.get("evaluate", {}).get("selected_intervention") or {}
    evaluated_stage = results.get("evaluate", {}).get("candidate_source_stage")
    rows = []
    for stage in ("sweep", "refine", "ablate", "trace", "persistent"):
        source = results.get(stage, {})
        entries = source.get("summary", [])
        if stage in {"sweep", "refine"}:
            chosen_layers = {r.get("layer") for r in source.get("exploratory_candidates", [])}
            entries = [r for r in entries if r.get("layer") in chosen_layers]
        for item in entries:
            intervention = item.get("intervention", {})
            layers = intervention.get("layers", [item["layer"]] if "layer" in item else [])
            if not isinstance(layers, list):
                layers = list(layers)
            status = item.get("evidence_status")
            if not status:
                if stage in {"sweep", "refine"}:
                    status = "steering candidate" if item.get("eligible") else "exploratory steering"
                else:
                    status = "legacy provisional" if item.get("provisional_pass") else "legacy unclassified"
            confirmation = item.get("confirmation") or {}
            if isinstance(confirmation, dict):
                confirmation_text = (f"attempted at {confirmation.get('final_max_new_tokens', '?')} tokens; "
                                     f"{confirmation.get('final_status', status)}"
                                     if confirmation.get("attempted") else confirmation.get("reason", "none"))
            else:
                confirmation_text = str(confirmation)
            match = bool(selected and (evaluated_stage is None or evaluated_stage == stage)
                         and all(selected.get(key) == intervention.get(key)
                                          for key in ("layers", "site", "operation", "reference", "strength")))
            evaluation = (results.get("evaluate", {}).get("evaluation_status")
                          or ("passed" if results.get("evaluate", {}).get("passed") else "failed")) if match else "—"
            ci = (f"[{_number(item.get('ci95_low'))}, {_number(item.get('ci95_high'))}]"
                  if item.get("ci95_low") is not None else "—")
            rows.append((item.get("label", f"L{item.get('layer', '?')}"), stage,
                         ", ".join(map(str, layers)) or "—",
                         _number(item.get("layer_count", len(layers) if layers else None)),
                         _number(item.get("region_width", max(layers)-min(layers)+1 if layers else None)),
                         intervention.get("site", "residual"),
                         intervention.get("operation", "steer"), intervention.get("reference", "zero"),
                         _number(intervention.get("strength", "—")), _number(item.get("generation_cap", item.get("max_new_tokens"))),
                         _number(item.get("mean_baseline")),
                         _number(item.get("mean_intervention")),
                         _number(item.get("gain_fraction_of_baseline_mean", item.get("effect_fraction"))),
                         _number(item.get("win_rate")), ci,
                         _number(item.get("random_control_max_gain_fraction", item.get("random_control_max_effect"))),
                         _number(item.get("cap_rate")), _number(item.get("uncensored_pair_fraction")),
                         status, confirmation_text, evaluation))
    return rows


def _markdown_table(rows):
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(_COLUMNS) + " |",
            "| " + " | ".join("---" for _ in _COLUMNS) + " |"] + [
                "| " + " | ".join(safe(value) for value in row) + " |" for row in rows]


def _html_table(rows):
    head = "".join(f"<th>{html.escape(col)}</th>" for col in _COLUMNS)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
                   for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

def build_report(store, destination: Path):
    from .planner import choose_next
    results={s:store.result(s) for s in STAGES if s!="report" and store.completed(s)}
    decision=choose_next(results, mode=store.config.get("search", {}).get("mode", "strict"))
    title=f"Abliteration | {store.config['behavior']['name']}"
    final_status=decision.status
    screened=len(results.get("directions",{}).get("eligible_layers",[]))
    sampled=len(results.get("sweep",{}).get("summary",[]))
    confirmed=len(results.get("refine",results.get("sweep",{})).get("candidates",[]))
    revision=results.get("inspect",{}).get("capabilities",{}).get("resolved_revision")
    revision_note=(f"Resolved checkpoint commit: `{revision}`." if revision else
                   "Checkpoint commit was not resolved for this run; do not assume `main` is immutable.")
    mode=store.config.get("search", {}).get("mode", "strict")
    rows=_candidate_rows(results)
    decisions=[r for r in read_journal(store.root/"events.jsonl", ignore_tail=True) if r.get("kind")=="planner_decision"]
    selection=results.get("evaluate", {})
    refs={r[_COLUMNS.index("reference")] for r in rows
          if r[_COLUMNS.index("stage")] in {"ablate", "trace", "persistent"}
          and r[_COLUMNS.index("site")] == "residual"}
    ref_note=("Residual zero and negative-centroid references were both measured in this run."
              if {"zero", "negative"} <= refs else
              "Residual references measured: " + (", ".join(sorted(refs)) if refs else "none") + ".")
    censored=[r for r in rows if r[_COLUMNS.index("evidence")] == "promising_censored"]
    strong_censored=any(item.get("evidence_status")=="promising_censored" and item.get("behaviorally_promising", True)
                        for stage in ("ablate", "trace", "persistent")
                        for item in results.get(stage, {}).get("summary", []))
    compared=any(results.get(stage, {}).get("reference_comparison_performed")
                 for stage in ("ablate", "trace", "persistent"))
    omitted=[(stage, label, reason) for stage in ("ablate", "trace", "persistent")
             for label, reason in (results.get(stage, {}).get("omitted_regions") or {}).items()]
    removal_signal=results.get("trace", {}).get("representation_removal_behavior_weak")
    lines=[f"# {title}","",f"Model: `{store.config['model']['id']}`",f"Run: `{store.root.name}`",
           "",f"**Outcome:** {final_status}",f"**Reason:** {decision.reason}",
           f"**Planner mode:** `{mode}`.",
           f"**Layer funnel:** {screened} representation-screen passes → {sampled} behaviorally sampled → {confirmed} refined steering candidates.",
           revision_note,"",
           "Automatic research stages do not edit model parameters. A separate explicit export command writes a copied checkpoint.",
           "Results describe this dataset, metric, precision, token scope, and the recorded model identity. They are not universal layer labels.","",
           "## Candidate and evidence summary","",
           "Representation separation is a screening result. Steering measures behavioral sensitivity; residual ablation tests a geometric necessity hypothesis. Multi-layer interventions measure a distributed intervention, not single-layer attribution.",
           "",ref_note, f"Automatic residual reference comparison performed: **{'yes' if compared else 'no'}**.",""]
    if omitted:
        lines += ["Automatic regions omitted by span bounds: " + "; ".join(f"{stage}/{label}: {reason}" for stage,label,reason in omitted),""]
    if removal_signal:
        lines += ["Trace diagnostic: the measured coordinate was removed while the target behavior changed weakly. This does not establish behavioral necessity; compare the calibrated negative-centroid target where available.",""]
    if strong_censored:
        lines += ["**Strong behavioral effect detected, but generation censoring prevents final promotion.**",
                  "The target metric changed on these candidates, but response quality and termination remain unconfirmed. Inspect the higher-cap confirmation and outputs before interpreting the effect.",""]
    elif censored:
        lines += ["**Censored length evidence requires uncensored confirmation.**",
                  "A generation ceiling directly limits token or word scores. The capped score difference is recorded, but it cannot establish the natural response-length effect.",""]
    lines += _markdown_table(rows) if rows else ["No candidate summaries were recorded."]
    lines += ["","## Planner and held-out selection","",f"Current decision: {decision.reason}",""]
    if decisions:
        lines += ["Recorded route:",""] + [f"- {r.get('next_stage') or 'stop'}: {r.get('reason','')}" for r in decisions] + [""]
    if selection:
        lines += [f"Selected from: `{selection.get('candidate_source_stage', 'legacy selection not recorded')}`",
                  f"Selection reason: {selection.get('selection_reason', 'Legacy run did not record a selection reason.')}",
                  f"Selected evidence status: `{selection.get('candidate_evidence_status', 'legacy unknown')}`",
                  f"Held-out evaluation: `{selection.get('evaluation_status', 'passed' if selection.get('passed') else 'failed or inconclusive')}`",""]
    lines += ["## Stage record",""]
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
    route_html="<ul>"+"".join(f"<li>{html.escape(str(r.get('next_stage') or 'stop'))}: {html.escape(r.get('reason',''))}</li>" for r in decisions)+"</ul>"
    selected_html=(f"<p>Selected from: <code>{html.escape(selection.get('candidate_source_stage', 'legacy unknown'))}</code>. "
                   f"{html.escape(selection.get('selection_reason', 'Legacy run did not record a selection reason.'))} "
                   f"Evidence: <code>{html.escape(selection.get('candidate_evidence_status', 'legacy unknown'))}</code>. "
                   f"Held-out status: <code>{html.escape(selection.get('evaluation_status', 'passed' if selection.get('passed') else 'failed or inconclusive'))}</code>.</p>"
                   if selection else "")
    censored_html=("<p class=\"censored\"><strong>Strong behavioral effect detected, but generation censoring prevents final promotion.</strong> "
                   "Review higher-cap confirmation and response quality.</p>" if strong_censored else
                   "<p class=\"censored\"><strong>Censored length evidence requires uncensored confirmation.</strong> "
                   "The generation ceiling directly limits token or word scores.</p>" if censored else "")
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:48px auto;padding:0 24px}}h1{{font-size:32px}}summary{{cursor:pointer;font-weight:600;padding:14px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;padding:18px;background:#f5f5f5;border-radius:8px}}details{{border-bottom:1px solid #ddd}}.note{{padding:18px;background:#eef4f7}}.censored{{padding:14px;background:#fff0d8}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px 10px;border:1px solid #ddd;text-align:left;white-space:nowrap}}</style>
<h1>{html.escape(title)}</h1><p>Model: <code>{html.escape(store.config['model']['id'])}</code></p>
<p class="note">Outcome: <strong>{html.escape(final_status)}</strong>. {html.escape(decision.reason)}<br>
Planner mode: {html.escape(mode)}.<br>Layer funnel: {screened} representation-screen passes → {sampled} behaviorally sampled → {confirmed} refined steering candidates.<br>{html.escape(revision_note)}</p>
<p>This is a local research report. No scientific claim follows from a heuristic label alone. Runtime stages leave checkpoint weights unchanged.</p>
<h2>Candidate and evidence summary</h2><p>Representation separation is a screen. Steering measures sensitivity; residual ablation tests a geometric necessity hypothesis. Multi-layer effects retain their full region provenance.</p><p>{html.escape(ref_note)} Automatic comparison: {'yes' if compared else 'no'}.</p>{censored_html}
{_html_table(rows) if rows else '<p>No candidate summaries were recorded.</p>'}
<h2>Planner and held-out selection</h2><p>{html.escape(decision.reason)}</p>{route_html}{selected_html}
{figure_html}<h2>Stage record</h2>{''.join(sections)}<h2>Review the actual responses</h2><p>Each completed stage has outputs.json and results.csv. Compare answer content, stopping reasons, objective scores, random controls and held-out performance before interpreting any effect.</p></html>'''
    atomic_text(destination/"report.html",doc)
    return ["report.md","report.html"]
