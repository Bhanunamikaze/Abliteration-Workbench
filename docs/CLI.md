# CLI and experiment control

Use `abliteration --help` or `abliteration COMMAND --help`. The original `ablab` command and `python -m ablationlab` remain available for existing runs.

## Commands

| Command | Purpose | Loads model? |
|---|---|---|
| `validate --config FILE` | Validate config, paired splits, and dataset format | No |
| `init --config FILE --run DIR` | Create an immutable run snapshot | No |
| `doctor` | Report Python, dependencies, CUDA and VRAM | No |
| `status --run DIR` | Stage states and budget counters | No |
| `plan --run DIR` | Next automatic action and reason | No |
| `run --run DIR` | Execute the bounded automatic pipeline | As needed |
| `run --stage NAME --with-deps` | Execute a selected stage and required prerequisites | As needed |
| `run --from-stage NAME --until NAME` | Begin explicitly, then continue with planner | As needed |
| `run --dry-run` | Preview next planner decision only | No |
| `fork --run DIR --new-run DIR --set KEY=VALUE` | Branch configuration and preserve compatible upstream work | No |
| `reset --run DIR --stage NAME` | Archive stage and invalidate affected downstream artifacts | No |
| `report --run DIR` | Render recorded evidence to local HTML/Markdown | No model |
| `plot --run DIR [--out DIR]` | Plot recorded measurements | No |
| `import-legacy --source DIR --out DIR` | Preserve/audit old 05–09 CSV/JSON files | No |
| `import-directions --run DIR --file FILE.pt` | Load actual old vectors with explicit provenance warnings | Yes, inspect only |
| `recipe --run DIR --out FILE` | Save a passing held-out runtime intervention | No |
| `generate --run DIR --prompt TEXT [--recipe FILE]` | Generate without or with runtime hooks | Yes |
| `benchmark --run DIR --tokens N` | Time this machine, not an assumed GPU benchmark | Yes |
| `export-plan --run DIR --out FILE` | Draft eligible writer-projection plan; may contain zero edits | No |
| `export --run DIR --plan FILE --out DIR` | Apply projection to a copied floating checkpoint | Yes |
| `bundle --run DIR --out DIR` | Package a passing runtime intervention, unchanged model files, directions, and runner; replay held-out outputs | Yes |
| `replication-dataset --outputs DIR --out FILE` | Recover prompts only from historical baseline/intervention outputs | No |
| `replicate --run SOURCE --dataset FILE --out DIR` | Resume or create a frozen intervention replication | Yes |
| `replication-status --run DIR` | Verify and display replication state and result | No |
| `replication-report --run DIR` | Verify and print the completed replication report path | No |
| `recipe --replication DIR --out FILE` | Save a recipe authorized by a passing replication | No |
| `bundle --replication DIR --out DIR` | Package and replay a passing replication | Yes |
| `evaluate-checkpoint --run DIR --model ID_OR_PATH --out DIR` | Fresh original/candidate held-out and control responses | One model at a time |
| `compare RUN1 RUN2` | Display two runs' stored evaluation summaries | No |
| `unlock --run DIR` | Remove stale run lock only after its process has exited | No |

Options shown without `--run` in abbreviated rows still need `--run DIR`.

## Frozen-intervention replication

`replicate` starts from the source run's completed `evaluate.selected_intervention`. It copies and fingerprints the existing direction bundle, snapshots a new test/control-only dataset, and records the source config and dataset hashes, model identity, scorer provenance, source evaluation hash, generation settings, and tool version. It does not invoke capture, direction estimation, layer search, or other discovery stages. The new prompts must not duplicate source prompts or each other. Rerunning the same command resumes from the content-addressed generation cache; a changed dataset or source is rejected.

```bash
abliteration replicate --run runs/discovery \
  --dataset fresh-test-control.json --out runs/external-replication
abliteration replication-status --run runs/external-replication
abliteration replication-report --run runs/external-replication
```

The initial ceiling is the source evaluation ceiling. If a promising effect fails the unchanged `search.max_cap_rate` gate, replication saves the initial outputs and summary under `attempts/initial/`, then automatically retries matched baseline and intervention arms at the larger of twice the ceiling or `generation.confirm_max_new_tokens`. Use `--retry-max-new-tokens N` at creation to set a different higher ceiling when the model context requires it. Both attempts retain their own cap rates; only the final attempt can pass. `--max-generations` and `--max-seconds` set immutable budgets at creation; the generation budget must cover both arms and one retry. Interruptions preserve finished batches. A retry that still caps remains `promising_censored` and cannot promote.

Binary observed scores use favorable/unfavorable paired transitions, a bootstrap paired-gain interval, and an exact two-sided paired binomial p-value. Promotion requires positive effect and CI, sufficient samples and discordant pairs, favorable fraction, low test and control regression, control drift, cap and repetition/content quality, source specificity and quality confirmation, and acceptable reference NLL when enabled. Binary promotion does not use the fraction of all examples that changed. Continuous scores retain the existing paired win-rate gate. The source config sets `evaluation.binary_min_samples`, `binary_min_discordant`, `binary_min_favorable_fraction`, `binary_max_regression_fraction`, `binary_max_control_regression_fraction`, and `max_scorer_review_fraction` before replication is created.

`summary.json` contains provenance, `test` and `control` paired statistics and review counts, `attempts`, `promotion_gates`, `failed_gates`, `status`, and `passed`. Four JSON output files, `results.csv`, `manual_review.csv`, fixed-reference NLL, and Markdown/HTML reports preserve the review trail. Review flags do not turn classifier scores into confirmed semantic labels. A passing result may authorize `recipe --replication` or `bundle --replication`; bundle creation replays the saved intervention outputs at the final ceiling. The source run and its `evaluate/result.json` remain unchanged.

For older external files with `test_baseline.json`, `test_intervention.json`, `control_baseline.json`, and `control_intervention.json`, recover only their prompts and labels:

```bash
abliteration replication-dataset --outputs runs/llama32-refusal-replication-targeted \
  --out runs/llama32-refusal-replication-prompts.json
abliteration replicate --run runs/llama32-refusal-auto-v2 \
  --dataset runs/llama32-refusal-replication-prompts.json \
  --out runs/llama32-refusal-replication
```

The old scores are useful for checking metric interpretation, but they do not establish a first-class replication. The second command reruns generation and scoring using the frozen source intervention.

For the recorded Llama study, those commands have already been run; choose unused output paths for a new attempt. Its larger benchmark set overlapped earlier baseline prompts, and its first-class run reused the temporary evaluator's prompt set. The recorded `validated_replication` status therefore describes a **same-environment frozen-intervention reproduction**, not an independent external-lab or fully unseen test. Its binary policy was refined after earlier results. The [research record](research/llama32-refusal-2026-10-04.md) documents exposure and an unflagged false non-refusal label; `max_scorer_review_fraction: 1.0` passing does not mean human review occurred.

## Configuration knobs

The complete resolved schema is in `ablationlab/config.py`; every run stores its resolved `config.json`.

- `model`: checkpoint, pinned `revision`, dtype/device, attention implementation, optional bitsandbytes quantization, explicit offload, adapter path overrides, or custom backend factory.
- `behavior`: name, actual scorer, intended increase/decrease, regex/custom scorer configuration, normalization floor.
- `generation`: batch, output/input ceilings, seed, `scope=last|all`, `phase=prefill|decode|both`, refinement ceiling, and `confirm_max_new_tokens` for one bounded retry of a promising capped result. Baseline and intervention use the same ceiling in each comparison.
- `discovery`: mean contrast or mean plus paired-residual SVD, rank, minimum paired examples.
- `search`: automatic or explicit layers, top-k, symmetric steering grid, ablation strengths, residual `reference=zero|negative|auto`, randomized controls, evidence thresholds, permitted persistent regions and route overrides. `mode=strict` preserves conservative routing; `mode=explore` continues bounded writer, residual ablation, trace, persistent, and evaluation diagnostics after weak steering. These diagnostics retain their evidence status and do not automatically promote a winner.
- `search.persistent_max_span_layers=8` limits automatically generated contiguous spans. `persistent_cluster_gap=2` groups nearby candidate layers. Sparse peaks remain an exact layer set, while an oversized contiguous span requires an explicit `custom_regions` entry. Each candidate records its modified layer count and region width.
- `search.max_cap_confirmations=1` limits higher-ceiling retries of a promising capped candidate; values from 0 to 2 are accepted. `0` disables the retry. A capped result can remain `promising_censored` even after a retry and still requires quality review before validation.
- `search.random_controls=0` skips specificity measurement; such effects stay exploratory and cannot become `validated_candidate` automatically.
- `search.custom_regions`: optional named layer lists for an explicit persistent diagnostic. These are manual hypotheses, not automatically selected winners. For example: `--set 'search.regions=[]' --set 'search.custom_regions=[{"label":"legacy_set","layers":[13,16,18,20,22]}]'`.
- `evaluation`: fixed-reference token count and allowed control NLL increase. `baseline_include_test=false` keeps test prompts out of the baseline stage by default; explicitly setting it to `true` also retains vanilla test responses in that stage. Test scores do not enter the baseline validation summary. This exposes test responses, so later adaptive work needs a fresh held-out set.
- `trace`: matched baseline-prefix lengths, denominator floor for coordinate diagnostics.
- `budget`: generation and wall-clock ceilings, checked between batches.

Use JSON values in shell overrides. For example:

```bash
abliteration init --config examples/verbosity.json --run runs/custom \
  --set 'search.strengths=[-0.1,0.1]' \
  --set 'search.ablations=[0.25,0.5,1.0]' \
  --set 'generation.scope="all"' \
  --set 'generation.phase="both"' \
  --set 'model.dtype="fp16"'
```

For an end-to-end exploratory run, put these values in the source config or pass them to `init`:

```json
{
  "search": {
    "mode": "explore",
    "reference": "auto",
    "persistent_max_span_layers": 8,
    "persistent_cluster_gap": 2,
    "max_cap_confirmations": 1
  },
  "generation": {"confirm_max_new_tokens": 1024}
}
```

```bash
abliteration init --config experiment.json --run runs/experiment
abliteration run --run runs/experiment
```

`reference=auto` compares zero projection with a measured negative-class centroid where residual calibration exists. Writer outputs always use zero reference. An imported, uncalibrated direction cannot supply a negative centroid. A geometric coordinate removal alone does not prove that the coordinate is behaviorally necessary.

`scope=all` edits every currently processed nonpadding token. During cached decoding the current sequence ordinarily contains one token. `phase=decode` deliberately leaves prompt prefill untouched. The chosen scope is part of cache/provenance, not a hidden optimization.

## Switching direction without discarding evidence

```bash
abliteration fork --run runs/custom --new-run runs/custom-centered \
  --set 'search.reference="negative"'
abliteration run --run runs/custom-centered --stage trace --with-deps
```

The negative class centroid is calibrated for residual outputs, not the individual writers. Writer ablation therefore uses zero. Imported old vectors cannot supply a missing centroid.

Manual routing example:

```bash
abliteration fork --run runs/custom --new-run runs/residual-first \
  --set 'search.overrides={"sweep":"trace","trace":"persistent"}'
abliteration run --run runs/residual-first
```

An override does not waive dependencies or relabel a weak result as confirmed. `"stop"` stops after the specified stage. The planner has a finite execution bound; it does not repeatedly amplify perturbations until some metric improves.

## Troubleshooting

**CUDA false:** use a CUDA-enabled PyTorch installation and a GPU runtime. A Colab TPU will not be used by this CUDA backend. CPU runs require explicit `model.device=cpu` or `auto`.

**403 gated model:** obtain the model provider's access using the same Hugging Face account/token. Application-local operation does not override model distribution permissions.

**OOM:** generation batches are halved. At batch size one, reduce model/context or deliberately enable quantization/offload. Model-loading OOM cannot be fixed by batching. Capture/trace loading errors are surfaced rather than silently falling back to CPU.

**No eligible direction:** inspect paired instructions, captured site and standardized validation metrics. A toy fixture is deliberately untrained. Weak real data may need more pairs, better matching, or a different observable. Automatic analysis may correctly stop.

**Lots of length caps:** observed counts are censored. A strong discrete-metric effect is retained as `promising_censored` and retried within the configured ceiling; termination quality remains unconfirmed if it still caps. Length metrics need uncensored confirmation for a behavioral claim. Do not treat `384` as the model's natural completion length.

**Automatic run does not visit every stage:** it is conditional. Set `search.mode=explore` to continue bounded diagnostics after weak steering, or use an explicit stage or manual route for a particular hypothesis. `strict` remains the default.

**Imported vectors stop automatic progression:** old 03 files lack full revision/precision/capture metadata and centroids. Recapture in a fresh run for full calibration. Manual exploratory use is still available; the legacy final layer is excluded unless its raw-block origin is known.

**Empty export plan:** the winning runtime experiment may be residual-only, last-token-only, noisy, or fail held-out validation. None makes a corresponding static writer edit automatically justified.

**Unsupported MoE export:** aggregate runtime hooks and expert weight layouts are separate capabilities. Fused/stacked expert banks need an explicit tested exporter. Do not rename tensor paths to make checks pass.

**Interrupted run:** rerun the identical command. A malformed partial final cache record can be repaired; corruption in the middle is not silently ignored. Reset dependent stages after changing experiment inputs through a fork.

**Changed results across machines/batches:** dtype, kernels, padding, model revision and greedy near-ties can alter generation. Runs retain identity and batch composition; no-op checks and identical baselines help detect differences, but bitwise cross-hardware equivalence is not promised.
