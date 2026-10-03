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
| `evaluate-checkpoint --run DIR --model ID_OR_PATH --out DIR` | Fresh original/candidate held-out and control responses | One model at a time |
| `compare RUN1 RUN2` | Display two runs' stored evaluation summaries | No |
| `unlock --run DIR` | Remove stale run lock only after its process has exited | No |

Options shown without `--run` in abbreviated rows still need `--run DIR`.

## Configuration knobs

The complete resolved schema is in `ablationlab/config.py`; every run stores its resolved `config.json`.

- `model`: checkpoint, pinned `revision`, dtype/device, attention implementation, optional bitsandbytes quantization, explicit offload, adapter path overrides, or custom backend factory.
- `behavior`: name, actual scorer, intended increase/decrease, regex/custom scorer configuration, normalization floor.
- `generation`: batch, output/input ceilings, seed, `scope=last|all`, `phase=prefill|decode|both`, refinement ceiling.
- `discovery`: mean contrast or mean plus paired-residual SVD, rank, minimum paired examples.
- `search`: automatic or explicit layers, top-k, symmetric steering grid, ablation strengths, zero/negative centroid reference, randomized controls, evidence thresholds, permitted persistent regions and route overrides.
- `search.custom_regions`: optional named layer lists for an explicit persistent diagnostic. These are manual hypotheses, not automatically selected winners. For example: `--set 'search.regions=[]' --set 'search.custom_regions=[{"label":"legacy_set","layers":[13,16,18,20,22]}]'`.
- `evaluation`: fixed-reference token count and allowed control NLL increase.
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

**Lots of length caps:** observed counts are censored. Use a larger ceiling in a fork or inspect shorter neutral test tasks. Do not treat `384` as the model's natural completion length.

**Automatic run does not visit every stage:** it is conditional. Use an explicit stage for a diagnostic, or a manual route, rather than forcing automatic scientific conclusions.

**Imported vectors stop automatic progression:** old 03 files lack full revision/precision/capture metadata and centroids. Recapture in a fresh run for full calibration. Manual exploratory use is still available; the legacy final layer is excluded unless its raw-block origin is known.

**Empty export plan:** the winning runtime experiment may be residual-only, last-token-only, noisy, or fail held-out validation. None makes a corresponding static writer edit automatically justified.

**Unsupported MoE export:** aggregate runtime hooks and expert weight layouts are separate capabilities. Fused/stacked expert banks need an explicit tested exporter. Do not rename tensor paths to make checks pass.

**Interrupted run:** rerun the identical command. A malformed partial final cache record can be repaired; corruption in the middle is not silently ignored. Reset dependent stages after changing experiment inputs through a fork.

**Changed results across machines/batches:** dtype, kernels, padding, model revision and greedy near-ties can alter generation. Runs retain identity and batch composition; no-op checks and identical baselines help detect differences, but bitwise cross-hardware equivalence is not promised.
