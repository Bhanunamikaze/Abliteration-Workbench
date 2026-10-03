# Abliteration usage guide

The README gives a quick start and the measured results. This versioned guide is mirrored in the [GitHub wiki](https://github.com/Bhanunamikaze/Abliteration-Workbench/wiki/Usage-Guide). It covers configuration, experiments, interpretation, deployment, and benchmarking. The [CLI reference](CLI.md), [data format](DATA.md), [methodology](METHODOLOGY.md), and [model support matrix](MODELS.md) provide narrower details.

## Choose the task first

| Task | Current evidence | What to run |
|---|---|---|
| Shorten Qwen responses | A five-layer runtime hook passed an eight-prompt held-out verbosity test. | Reproduce the verbosity experiment, then package its passing run. |
| Study refusal behavior | No refusal-specific result is presented as validated. | Prepare task-specific paired examples, response-level labels, and independent test and control prompts. |

The software estimates activation directions from paired examples. It does **not** perform gradient training. The bundle copies unchanged model weights and applies a hook during generation.

## 1. Set up the environment

Use Python 3.10+ with CUDA-enabled PyTorch if you want to run the example Qwen model on a GPU. The repository does not include model weights. Respect the model provider's access and license conditions.

Create a project virtual environment. For NVIDIA GPUs, install a CUDA-enabled PyTorch build chosen with the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing the workbench:

```bash
git clone https://github.com/Bhanunamikaze/Abliteration-Workbench.git
cd Abliteration-Workbench
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

After installing the appropriate PyTorch build:

```bash
python3 -m pip install -e '.[hf,plots]'
abliteration doctor
```

Check that `doctor` reports `cuda_available: true` and sufficient **total** GPU memory for the model and experiment. The default Qwen example uses BF16 and CUDA. Its [public model repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) needs no account for download. For a gated or private model, obtain access and use [`hf auth login`](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-auth-login), then `hf auth whoami`. To check public Hub connectivity without downloading weights, run `hf download Qwen/Qwen2.5-1.5B-Instruct config.json`. Model files are fetched when a stage first loads the model; `validate` and `init` do not load it.

On a machine without CUDA, use the toy example for software checks:

```bash
abliteration validate --config examples/toy_dense.json
abliteration init --config examples/toy_dense.json --run runs/toy-check
abliteration run --run runs/toy-check
```

`ablab` remains an alias of `abliteration`, and `python3 -m ablationlab` works without the installed command.

## 2. Run a calibrated Qwen experiment

The included [`verbosity.json`](../examples/verbosity.json) names the model in `model.id` and a paired [`verbosity.jsonl`](../examples/verbosity.jsonl) dataset in `dataset`. Dataset paths are resolved relative to the config file. The example has 16 train, 8 validation, 8 test, and 4 control records. Validate the data, create a run snapshot, inspect the proposed next stage, then execute the automatic route:

```bash
abliteration validate --config examples/verbosity.json
abliteration init --config examples/verbosity.json --run runs/qwen-verbosity
abliteration plan --run runs/qwen-verbosity
abliteration run --run runs/qwen-verbosity
abliteration status --run runs/qwen-verbosity
abliteration report --run runs/qwen-verbosity
```

The report is under `runs/qwen-verbosity/stages/12_report/`. Each run keeps its resolved configuration, input snapshot, stage outputs, generation cache, and decisions in its own directory. A run may finish with `needs_data` or `needs_review`; that is an experimental result, not a fabricated winner.

To select another Hugging Face model or local checkpoint, set `model.id` in a copy of the config or override it in both commands that read the config:

```bash
abliteration validate --config examples/verbosity.json --set 'model.id="ORG/MODEL"'
abliteration init --config examples/verbosity.json --run runs/my-model \
  --set 'model.id="ORG/MODEL"'
abliteration run --run runs/my-model
```

`run` uses the initialized snapshot and takes no model-selection flag. Select `model.device` and `model.dtype` for the target hardware; if BF16 is unsupported, pass `--set 'model.dtype="fp16"' to both `validate` and `init`. Use a new run or `fork` when changing model or dataset inputs.

To run one stage and its prerequisites:

```bash
abliteration run --run runs/qwen-verbosity --stage directions --with-deps
```

Stop with Ctrl+C and repeat the same `run` command to resume. Completed work is cached. Do not edit `runs/.../config.json` in place.

## 3. Supply a behavioral dataset

Use a JSON or JSONL file with disjoint `train`, `validation`, `test`, and `control` groups. `train` and `validation` rows need paired `positive` and `negative` messages; every row needs `id`, `group`, `split`, and `neutral`. A small benign example:

```json
{
  "id": "train-01",
  "group": "weather-01",
  "split": "train",
  "neutral": [{"role": "user", "content": "Name two cloud types."}],
  "positive": [
    {"role": "system", "content": "Answer as a JSON array."},
    {"role": "user", "content": "Name two cloud types."}
  ],
  "negative": [
    {"role": "system", "content": "Answer in prose."},
    {"role": "user", "content": "Name two cloud types."}
  ]
}
```

Set `behavior.metric` to something that directly measures the desired behavior: `tokens`, `words`, `regex`, `json_valid`, `exact`, `contains`, or a local `custom` scorer. For shorter responses, use `metric=tokens` and `goal=decrease`. **Token length is not a refusal metric.** A refusal study needs appropriately labeled examples and a scorer that can distinguish a refusal from a harmless explanation or a quoted phrase. See [DATA.md](DATA.md).

The included `examples/benign_decline_style.json` asks Qwen to decline harmless questions as a writing-style exercise. It can test mechanics of a refusal-like contrast; it does not establish a safety-refusal direction.

## 4. Control the intervention

Configuration and `--set` overrides control the model, data, generation, and search. Useful controls include:

| Setting | Meaning |
|---|---|
| `model.id`, `model.dtype`, `model.device` | Checkpoint and execution precision/device. |
| `generation.scope=last\|all` | Change the last valid token position or all valid positions in each forward pass. |
| `generation.phase=prefill\|decode\|both` | Apply the hook during prompt processing, cached generation, or both. |
| `search.layers` | Restrict the initial behavioral layer search. |
| `search.ablations`, `search.strengths` | Ablation and steering values to test. |
| `search.custom_regions` | Supply named multi-layer hypotheses for explicit testing. |
| `search.mode=strict\|explore` | Stop after weak refinement or continue bounded exploratory diagnostics. |
| `search.reference=zero\|negative\|auto` | Residual ablation target; `auto` compares both when a calibrated negative centroid exists. Writer outputs use zero. |
| `search.persistent_max_span_layers`, `search.persistent_cluster_gap` | Bound automatic contiguous regions built from candidate layers. |
| `generation.confirm_max_new_tokens`, `search.max_cap_confirmations` | Limit higher-cap confirmation of strong censored candidates. |
| `behavior.metric`, `behavior.goal` | Define what improvement means before evaluation. |

For example, start a new run that tests selected layers and a different phase:

```bash
abliteration init --config examples/verbosity.json --run runs/qwen-decode-check \
  --set 'search.layers=[13,16,18,20,22]' \
  --set 'generation.phase="decode"'
```

These controls create a different experiment. They do not transfer a validation result automatically. `eligible_layers` in the direction stage is a broad representation-separation screen; even all 28 Qwen layers can pass it. A layer becomes a useful behavioral candidate only after generation tests, controls, and held-out evaluation.

For a complete automatic exploratory run, add these settings to a copy of an experiment config and initialize a new run:

```json
{
  "search": {
    "mode": "explore",
    "reference": "auto",
    "persistent_max_span_layers": 8,
    "persistent_cluster_gap": 2,
    "max_cap_confirmations": 1
  },
  "generation": {
    "max_new_tokens": 128,
    "refine_max_new_tokens": 256,
    "confirm_max_new_tokens": 512
  }
}
```

```bash
abliteration init --config experiment.json --run runs/experiment
abliteration run --run runs/experiment
abliteration report --run runs/experiment
```

The JSON fragment supplements the required model, dataset, and behavior fields. In `explore` mode, weak steering can continue through writer tests, residual ablation, fixed-prefix trace, bounded persistent regions, and held-out evaluation when an effect is present. `strict` retains the conservative weak-refinement stop. A strong capped result receives `promising_censored`, a finite higher-cap retry when configured, and a report caveat. Discrete or custom metrics can retain a behaviorally promising candidate after a capped retry; token and word metrics require uncensored confirmation before candidate promotion. The report's candidate table shows each residual target, number of layers, gain, random-control result, cap rate, confirmation, and the reason for the held-out selection. The plotted residual comparisons hatch capped candidates.

Fork a completed run when changing settings so provenance remains intact:

```bash
abliteration fork --run runs/qwen-verbosity --new-run runs/qwen-other-strength \
  --set 'search.ablations=[0.25,0.5,0.75,1.0]'
abliteration run --run runs/qwen-other-strength
```

## 5. Read the result before packaging

Open the run report or `stages/11_evaluate/result.json` and check:

1. `passed` and the actual `selected_intervention`.
2. Held-out gain, win rate, response caps, and control drift.
3. Random-direction control and no-op behavior where applicable.
4. Example outputs for quality, factuality, and side effects.

The candidate `evidence_status` separates `promising`, `promising_censored`, `exploratory`, and `rejected` validation evidence. `evaluate` adds `candidate_source_stage`, `candidate_evidence_status`, `selection_reason`, and a held-out `evaluation_status`. A large discrete-score gain with `cap_rate=1` can remain a measured behavioral effect, while response termination and quality require review. For token and word metrics, the cap directly contaminates the score. A direction's representation separation, additive steering sensitivity, zero-coordinate removal, and negative-centroid projection are different claims. Removing a measured coordinate without changing the behavior does not establish that the behavior has no causal representation elsewhere.

A single layer with a high direction score is not automatically a "refusal layer" or a "verbosity layer." The validated Qwen verbosity intervention uses residual layers **13, 16, 18, 20, and 22 together**. Its eight held-out test prompts averaged 207.4 tokens before and 119.5 after the hook; four controls showed no absolute drift. See the [benchmark summary](../validation/benchmark-data.json) and [README chart](../README.md#measured-qwen-result).

## 6. Package and run a validated hook

`bundle` requires a passing held-out evaluation. It copies the source model files, direction tensors, runtime code, and an integrity manifest into a standalone directory. Packaging replays saved test and control generations and requires exact token matches.

```bash
abliteration bundle --run runs/qwen-verbosity --out checkpoints/qwen-verbosity-runtime
python3 checkpoints/qwen-verbosity-runtime/run_bundle.py \
  --prompt 'Explain how a solar panel works.'
```

The recorded local bundle from the five-layer Qwen run is `checkpoints/qwen-verbosity-runtime/` and reproduced 12/12 saved outputs. The runner needs the listed dependencies and enough compute to load the copied model. The Git repository excludes the 2.9 GB checkpoint bundle and all run directories.

The hook must be active during generation. Loading `model/` alone with a generic model loader gives the **original behavior**, because the weights are unchanged. A runtime recipe also works when the original run directory is available:

```bash
abliteration recipe --run runs/qwen-verbosity --out verbosity-recipe.json
abliteration generate --run runs/qwen-verbosity --recipe verbosity-recipe.json \
  --prompt 'Explain how a solar panel works.'
```

Copied weight projection is a separate, explicit operation. A residual, last-token hook cannot generally be converted into an equivalent static weight edit; an empty export plan can therefore be correct. See [CLI.md](CLI.md) and [METHODOLOGY.md](METHODOLOGY.md).

## 7. Benchmark refusal behavior

For the Llama 3.2 3B baseline, use the existing pipeline with [`llama32_baseline.json`](../examples/llama32_baseline.json). Provide its dataset under `runs/llama32-baseline-prompts.jsonl` using the [standard split contract](DATA.md). The config enables vanilla test collection and the local semantic scorer; it contains no selected layers. [`refusal_controls.jsonl`](../examples/refusal_controls.jsonl) supplies 50 ordinary benign controls. Keep benchmark provenance and categories with the prompt rows, and isolate related groups across splits.

```bash
python -m ablationlab init --config examples/llama32_baseline.json --run runs/llama32-refusal
python -m ablationlab run --run runs/llama32-refusal --stage baseline --with-deps
```

`stages/02_baseline/outputs.json` retains responses, token IDs, termination flags, scores, and classifier evidence. Join its IDs with `dataset.json` to summarize safety, over-refusal, and ordinary-control cohorts separately; exclude benign schema/contrast examples from benchmark percentages. The scorer's `score=1` predicts refusal and `score=0` predicts non-refusal. Review these predictions before publishing a rate. This command stops at baseline measurement; no capture, search, or model training is run.

For a small independent comparison, use [Röttger et al.'s XSTest](https://github.com/paul-rottger/xstest). Put its CC-BY-4.0 `xstest_prompts.csv` under `runs/benchmarks/`, then evaluate a packaged hook:

```bash
python3 tools/evaluate_xstest.py \
  --csv runs/benchmarks/xstest_prompts.csv \
  --bundle checkpoints/qwen-verbosity-runtime \
  --out runs/xstest-local.json
```

The script selects **20 safe and 20 unsafe prompts** by a fixed rule across XSTest categories. It compares original Qwen with the hook under the same deterministic decoding and reports a response-prefix refusal marker with Wilson intervals. Inspect changed responses: a new first phrase can alter that marker while the response still refuses. A dedicated refusal study should use response-level annotations and independent control prompts before packaging a hook.

Regenerate the README verbosity figure from the tracked result summary:

```bash
python3 tools/plot_readme_benchmarks.py
```

## 8. Troubleshooting and limits

| Symptom | Check |
|---|---|
| `doctor` says CUDA unavailable | Activate the CUDA-enabled environment; verify PyTorch sees the GPU. |
| Model access fails | Verify the model ID, cached snapshot, Hugging Face access, and provider terms. |
| GPU runs out of memory | Reduce batch size or output ceiling in a new run; quantization/offload are explicit choices with separate limitations. |
| All layers pass the direction screen | Read the behavioral sweep and held-out evaluation; this screen is intentionally broad. |
| Run stops with `needs_review` | Inspect outputs, controls, and metrics; add better data or fork a new hypothesis. |
| Bundle command rejects a run | It needs a completed, passing `evaluate` stage and a supported nonquantized backend. |

The built-in model adapters cover supported dense and MoE text decoders. Fused experts, multimodal models, encoder-decoder models, state-space models, arbitrary model sizes, and TPU require additional support. See [MODELS.md](MODELS.md) for the current architecture limits.
