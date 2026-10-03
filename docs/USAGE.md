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

For the local `alter` environment used in the recorded runs:

```bash
conda activate alter
git clone https://github.com/Bhanunamikaze/Abliteration-Workbench.git
cd Abliteration-Workbench
python3 -m pip install -e '.[hf,plots,test]'
abliteration doctor
```

Check that `doctor` reports `cuda_available: true` and enough free VRAM. The default Qwen example uses BF16 and CUDA. On a machine without CUDA, use the toy example for software checks:

```bash
abliteration validate --config examples/toy_dense.json
abliteration init --config examples/toy_dense.json --run runs/toy-check
abliteration run --run runs/toy-check
```

`ablab` remains an alias of `abliteration`, and `python3 -m ablationlab` works without the installed command.

## 2. Run a calibrated Qwen experiment

Validate the data, create a run snapshot, inspect the proposed next stage, then execute the automatic route:

```bash
abliteration validate --config examples/verbosity.json
abliteration init --config examples/verbosity.json --run runs/qwen-verbosity
abliteration plan --run runs/qwen-verbosity
abliteration run --run runs/qwen-verbosity
abliteration status --run runs/qwen-verbosity
abliteration report --run runs/qwen-verbosity
```

The report is under `runs/qwen-verbosity/stages/12_report/`. Each run keeps its resolved configuration, input snapshot, stage outputs, generation cache, and decisions in its own directory. A run may finish with `needs_data` or `needs_review`; that is an experimental result, not a fabricated winner.

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
| `behavior.metric`, `behavior.goal` | Define what improvement means before evaluation. |

For example, start a new run that tests selected layers and a different phase:

```bash
abliteration init --config examples/verbosity.json --run runs/qwen-decode-check \
  --set 'search.layers=[13,16,18,20,22]' \
  --set 'generation.phase="decode"'
```

These controls create a different experiment. They do not transfer a validation result automatically. `eligible_layers` in the direction stage is a broad representation-separation screen; even all 28 Qwen layers can pass it. A layer becomes a useful behavioral candidate only after generation tests, controls, and held-out evaluation.

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
