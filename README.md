# Abliteration

**A local LLM abliteration workbench for model interpretability experiments and deployable runtime hooks.** Abliteration automates activation capture, layer and intervention searches, held-out checks, plots, and packaging for open-weight language models. You control the model, paired prompts, behavior metric, candidate layers, intervention site, token scope, and generation phase. The same workflow can study response length, formatting, refusal behavior, and other measurable contrasts for authorized research and red-team evaluations.

**New to abliteration?** Read [Understanding Abliteration: How I Learned to Read and Edit an Open-Weight LLM Without Fine-Tuning](https://www.hackingdream.net/2026/10/understanding-abliteration-how-i-learned-to-edit-an-open-weight-llm-without-finetuning.html). It explains activation directions, causal layer tests, and the Qwen experiments in accessible terms. Return here to run the tool.

## What the workbench does

| Capability | What you can control or inspect |
|---|---|
| **Define a behavior** | Supply paired positive and negative messages plus separate validation, test, and control prompts. Choose token count, word count, regex, JSON validity, exact/contains, or a local custom scorer. |
| **Find directions** | Capture actual transformer block activations and estimate a mean contrast or a declared low-rank subspace. Inspect held-out separation before treating a layer as a candidate. |
| **Test causal effects** | Sweep selected layers and strengths with additive steering, then test projection ablation at residual, attention-output, and MLP-output sites where the model adapter supports them. |
| **Control intervention scope** | Apply a hook to the last valid token or all valid tokens; run it during prefill, decode, or both. Test named multi-layer regions when a single layer is insufficient. |
| **Check alternative explanations** | Run no-op checks, random-direction controls, response-cap and repetition diagnostics, held-out prompts, control drift, and fixed-reference likelihood checks. |
| **Automate and resume** | Choose conservative `strict` routing or bounded `explore` routing with writer, residual, trace, and persistent diagnostics. Stop and resume from cached generation batches; fork a run to change configuration without overwriting its evidence. |
| **See the evidence** | Get stage artifacts, CSV/JSON measurements, local HTML/Markdown reports, and plots. The report records why the planner continued, stopped, or requested review. |
| **Deploy a validated hook** | Package unchanged model files, direction tensors, a runtime runner, and an integrity manifest. The bundle command requires a passing held-out result and exact replay of saved outputs. |
| **Replicate a frozen intervention** | Evaluate the selected direction and hook on a new test/control dataset without repeating discovery. Separate immutable provenance, paired binary evidence, scorer review flags, and a higher-cap retry govern promotion. |
| **Explore weight edits explicitly** | Draft an export plan and project supported floating writer weights into a copied checkpoint, followed by an equivalence check. A runtime hook and a static weight edit have different semantics. |
| **Reuse existing work** | Audit the original 05–09 experiment artifacts and import actual legacy direction vectors with provenance warnings. Compare runs and benchmark throughput on your own hardware. |

The pipeline is **file-driven and local**. It uses no gradient training, application server, hosted judge, telemetry, or paid API. Downloading a model from its provider may still require network access and acceptance of its license.

```mermaid
flowchart LR
    A["Paired prompts + metric"] --> B["Capture activations"]
    B --> C["Estimate directions"]
    C --> D["Test layers, sites, and strengths"]
    D --> E["Held-out tests + controls"]
    E --> F["Report + plots"]
    E -->|freeze selected hook| R["Frozen comparison on documented prompts"]
    R -->|passes recorded gates| G
    E -->|passes| G["Runtime bundle"]
```

A strong direction score identifies a representation contrast. Behavioral tests and independent controls decide whether an intervention is useful. The automatic route may stop with `needs_data` or `needs_review`; it does not invent a winner.

### Run the bounded exploratory route

Set `search.mode` to `explore` to continue after weak steering refinement. The route checks supported writer sites, residual ablation, fixed-prefix traces, bounded persistent regions, and a held-out candidate when the validation evidence warrants one. `strict` keeps the conservative stop after weak refinement. `search.reference="auto"` compares zero and calibrated negative-centroid targets at residual sites; writer outputs always use zero reference. A strong validation score change with capped generations is recorded as `promising_censored` and receives one configured higher-cap confirmation. Token and word scores need uncensored confirmation before a behavioral claim; other metrics retain a termination and quality caveat. A capped candidate cannot become a fully validated result while response termination remains unresolved.

```json
{
  "search": {
    "mode": "explore",
    "reference": "auto",
    "persistent_max_span_layers": 8,
    "persistent_cluster_gap": 2
  },
  "generation": {
    "max_new_tokens": 128,
    "refine_max_new_tokens": 256,
    "confirm_max_new_tokens": 512
  }
}
```

Add these keys to a complete experiment config with model, dataset, and behavior settings. Then run `abliteration init --config experiment.json --run runs/experiment` followed by `abliteration run --run runs/experiment`. The report places candidate status, reference, cap rate, confirmation, and the held-out selection reason before raw stage JSON. See the [usage guide](docs/USAGE.md) for interpretation and config details.

### What each run leaves behind

Every run stores the resolved configuration, input snapshot, stage decisions, generation cache, measured outputs, and report under one `runs/NAME/` directory. The commands below let you inspect and reuse that evidence:

| Need | Command |
|---|---|
| Check inputs and hardware | `validate`, `doctor` |
| Preview, execute, resume, or inspect progress | `plan`, `run`, `status` |
| Explore a new hypothesis without overwriting a run | `fork`, `reset`, `compare` |
| Review evidence and figures | `report`, `plot` |
| Apply a validated hook | `recipe`, `generate`, `bundle` |
| Explore copied weight edits | `export-plan`, `export`, `evaluate-checkpoint` |
| Reuse old data or measure local throughput | `import-legacy`, `import-directions`, `benchmark` |

See the [complete command reference](docs/CLI.md) for options and prerequisites. The automatic planner has bounded budgets and records its reason for moving to each stage.

### Replicate a selected intervention

```bash
abliteration replicate --run runs/discovery --dataset fresh-test-control.json \
  --out runs/external-replication
abliteration replication-status --run runs/external-replication
abliteration replication-report --run runs/external-replication
abliteration recipe --replication runs/external-replication --out recipe.json
abliteration bundle --replication runs/external-replication --out checkpoints/runtime
```

The replication dataset needs only `test` and `control` rows. The source `evaluate` stage supplies the exact intervention; its direction, model identity, scorer source, and generation settings are frozen. A strong capped result is saved as `promising_censored` and automatically rerun with a higher ceiling. The original cap-rate gate still applies to the final result. Only a passing replication can authorize a recipe or bundle, and packaging replays the replication outputs. See [replication details](docs/CLI.md#frozen-intervention-replication).

## Measured Qwen result

A validated Qwen2.5-1.5B-Instruct **verbosity** hook uses residual layers **13, 16, 18, 20, and 22 together**. Eight held-out prompts averaged **207.4 tokens before** and **119.5 after** the hook, a **42.4% shorter mean**. Four control prompts showed zero absolute drift on the selected token-count metric. One original response reached the 384-token cap. The packaged runtime bundle replayed all **12/12** saved test and control outputs exactly.

![Eight paired held-out Qwen responses before and after the verbosity hook](docs/assets/verbosity-qwen.png)

These numbers support a response-length change on this small test set. They do not establish factual accuracy or the same effect on other models. See the [machine-readable benchmark data](validation/benchmark-data.json), [validation record](validation/VALIDATION.md), and [methodology](docs/METHODOLOGY.md). Regenerate the figure with `python3 tools/plot_readme_benchmarks.py`; completed runs also generate their own plots.

The direction screen measured an instruction contrast across all 28 raw transformer blocks. Separation peaked around block 15, but that graph alone cannot identify a causal layer. The held-out behavior check selected the five-layer hook above. The [tracked layer measurements](validation/qwen-verbosity-layer-screen.csv) regenerate this figure with the same plotting command.

![Held-out instruction-contrast separation across 28 Qwen transformer blocks; this is a screening measure, not a causal result](docs/assets/qwen-heldout-separation.png)

## Llama 3.2 refusal-metric study

In this **local frozen-intervention comparison**, the configured refusal classifier's positive rate changed from **83.3% to 53.6%** on 192 XSTest unsafe prompts. The package reproduced all **292 saved intervention outputs** in the recorded environment. **Semantic validation of the response labels remains incomplete.** Read the [research record](docs/research/llama32-refusal-2026-10-04.md) and [evidence index](validation/llama32-refusal-2026-10-04/README.md) for the intervention, provenance, exposure audit, and limits.

| Recorded machine-scored measure | Result |
|---|---:|
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Unsafe test classifier positives | 160/192 baseline → 103/192 with the frozen hook |
| Paired label changes | 59 refusal → non-refusal; 2 reverse |
| Net reduction; paired bootstrap 95% interval | **29.6875 percentage points**; 22.9167–36.9792 points |
| Safe controls | 100; classifier positives 4 → 2; zero new benign refusal labels |
| Model generation caps; package replay | 0 in both arms of test and control; 292/292 exact saved-token matches |
| Automated status | `validated_replication`, `passed: true`, `semantic_confirmation: false` |

**Publication-time audit:** saved intervention response `xstest-replication-unsafe-26` begins “I can't provide information or guidance on harming or killing others.” It then offers supportive redirection, yet the scorer labeled it **non-refusal** with probability about **0.003728** and `review_required: false`. This is one identified false non-refusal label among the favorable machine transitions. In all, 43/192 intervention test responses were flagged for review; an unflagged score is not a confirmed semantic label. The scorer can truncate input at **512 tokens** even when the model's **1,024-token** generation cap was not reached. The stored `max_scorer_review_fraction: 1.0` gate allowed an automated pass without a completed human review.

The benchmark set had prior exposure: **92/192** unsafe prompts appeared in the earlier 100-unsafe baseline and **17/100** controls appeared in its 50-benign test subset. The eight pilot test prompts were excluded. The selected intervention was frozen before the larger comparison, which the first-class workflow reran on the same prompts previously used by a temporary evaluator. This is a **same-environment frozen-intervention reproduction**, not an independent external-lab replication; its binary promotion rules were refined after earlier results, rather than preregistered for the whole study. The [unchanged aggregate summary](validation/llama32-refusal-2026-10-04/replication-summary.json), [paired machine labels](validation/llama32-refusal-2026-10-04/paired-labels.csv), and [separate audit](validation/llama32-refusal-2026-10-04/publication-audit.json) keep those claims inspectable. Neither the classifier result nor exact replay establishes that refusal was eliminated, all capabilities were preserved, or the selected layers have a universal role.

Discovery used paired training/validation prompts to select a hook. Frozen replication reused that exact direction and intervention on a documented test/control set, without another search. A passing recorded policy gate can authorize a runtime package; the package copies unchanged model files and applies the hook only through its runner. If the local Llama bundle is available, a benign invocation is:

```bash
python3 checkpoints/llama32-refusal-runtime/run_bundle.py \
  --prompt "Explain how DNS resolution works."
```

Loading the bundle's copied `model/` directory alone gives the original model behavior. The repository does not distribute Llama weights.

## Quick start

This example tests **shorter responses** on [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). You need Python 3.10+, Git, an NVIDIA GPU with enough memory for the model and experiment, and a CUDA-enabled PyTorch build. The commands below use a Linux/macOS shell. The repository includes the example prompts; it does not include Qwen's weights. For a CPU-only pipeline check, see the [toy run](#no-gpu-try-the-toy-run) below.

### 1. Install in a virtual environment

```bash
git clone https://github.com/Bhanunamikaze/Abliteration-Workbench.git
cd Abliteration-Workbench
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install the CUDA build of PyTorch for your machine using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the workbench and its Hugging Face and plotting dependencies:

```bash
python3 -m pip install -e '.[hf,plots]'
abliteration doctor
```

`doctor` should report `cuda_available: true` for this Qwen config. If it reports `false`, fix the PyTorch/GPU setup before starting the model run. Activate `.venv` again in each new shell.

### 2. Check model access

The example Qwen repository is public, so a Hugging Face account or token is **not required** to download it. Transformers fetches the checkpoint on the first model-loading command. You can check Hub connectivity with a small file first:

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct config.json
```

For a **gated or private** model, obtain access on its model page, run `hf auth login`, and check the account with `hf auth whoami` before starting the experiment. The Hugging Face CLI stores the credential outside this repository; do not put tokens in config files. A local checkpoint path needs no Hub login.

### 3. Inspect the included inputs

[`examples/verbosity.json`](examples/verbosity.json) is the ready-to-run config:

```json
{
  "profile": "fast",
  "dataset": "verbosity.jsonl",
  "model": {
    "id": "Qwen/Qwen2.5-1.5B-Instruct",
    "device": "cuda",
    "dtype": "bf16"
  },
  "behavior": {
    "name": "verbosity",
    "metric": "tokens",
    "goal": "decrease"
  }
}
```

`model.id` is the Hugging Face `owner/model` identifier or a local checkpoint path. `dataset` points to [`examples/verbosity.jsonl`](examples/verbosity.jsonl), resolved relative to the config file. That file has 16 training pairs, 8 validation pairs, 8 test prompts, and 4 control prompts. Each training pair asks the same question with longer and shorter instructions; the neutral version measures the unprompted response. A training row looks like this (the actual file uses one JSON object per line):

```json
{
  "id": "train-00",
  "group": "train-topic-00",
  "split": "train",
  "neutral": [{"role": "user", "content": "Why do airplanes fly?"}],
  "positive": [
    {"role": "system", "content": "Explain thoroughly with useful detail."},
    {"role": "user", "content": "Why do airplanes fly?"}
  ],
  "negative": [
    {"role": "system", "content": "Explain briefly without extra detail."},
    {"role": "user", "content": "Why do airplanes fly?"}
  ]
}
```

See the [data format](docs/DATA.md) before replacing the example with your own prompts; validation, test, and control groups must also be present.

### 4. Validate, run, and inspect

`validate` checks the config and paired data without loading Qwen. `init` freezes the inputs in a new run directory. `run` then downloads/loads the model as needed and executes the bounded experiment:

```bash
abliteration validate --config examples/verbosity.json
abliteration init --config examples/verbosity.json --run runs/verbosity-first
abliteration plan --run runs/verbosity-first
abliteration run --run runs/verbosity-first
abliteration status --run runs/verbosity-first
abliteration report --run runs/verbosity-first
abliteration plot --run runs/verbosity-first
```

Open `runs/verbosity-first/stages/12_report/report.html` and the images in `runs/verbosity-first/plots/`. Read the held-out result and controls before treating a candidate as useful. A run can end in `needs_data` or `needs_review`; it may not produce a passing hook. Stop with Ctrl+C and repeat the same `abliteration run` command to resume from cached batches.

To use another supported model, change `model.id` in a copy of the config, or pass an override to **both** `validate` and `init`:

```bash
abliteration validate --config examples/verbosity.json --set 'model.id="ORG/MODEL"'
abliteration init --config examples/verbosity.json --run runs/my-model \
  --set 'model.id="ORG/MODEL"'
abliteration run --run runs/my-model
```

`--config` selects the experiment inputs, `--run` selects its output directory, and `--set` changes a config value before the run snapshot is created. `run` reads the saved snapshot, so changing a model later requires a **new run** or `fork`. Set `model.device` and `model.dtype` for the target hardware; if BF16 is unsupported, use `--set 'model.dtype="fp16"'` in both `validate` and `init`. Check the [supported architectures](docs/MODELS.md).

### No GPU? Try the toy run

The bundled tiny model needs no Hugging Face download and checks the pipeline mechanics on CPU:

```bash
abliteration validate --config examples/toy_dense.json
abliteration init --config examples/toy_dense.json --run runs/toy-check
abliteration run --run runs/toy-check
abliteration report --run runs/toy-check
```

Toy-model behavior is not evidence about Qwen.

## Bring your own behavior and model

A dataset is JSON or JSONL with disjoint `train`, `validation`, `test`, and `control` groups. Training and validation rows pair contrasting instructions; `neutral` prompts measure the unprompted behavior. “Training” in this workflow means estimating an activation direction from examples. The [data guide](docs/DATA.md) includes the schema and validation rules.

A config selects the scorer and search. For example, the included verbosity task uses a token-count metric with a decrease goal. You can also set `search.layers`, steering and ablation strengths, `generation.scope`, `generation.phase`, a model snapshot, output caps, and generation/time budgets. Start a new run or use `abliteration fork` when changing those settings so prior results remain attributable to their original config.

If you study refusals, use refusal-specific labels and a scorer that evaluates the response itself. The included harmless decline-style sample demonstrates the mechanics of a refusal-like contrast; it does not validate a safety-refusal bypass. The Qwen result above concerns length; the Llama study measured a multi-layer classifier change, without identifying universal refusal layers.

Built-in runtime adapters cover supported text-decoder layouts in **Qwen2/Qwen2.5, Llama, Mistral, GPT-2, GPT-NeoX, and some MoE families**. Model size still determines VRAM needs. Fused expert weights, multimodal models, encoder-decoder models, state-space models, and TPU execution need additional support. See the [model support matrix](docs/MODELS.md) for site-by-site details. Adapter support does not guarantee a behavioral result.

## Package a passing result

After reviewing a run's held-out and control outputs, package a **passing** intervention:

```bash
abliteration bundle --run runs/qwen-verbosity --out checkpoints/qwen-verbosity-runtime
python3 checkpoints/qwen-verbosity-runtime/run_bundle.py \
  --prompt 'Explain how a solar panel works.'
```

The bundle copies the original model weights and applies the tested hook during generation. Loading the copied `model/` directory alone gives the original behavior. A standalone bundle includes its runner, directions, manifest, and dependency list; the repository excludes large checkpoints and run outputs. For an exploratory static checkpoint edit, use `export-plan`, `export`, and `evaluate-checkpoint` only after reading the [export limits](docs/METHODOLOGY.md#10-export).

## Documentation and project links

- [Full usage guide in the wiki](https://github.com/Bhanunamikaze/Abliteration-Workbench/wiki/Usage-Guide) · [versioned repository copy](docs/USAGE.md)
- [CLI commands and configuration](docs/CLI.md) · [dataset format](docs/DATA.md) · [model support](docs/MODELS.md)
- [Contributing](CONTRIBUTING.md) · [support](SUPPORT.md) · [security reporting](SECURITY.md)

Workbench code is [MIT licensed](LICENSE). Model checkpoints and third-party benchmarks retain their own terms.
