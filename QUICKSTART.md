# Quick start

The [README quick start](README.md#quick-start) gives the full setup, config, model access, and output walkthrough. This page is a short command checklist for the included Qwen verbosity example.

1. Clone the repo and make a virtual environment with Python 3.10+:

   ```bash
   git clone https://github.com/Bhanunamikaze/Abliteration-Workbench.git
   cd Abliteration-Workbench
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install --upgrade pip
   ```

2. For a GPU run, install CUDA-enabled PyTorch using the [official selector](https://pytorch.org/get-started/locally/). Then install the workbench:

   ```bash
   python3 -m pip install -e '.[hf,plots]'
   abliteration doctor
   ```

   The Qwen example requires `cuda_available: true`. The [Qwen model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) is public and downloads automatically when the model first loads. For a gated/private model, obtain access and run `hf auth login`; check it with `hf auth whoami`. A small public download check is `hf download Qwen/Qwen2.5-1.5B-Instruct config.json`.

3. Inspect [`examples/verbosity.json`](examples/verbosity.json) and its paired [`verbosity.jsonl`](examples/verbosity.jsonl). `model.id` is the Hugging Face model ID or local checkpoint path; `dataset` is relative to the config file. `--run` names the output directory. Then execute:

   ```bash
   abliteration validate --config examples/verbosity.json
   abliteration init --config examples/verbosity.json --run runs/qwen-style
   abliteration plan --run runs/qwen-style
   abliteration run --run runs/qwen-style
   abliteration status --run runs/qwen-style
   abliteration report --run runs/qwen-style
   abliteration plot --run runs/qwen-style
   ```

   Open `runs/qwen-style/stages/12_report/report.html` and `runs/qwen-style/plots/`. An automatic run can finish with `needs_review` or `needs_data`; a passing hook is not guaranteed. Stop with Ctrl+C and repeat the same `run` command to resume.

To try another model, pass `--set 'model.id="ORG/MODEL"'` to **both** `validate` and `init`, using a new run directory. `run` reads that saved configuration. See the [model support matrix](docs/MODELS.md), [data format](docs/DATA.md), and [CLI reference](docs/CLI.md).

For a CPU-only pipeline check, use `examples/toy_dense.json` and `runs/toy-check` in place of the Qwen config and run directory. The toy model needs no Hugging Face download. The full [wiki usage guide](https://github.com/Bhanunamikaze/Abliteration-Workbench/wiki/Usage-Guide) covers custom datasets, controlled interventions, reports, and bundles.
