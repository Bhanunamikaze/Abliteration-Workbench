# Quick start

1. Clone the repository and use your CUDA-enabled PyTorch environment.

```bash
git clone https://github.com/Bhanunamikaze/Abliteration-Workbench.git
cd Abliteration-Workbench
python -m pip install -e '.[hf,plots,test]'
abliteration doctor
```

2. Start a new calibrated Qwen verbosity experiment:

```bash
abliteration init --config examples/verbosity.json --run runs/qwen-style
abliteration run --run runs/qwen-style
```

3. Inspect the planner and local report:

```bash
abliteration plan --run runs/qwen-style
abliteration report --run runs/qwen-style
```

Open `runs/qwen-style/stages/12_report/report.html` in your browser. A result of `needs_review` or `needs_data` is a valid experimental outcome, not a software failure.

4. Reuse your historical numeric results without loading the model:

```bash
abliteration import-legacy --source /home/stark/Projects/Abliteration --out runs/legacy-review
```

5. Stop with Ctrl+C and resume by repeating the same run command. Change the configuration through `abliteration fork`, not by editing a run snapshot.

The [README](README.md) summarizes the workbench and results. The [full usage guide](https://github.com/Bhanunamikaze/Abliteration-Workbench/wiki/Usage-Guide) covers the workflow; model support and non-support are listed in [docs/MODELS.md](docs/MODELS.md), task data in [docs/DATA.md](docs/DATA.md), and commands in [docs/CLI.md](docs/CLI.md).

**Validation:** see [validation/VALIDATION.md](validation/VALIDATION.md). Core dense/MoE fixture tests ran on CPU; full HF model inference and RTX 3060 performance were not tested in the build environment.
