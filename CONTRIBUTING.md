# Contributing to Abliteration

Issues and pull requests are welcome for reproducible bugs, supported model adapters, metrics, and documentation.

1. Open an issue with the model architecture, command, expected result, observed result, and a minimal nonprivate example. Do not attach model weights, credentials, or private datasets.
2. For code changes, keep existing commands compatible where practical and explain any behavior change.
3. Run `python3 -m pytest -q` in an environment with the project dependencies. Optional Hugging Face tests require the `hf` extra.
4. Include the evaluation method and limitations when proposing claims about behavior changes. Response length alone does not measure refusal behavior.

The current support matrix is in [docs/MODELS.md](docs/MODELS.md). For a sensitive security issue, use a private GitHub security advisory instead of a public issue.
