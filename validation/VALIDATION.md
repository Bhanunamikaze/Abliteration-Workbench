# Validation record: Abliteration

Initial build date: 2026-10-03 (UTC). The initial fixture test record below is retained for provenance. Later Qwen runs used the local CUDA-enabled `alter` environment.

## Later real-model evidence

The local Qwen2.5-1.5B-Instruct verbosity experiment (`runs/qwen-legacy-region`, excluded from Git) passed its held-out evaluation: eight test prompts averaged 207.375 original tokens and 119.5 hook tokens, four controls had no absolute metric drift, and a runtime bundle replayed all 12 saved outputs exactly. The [tracked result data](benchmark-data.json), [layer-screen measurements](qwen-verbosity-layer-screen.csv), and [README figures](../README.md#measured-qwen-result) summarize it. This validates that specific five-layer runtime hook on that small test; it does not generalize to every model or prompt.

The current `alter` environment also ran the full test suite after the plot addition; 138 tests passed. The historical 124-test fixture snapshot below remains an earlier build record.

## What actually ran

**124 tests passed; 1 optional test module skipped.** See [pytest output](pytest.log) and [JUnit XML](pytest.xml).

The following checks were executed against the delivered source, not merely described:

- Finite input/config validation, group leakage and duplicate-prompt detection.
- FP32 directional projection math, rank-one and multi-rank bases, zero-strength no-op, partial/full removal, and negative-reference semantics.
- Same-rank random controls orthogonal to learned subspaces; impossible complements rejected.
- Tensor/tuple/list/dictionary outputs, dense and aggregated MoE runtime adapters, hook scope and cleanup after exceptions.
- Actual cached generation on small real PyTorch dense and top-2/3-expert MoE decoder fixtures.
- Execution of all twelve pipeline stages, adaptive stopping on insufficient evidence, explicit route controls, and dependency invalidation.
- Generation caching, truncated-final-journal recovery, interruption/resume, immutable snapshots, exclusive run locking, and time/generation budget handling, and rejection of changed model/runtime identity during resume.
- Legacy .pt direction imports with weights-only loading, model/shape checking, missing-calibration warnings, and final-layer exclusions.
- Fresh held-out/control comparison, cached fixed-token NLL drift evaluation, matched-prefix coordinate diagnostics, and offline report/plot generation.
- Dense linear weight/bias projection equivalence to all-token runtime writer projection.
- Projection through unfused scalar-mixture experts, copied-checkpoint save/reload, original in-memory parameter restoration, rejection of tied/aliased selected parameters, and no implicit export when no valid plan exists.
- Legacy numeric replay using your actual uploaded experiment results.

In addition, **34 CLI-entrypoint invocations passed** on dense and MoE fixtures. The real argument parser/dispatcher was invoked programmatically in one warm process; this is not presented as 34 fresh Python process starts. Logs and example reports are in [smoke/](smoke/README.md).

`python -m compileall -q ablationlab tests tools` completed successfully. A Python wheel was built using `pip wheel --no-deps --no-build-isolation .`; see [wheel build log](wheel_build.log).

## Legacy-data verification

The audit read **15 existing 05–09 artifacts** and retained their input hashes. Full historical answers are not bundled as newly verified ground truth.

- Code 05 table: **560 rows**.
- Recomputed mean unmodified response length: **315.375 tokens**.
- Layer 16 mean symmetric steering effect: **516.875 tokens**.
- Recomputed legacy Layer 16 composite score: **0.8155359033143397**.
- The old layer-13 intervention / observed-layer-15 mean of individual absolute ratios remains **14.317313824952969**.
- The separately calculated ratio of mean absolute coordinates is **2.9824034316499923**.

Those last two values are different statistics. Neither proves semantic reconstruction. Original observations are preserved; the audit adds alternative diagnostics and warnings rather than silently replacing them.

See [audit.md](legacy_replay/audit.md), [audit.json](legacy_replay/audit.json), and the recalculated CSVs. The actual user `verbosity_directions.pt` file was not present in the build environment; it was not fabricated or reconstructed from scalar summaries.

## Environment

- Python: **3.13.5**.
- PyTorch: **2.10.0+cpu**, CPU build.
- NumPy: **2.3.5**.
- safetensors: **0.7.0**.
- CUDA available: **False**.

Full metadata: [environment.json](environment.json).

## What did not run in the initial fixture build

**No CUDA/RTX 3060 benchmark, full Qwen checkpoint inference, TPU execution, bitsandbytes load, or quantized-model experiment was run in that initial fixture build.** That environment had CPU PyTorch only. The later local Qwen experiments described above used CUDA and full checkpoint inference.

The optional Hugging Face architecture test module was **skipped because Transformers was absent**. An attempt to install Transformers/Accelerate/Tokenizers failed because the build container could not resolve the package-download endpoint. Therefore the implementation's HF architecture paths are documented support targets, not empirically certified full-model integrations in this environment.

The optional suite contains tiny random Qwen2, Llama, Mistral, GPT-2, GPT-NeoX, Mixtral and Qwen2-MoE cases. Run them locally after installing the HF extra:

```bash
python -m pip install -e '.[hf,plots,test]'
python -m pytest -q -m hf
```

The dense/MoE fixtures used by the core suite are deliberately untrained. Passing their tests validates software mechanics, not successful semantic editing or harmlessness of a behavioral intervention. No benchmark accuracy, factual correctness, general refusal-removal effectiveness, or arbitrary-model compatibility is claimed.

## Reproduce locally

```bash
python tools/validate_local.py
```

This retains logs under `validation/local/` and returns nonzero on a failed command. It does not silently install packages or download models. For your RTX 3060, also run `abliteration doctor`, a tiny `abliteration benchmark`, and then a fresh calibrated run using your own data.
