# Model/backend support

## Built-in backend

PyTorch with Hugging Face `AutoModelForCausalLM`, text inputs, cached greedy generation, floating weights or optional bitsandbytes loading. Supported device choices: `cuda`, `cuda:0`, `cpu`, `auto`. CPU must be selected explicitly unless using auto. TPU/XLA, TensorFlow, JAX, GGUF/llama.cpp and external HTTP inference servers are not built-in backends.

| Architecture/layout | Runtime support path | Checkpoint projection |
|---|---|---|
| Qwen2/Qwen2.5, Llama, Mistral-like dense decoders | Raw block, attention output, MLP output | Plain output `nn.Linear` weights/biases |
| GPT-2-style `transformer.h` | Block and detected output modules | Conv1D orientation is not implemented; reject rather than guess |
| GPT-NeoX-style | Block, attention dense, MLP dense_4h_to_h | Plain linears only; parallel residual topology must be interpreted correctly |
| Phi/Gemma-like layouts | Module-path candidates plus actual shape validation | Only verified plain linear output sites; post-norm/parallel effects need model-specific interpretation |
| Mixtral/Qwen2-MoE-like layouts | Block, attention, aggregated sparse-mixture output | Registered **unfused** scalar-mixture expert output linears; no fused/stacked expert tensor edits |
| New model with ordinary block stack | JSON adapter map can override paths | No automatic arbitrary parameter editing |
| Multimodal, encoder-decoder, state-space, variable-width | Custom backend/adapter required | Not supported by generic exporter |

“Detected” means that actual outputs passed `[batch, sequence, hidden]` shape checks. It does not certify the semantics of that component or guarantee behavioral success. See validation logs: the build environment tested small real PyTorch fixtures, not downloaded full-sized HF checkpoints or RTX inference.

## Why MoE needs different handling

Expert routing often flattens or reorders tokens and may only call an expert for a subset of a batch. Applying `hidden[:, -1, :]` to an expert output is therefore not generally valid. The default intervention target is the **reassembled mixture output**, where batch/sequence alignment is available. Individual experts are inventoried, not falsely treated as independent batch-aligned layers.

For recognized scalar-weighted mixtures, a zero projection P obeys `P(sum_i g_i*y_i) = sum_i g_i*P(y_i)`. The registered unfused exporter can therefore project each recognized expert output linear, including a supported shared expert output. It does not edit the router. Nonlinear post-mixture transformations, fused tensors, quantized experts, unknown sharing, or unrecognized implementations are not assumed equivalent. The exporter also performs a full-model fixed-input check before saving.

## Memory

MoE active parameters per token are not total resident weights. A sparse model can require much more memory than its active-parameter label suggests. A 3060 cannot run arbitrary-sized models just because the adapter understands their blocks. `allow_offload` and bitsandbytes loading are explicit options, not promises of speed. Unsupported/OOM situations stop with retained artifacts.

## Custom maps

Place this under `model.adapter` in the run configuration:

```json
{
  "layers": "model.layers",
  "attention": "self_attn.o_proj",
  "mlp": "mlp.down_proj"
}
```

For a known aggregate mixture mapping, include `"moe": true`. Paths are relative to the loaded model for `layers`, and relative to each block for attention/MLP. No suffix-only “first thing that happens to match” weight surgery is used.

## New backends

Set `model.backend_plugin` to `your_module:factory`. The factory receives the resolved config and returns a backend with:

- `identity`, `capabilities`, `adapter`, `model`, `tokenizer`, and `device`.
- `generate(records, bundle=None, spec=None, max_new_tokens=None)` returning ordered text/token/termination dictionaries.
- `capture(conversations)` returning CPU FP32 `[examples,layers,hidden]` raw-site activations.
- `trace(record, bundle, spec=None, prefix_ids=None)` returning `[layers,rank]` projections on matched fixed inputs.
- `reference_nll(record, reference_ids, bundle=None, spec=None)` for cached teacher-forced drift evaluation.

A custom backend must retain the same meaning of capture and intervention sites, honor token masks, remove hooks on failure, and expose unsupported capabilities. Custom-backend checkpoint export is not automatically enabled.
