# Implementation references

Consulted official documentation/source while building this package. These sources describe the underlying APIs and architectures, not a validation of this package.

- Hugging Face model loading, dtype, device maps, expert implementations and serialization:
  https://huggingface.co/docs/transformers/main_classes/model
- Hugging Face chat templates, generation prompts and special-token handling:
  https://huggingface.co/docs/transformers/chat_templating
- Hugging Face Qwen2 source, sequential residual additions and final normalization:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
- Hugging Face Mixtral documentation, sparse experts and resident parameter requirements:
  https://huggingface.co/docs/transformers/model_doc/mixtral
- PyTorch module hooks and handle removal:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
- PyTorch safetensors-compatible CPU/GPU tensor operations and model inference APIs are used directly; no GPU benchmark is inferred from documentation.

The uploaded 05–09 experiment files are separately hashed in `validation/legacy_replay/audit.json`. The original reports' “reconstructed/redundant” labels are preserved as historical interpretations, not adopted as independently established conclusions.
