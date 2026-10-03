"""Offline optional tests with REAL randomly initialized Hugging Face models.
No downloaded weights. These test architecture plumbing, not model quality.
Run after installing .[hf]: pytest -q -m hf
"""
import pytest
import torch
transformers=pytest.importorskip("transformers",reason="Optional HF package is not installed in this environment")
from ablationlab.adapters import Adapter
from ablationlab.directions import DirectionBundle
from ablationlab.interventions import Intervention,intervention_hooks

@pytest.mark.hf
@pytest.mark.parametrize("family",["qwen2","llama","mistral","gpt2","gpt_neox","mixtral","qwen2_moe"])
def test_tiny_hf_architecture(family):
    if family=="qwen2":
        cfg=transformers.Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    elif family=="llama":
        cfg=transformers.LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    elif family=="mistral":
        cfg=transformers.MistralConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    elif family=="gpt2":
        cfg=transformers.GPT2Config(vocab_size=64,n_embd=32,n_layer=2,n_head=4,n_positions=64)
    elif family=="gpt_neox":
        cfg=transformers.GPTNeoXConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4)
    elif family=="mixtral":
        cfg=transformers.MixtralConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,num_local_experts=3,num_experts_per_tok=2)
    else:
        cfg=transformers.Qwen2MoeConfig(vocab_size=64,hidden_size=32,intermediate_size=64,moe_intermediate_size=24,shared_expert_intermediate_size=24,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,num_experts=3,num_experts_per_tok=2)
    cfg._attn_implementation="eager"
    m=transformers.AutoModelForCausalLM.from_config(cfg).eval()
    ids=torch.tensor([[3,4,5,6]])
    mask=torch.ones_like(ids)
    adapter=Adapter(m)
    adapter.probe(lambda:m(input_ids=ids,attention_mask=mask,use_cache=False))
    assert adapter.hidden_size==32 and len(adapter.blocks)==2
    assert all((l,"residual") in adapter.validated for l in range(2))
    q=torch.eye(32)[0].expand(2,1,32).clone()
    bundle=DirectionBundle(q,torch.zeros(2,32),torch.zeros(2,32),torch.ones(2),{}).validate()
    with torch.inference_mode():before=m(ids,attention_mask=mask).logits
    with intervention_hooks(adapter,bundle,Intervention((0,),strength=1),mask),torch.inference_mode():
        changed=m(ids,attention_mask=mask).logits
    with torch.inference_mode():after=m(ids,attention_mask=mask).logits
    assert torch.equal(before,after)
    assert not torch.equal(before,changed)
