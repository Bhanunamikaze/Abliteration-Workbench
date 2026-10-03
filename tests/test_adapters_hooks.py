import pytest
import torch
from ablationlab.toy import TinyLM
from ablationlab.adapters import Adapter,hidden_of,replace_hidden
from ablationlab.directions import DirectionBundle
from ablationlab.interventions import Intervention,intervention_hooks
from ablationlab.backend import Backend
from ablationlab.util import Unsupported,LabError

def bundle_for(adapter):
    n,d=len(adapter.blocks),adapter.hidden_size
    q=torch.randn(n,d);q=q/q.norm(dim=-1,keepdim=True)
    return DirectionBundle(q[:,None],torch.zeros(n,d),torch.zeros(n,d),torch.ones(n),{}).validate()

@pytest.mark.parametrize("moe",[False,True])
def test_adapter_real_torch_dense_and_moe(moe):
    m=TinyLM(moe=moe);a=Adapter(m)
    ids=torch.tensor([[8,9,10]]);mask=torch.ones_like(ids)
    cap=a.probe(lambda:m(ids,attention_mask=mask))
    assert cap["hidden_size"]==24 and len(a.validated)==12
    assert bool(a.moe_layers)==moe
    assert a.get(1,"mlp").exportable is not moe

@pytest.mark.parametrize("operation",["steer","ablate"])
@pytest.mark.parametrize("scope",["last","all"])
@pytest.mark.parametrize("site",["residual","attention","mlp"])
def test_hooks_scoped_and_cleaned(toy_config,operation,scope,site):
    b=Backend(toy_config);bundle=bundle_for(b.adapter)
    inputs=b.build_inputs([[{"role":"user","content":"test"}]])
    before=b.model(**inputs).logits.detach()
    spec=Intervention((1,),site,operation,.7,"zero",scope,"both")
    with intervention_hooks(b.adapter,bundle,spec,inputs["attention_mask"]):
        changed=b.model(**inputs).logits.detach()
    after=b.model(**inputs).logits.detach()
    assert torch.equal(before,after)
    assert not torch.equal(before,changed)
    assert len(b.adapter.get(1,site).module._forward_hooks)==0

def test_hook_cleanup_on_failure(toy_config):
    b=Backend(toy_config);bundle=bundle_for(b.adapter)
    with pytest.raises(RuntimeError,match="injected"):
        with intervention_hooks(b.adapter,bundle,Intervention((1,)),torch.ones(1,3)):
            raise RuntimeError("injected error")
    assert not b.adapter.blocks[1]._forward_hooks

def test_negative_centroid_writer_rejected(toy_config):
    b=Backend(toy_config)
    with pytest.raises(Unsupported):
        with intervention_hooks(b.adapter,bundle_for(b.adapter),Intervention((1,),"mlp",reference="negative"),torch.ones(1,3)):pass

def test_unknown_architecture_is_not_guessed():
    with pytest.raises(Unsupported):Adapter(torch.nn.Linear(3,4))

def test_output_types():
    t=torch.ones(1,2,3)
    for obj in [t,(t,"cache"),[t,"cache"],{"hidden_states":t,"other":1}]:
        assert hidden_of(obj) is t
        assert torch.equal(hidden_of(replace_hidden(obj,t+1)),t+1)

def test_adapter_override_path():
    m=TinyLM();wrapper=torch.nn.Module();wrapper.stack=m.model.layers
    a=Adapter(wrapper,{"layers":"stack","attention":"self_attn.o_proj","mlp":"mlp.down_proj"})
    assert len(a.blocks)==4

def test_cuda_not_silently_cpu(toy_config,monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    toy_config["model"]["device"]="cuda"
    with pytest.raises(LabError,match="CUDA requested"):Backend(toy_config)

def test_generation_and_cache_sequence(toy_config):
    b=Backend(toy_config)
    records=[{"neutral":[{"role":"user","content":"a"}]},{"neutral":[{"role":"user","content":"longer input"}]}]
    result=b.generate(records,max_new_tokens=5)
    assert len(result)==2
    assert all(0<=r["tokens"]<=5 for r in result)
    assert all(r["termination"] in {"eos","length"} for r in result)

def test_split_writer_hook_cleanup(toy_config):
    b=Backend(toy_config);bundle=bundle_for(b.adapter)
    inputs=b.build_inputs([[{"role":"user","content":"x"}]])
    with intervention_hooks(b.adapter,bundle,Intervention((1,),"both","steer",.7),inputs["attention_mask"]):
        b.model(**inputs)
    assert not b.adapter.get(1,"attention").module._forward_hooks
    assert not b.adapter.get(1,"mlp").module._forward_hooks
