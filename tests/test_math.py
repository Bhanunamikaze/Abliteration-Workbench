import numpy as np
import pytest
import torch
from ablationlab.interventions import transform
from ablationlab.directions import estimate,DirectionBundle,validation_metrics
from ablationlab.metrics import paired_stats,projection_diagnostics,corr,Scorer,repetition
from ablationlab.config import DEFAULTS
from ablationlab.util import LabError

@pytest.mark.parametrize("rank",[1,2,4])
@pytest.mark.parametrize("strength",[0,.25,.5,1])
def test_projection_math(rank,strength):
    torch.manual_seed(4)
    q=torch.linalg.qr(torch.randn(12,rank)).Q.T
    x=torch.randn(3,5,12)
    y=transform(x,q,strength,"ablate")
    assert torch.allclose(y@q.T,(1-strength)*(x@q.T),atol=2e-6)
    if strength==0:assert y is x

@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16])
def test_noop_exact(dtype):
    x=torch.randn(2,12).to(dtype);q=torch.eye(12)[:1]
    assert transform(x,q,0,"ablate") is x

def test_zero_not_negative():
    x=torch.tensor([[-4.,2.]])
    q=torch.tensor([[1.,0.]])
    y=transform(x,q,1,"ablate")
    assert y[0,0]==0 and y[0,0]>x[0,0]
    z=transform(x,q,1,"ablate",center=torch.tensor([-7.,0.]))
    assert z[0,0]==-7

def test_direction_roundtrip_and_holdout(tmp_path):
    torch.manual_seed(2)
    n=torch.randn(20,3,8);p=n+torch.tensor([1.,2.,3.,4.,0.,0.,0.,0.])
    bundle=estimate(p,n)
    bundle.save(tmp_path);loaded=DirectionBundle.load(tmp_path)
    assert torch.equal(bundle.basis,loaded.basis)
    assert all(x["validation_margin"]>0 for x in validation_metrics(loaded,p,n))
    assert loaded.fingerprint()==bundle.fingerprint()

def test_svd_subspace():
    torch.manual_seed(2)
    n=torch.randn(16,3,8);p=n+torch.randn_like(n)+2
    bundle=estimate(p,n,3,"mean_svd")
    assert bundle.basis.shape==(3,3,8)
    bundle.validate()

def test_no_signal_rejected():
    x=torch.randn(4,3,8)
    with pytest.raises(LabError,match="Zero contrast"):estimate(x,x)

def test_random_control_orthogonality():
    n=torch.randn(12,3,8);p=n+1
    b=estimate(p,n);r=b.random_control(5)
    assert torch.max(torch.abs(b.basis@r.basis.transpose(-1,-2)))<1e-5

def test_percentage_definitions_differ():
    r=paired_stats([10,100],[20,80])
    assert r["gain_fraction_of_baseline_mean"]==pytest.approx(5/55)
    assert r["mean_per_example_fraction"]==pytest.approx(-.4)

def test_ratios_no_tiny_denominator_inflation():
    d=projection_diagnostics([.001,1.],[1.,1.],.1)
    assert d["ratio_of_mean_abs"]<2.01
    assert d["eligible_fraction"]==.5
    assert d["mean_abs_paired_error"]==pytest.approx(.4995)

def test_sign_flip_not_recovery():
    d=projection_diagnostics([10,10],[-10,-10])
    assert d["ratio_of_mean_abs"]==1
    assert d["sign_agreement_eligible"]==0
    assert d["mean_abs_paired_error"]==20

def test_metric_not_behavior_name():
    c=dict(DEFAULTS["behavior"],name="refusal",metric="regex",pattern="decline")
    scorer=Scorer(c)
    assert scorer("I decline.",8,{})==1
    assert scorer("This answer is not a refusal",8,{})==0
    assert Scorer(dict(c,metric="tokens"))("I decline",8,{})==8

@pytest.mark.parametrize("text,expected",[("{}",1),("[1,2]",1),("JSON: {}",0),("oops",0)])
def test_json_scorer(text,expected):
    assert Scorer(dict(DEFAULTS["behavior"],metric="json_valid"))(text,5,{})==expected

def test_exact_and_contains():
    assert Scorer(dict(DEFAULTS["behavior"],metric="exact"))(" 42 ",1,{"expected":"42"})==1
    assert Scorer(dict(DEFAULTS["behavior"],metric="contains"))("SYN then ACK",4,{"required_terms":["SYN","ACK","FIN"]})==pytest.approx(2/3)

def test_custom_scorer_preserves_evidence_and_scalar_api():
    scorer=Scorer(dict(DEFAULTS["behavior"],metric="custom",plugin="examples.custom_scorer:score"))
    scorer.custom=lambda **kwargs:{"score":1,"details":{"confidence":.7,"review_required":True}}
    assert scorer("Example",1,{})==1.
    assert scorer.evaluate("Example",1,{})=={"score":1.,"score_details":{"confidence":.7,"review_required":True}}
    scorer.custom=lambda **kwargs:.25
    assert scorer.evaluate("Example",1,{})=={"score":.25}

@pytest.mark.parametrize("result",[
    {"score":1}, {"score":1,"details":[]},
    {"score":1,"details":{"confidence":float("nan")}},
    {"score":1,"details":{"object":object()}},
    {"score":float("inf"),"details":{}},
])
def test_custom_scorer_rejects_invalid_evidence(result):
    scorer=Scorer(dict(DEFAULTS["behavior"],metric="custom",plugin="examples.custom_scorer:score"))
    scorer.custom=lambda **kwargs:result
    with pytest.raises(LabError):scorer.evaluate("Example",1,{})

@pytest.mark.parametrize("text,probability,label,reasons",[
    ("A response",.99,1,[]), ("An ambiguous response",.49,0,["low_confidence"]),
    ("  ",.99,1,["empty_response"]), ("x"*600,.99,1,["classifier_input_truncated"]),
])
def test_local_refusal_scorer_review_flags(monkeypatch,text,probability,label,reasons):
    from types import SimpleNamespace
    from examples import refusal_scorer
    class Tokenizer:
        model_max_length=512
        def __call__(self,text,**kwargs):return {"input_ids":list(range(len(text)))}
    class Model:
        def __call__(self,**kwargs):
            return SimpleNamespace(logits=torch.log(torch.tensor([[1-probability,probability]])))
    monkeypatch.setattr(refusal_scorer,"classifier",lambda:(Tokenizer(),Model(),1))
    result=refusal_scorer.score(text=text,token_count=5,record={"neutral":[{"role":"user","content":"Question"}]})
    assert result["score"]==label
    assert result["details"]["refusal_probability"]==pytest.approx(probability)
    assert result["details"]["review_reasons"]==reasons
    assert result["details"]["status"]=="provisional"

def test_local_refusal_scorer_uses_prompt_context(monkeypatch):
    from types import SimpleNamespace
    from examples import refusal_scorer
    inputs=[]
    class Tokenizer:
        model_max_length=512
        def __call__(self,text,**kwargs):
            inputs.append(text)
            return {"input_ids":[1]}
    class Model:
        def __call__(self,**kwargs):return SimpleNamespace(logits=torch.tensor([[0.,4.]]))
    monkeypatch.setattr(refusal_scorer,"classifier",lambda:(Tokenizer(),Model(),1))
    record={"neutral":[{"role":"user","content":"What does 'cannot' mean?"}]}
    result=refusal_scorer.score(text="It expresses inability.",token_count=4,record=record)
    assert inputs==["[USER]\nWhat does 'cannot' mean?\n\n[ASSISTANT]\nIt expresses inability."]*2
    assert result["details"]["input"]=="prompt_response_exchange"
    missing=refusal_scorer.score(text="An answer",token_count=4,record={})
    assert "missing_prompt_context" in missing["details"]["review_reasons"]

@pytest.mark.parametrize("prompt_label,probability,reason", [
    ("benign", .99, "benign_prompt_predicted_refusal"),
    ("harmful", .01, "harmful_prompt_predicted_compliance"),
])
def test_local_refusal_scorer_reviews_label_disagreements(monkeypatch,prompt_label,probability,reason):
    from types import SimpleNamespace
    from examples import refusal_scorer
    class Tokenizer:
        model_max_length=512
        def __call__(self,text,**kwargs):return {"input_ids":[1]}
    class Model:
        def __call__(self,**kwargs):
            return SimpleNamespace(logits=torch.log(torch.tensor([[1-probability,probability]])))
    monkeypatch.setattr(refusal_scorer,"classifier",lambda:(Tokenizer(),Model(),1))
    record={"prompt_label":prompt_label,"neutral":[{"role":"user","content":"Question"}]}
    result=refusal_scorer.score(text="An answer",token_count=3,record=record)
    assert result["details"]["review_required"] is True
    assert reason in result["details"]["review_reasons"]
