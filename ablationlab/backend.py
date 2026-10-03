"""PyTorch/Hugging Face backend. Only text decoders with validated residual sites."""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import hashlib
import inspect
import time
import torch
from .adapters import Adapter, hidden_of
from .directions import DirectionBundle
from .interventions import Intervention, intervention_hooks
from .util import LabError, Unsupported, digest

def cached_hf_revision(model_id: str, revision: str) -> str | None:
    """Recover the immutable Hub commit when Transformers omits _commit_hash."""
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(model_id, "config.json", revision=revision)
    if not isinstance(cached, str):
        return None
    path = Path(cached)
    for parent in path.parents:
        if parent.parent.name == "snapshots" and len(parent.name) == 40:
            try:
                int(parent.name, 16)
            except ValueError:
                continue
            return parent.name
    return None

class Backend:
    def __init__(self, config: dict):
        self.config=config
        mc=config["model"]
        self.is_toy=mc["id"].startswith("toy:") or (Path(mc["id"])/"toy_model.json").exists()
        device=mc["device"]
        if device=="auto": device="cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise LabError("CUDA requested but unavailable. Select a CUDA PyTorch build/GPU runtime, or explicitly set model.device=cpu. TPU is not CUDA.")
        dtype=mc["dtype"]
        if dtype=="auto": dtype="bf16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else "fp16" if device.startswith("cuda") else "fp32"
        if device=="cpu" and dtype=="fp16": raise LabError("CPU fp16 decoding is not enabled; use fp32")
        self.dtype={"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}[dtype]
        self.device=torch.device(device)
        if self.is_toy:
            from .toy import TinyLM, ByteTokenizer
            self.model=TinyLM.load(mc["id"]) if (Path(mc["id"])/"toy_model.json").exists() else TinyLM(moe=mc["id"]=="toy:moe")
            self.tokenizer=ByteTokenizer()
            self.model.to(device=self.device,dtype=self.dtype)
        else:
            try:
                import transformers
                from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
            except ImportError as e:
                raise LabError("Install inference dependencies: pip install -e '.[hf]'") from e
            hfconfig=AutoConfig.from_pretrained(mc["id"],revision=mc["revision"],local_files_only=mc["local_files_only"],trust_remote_code=mc["trust_remote_code"])
            if getattr(hfconfig,"is_encoder_decoder",False):
                raise Unsupported("Encoder-decoder architecture requires a separate backend")
            resolved_hf_revision=None
            if self.config.get("identity_schema",1)>=2 and not Path(mc["id"]).is_dir():
                resolved_hf_revision=getattr(hfconfig,"_commit_hash",None) or cached_hf_revision(mc["id"],mc["revision"])
                if not resolved_hf_revision:
                    raise LabError("Could not resolve the model's immutable Hugging Face commit; use a pinned revision or local checkpoint")
            load_revision=resolved_hf_revision or mc["revision"]
            self.tokenizer=AutoTokenizer.from_pretrained(mc["id"],revision=load_revision,local_files_only=mc["local_files_only"],trust_remote_code=mc["trust_remote_code"])
            major_minor=tuple(int(x) for x in transformers.__version__.split(".")[:2])
            kw={"dtype" if major_minor>=(4,56) else "torch_dtype": self.dtype,
                "revision":load_revision,"local_files_only":mc["local_files_only"],
                "trust_remote_code":mc["trust_remote_code"],"attn_implementation":mc["attention"]}
            if mc["quantization"]!="none":
                if not device.startswith("cuda"): raise Unsupported("Quantized loading currently requires CUDA")
                from transformers import BitsAndBytesConfig
                kw["quantization_config"]=BitsAndBytesConfig(load_in_4bit=mc["quantization"]=="4bit",load_in_8bit=mc["quantization"]=="8bit",bnb_4bit_compute_dtype=self.dtype)
            kw["device_map"]="auto" if mc["allow_offload"] else {"":str(self.device)}
            # Do not catch every exception and silently reload/fallback on CPU or another kernel.
            self.model=AutoModelForCausalLM.from_pretrained(mc["id"],**kw)
            if getattr(self.model.config,"is_encoder_decoder",False):
                raise Unsupported("Encoder-decoder models need a separate backend")
            self.device=self.model.get_input_embeddings().weight.device
        torch.manual_seed(config["generation"]["seed"])
        self.model.eval()
        self.tokenizer.padding_side="left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None: raise Unsupported("Tokenizer has neither EOS nor padding token")
            self.tokenizer.pad_token=self.tokenizer.eos_token
        eos=self.model.generation_config.eos_token_id
        if eos is None: eos=self.tokenizer.eos_token_id
        self.eos=set([eos] if isinstance(eos,int) else eos or [])
        if not self.eos: raise Unsupported("No EOS ids configured; cannot distinguish stopping from truncation")
        self.adapter=Adapter(self.model,mc["adapter"])
        probe=self.build_inputs([[{"role":"user","content":"Hello"}]])
        probe=self.forward_inputs(probe)
        self.capabilities=self.adapter.probe(lambda:self.model(**probe,use_cache=False))
        resolved_revision=(resolved_hf_revision if not self.is_toy else None) or getattr(self.model.config,"_commit_hash",None)
        self.capabilities.update(model_id=mc["id"],dtype=str(self.dtype),device=str(self.device),
                                 toy_fixture=self.is_toy,parameter_count=sum(p.numel() for p in self.model.parameters()),
                                 parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()),
                                 resolved_revision=resolved_revision)
        self.identity={"model":mc,"resolved_revision":self.capabilities["resolved_revision"],
                       "dtype":str(self.dtype),"template":digest(getattr(self.tokenizer,"chat_template",None)),
                       "parameters":self.capabilities["parameter_count"],"backend_version":1,
                       "torch_version":torch.__version__,
                       "cuda_runtime":torch.version.cuda,
                       "accelerator_name":torch.cuda.get_device_name(self.device) if self.device.type=="cuda" else "cpu",
                       "transformers_version":None if self.is_toy else transformers.__version__}
        if (Path(mc["id"])).is_dir():
            from .util import file_hash
            # Local checkpoints have no HF revision; fingerprint files rather than trusting the folder name.
            self.identity["local_files"]={str(p.relative_to(mc["id"])):file_hash(p) for p in Path(mc["id"]).rglob("*") if p.suffix in {".safetensors",".bin",".json"}}

    def render(self,msgs):
        if getattr(self.tokenizer,"chat_template",None):
            return self.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        if self.config["model"]["adapter"].get("plain_text",False):
            return "\n".join(m["content"] for m in msgs)
        raise Unsupported("No chat template; explicitly set model.adapter.plain_text=true for a base text model")

    def build_inputs(self, conversations):
        texts=[self.render(x) for x in conversations]
        batch=self.tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=False)
        if batch["input_ids"].shape[1]>self.config["generation"]["max_input_tokens"]:
            raise LabError("Prompt exceeds max_input_tokens; not silently truncating paired conditions")
        return {k:v.to(self.device) for k,v in batch.items() if k in {"input_ids","attention_mask"}}

    def forward_inputs(self, batch):
        """Match generation's mask-derived positions for direct padded forward passes.

        generate() manages its own changing positions; do not pass static prefill
        positions into that loop. Models without a position_ids argument retain
        their documented attention-mask behavior.
        """
        values = dict(batch)
        if "position_ids" in inspect.signature(self.model.forward).parameters:
            positions = values["attention_mask"].long().cumsum(-1) - 1
            values["position_ids"] = positions.masked_fill(values["attention_mask"] == 0, 0)
        return values

    def _validate_bundle(self,bundle):
        if tuple(bundle.basis.shape[::2]) != (len(self.adapter.blocks),self.adapter.hidden_size):
            raise LabError("Direction shape does not match model")
        expected=bundle.metadata.get("model_identity")
        if expected is not None and expected!=digest(self.identity):
            raise LabError("Direction model identity/revision/dtype/template does not match this backend")

    @torch.inference_mode()
    def generate(self, records, bundle=None, spec=None, max_new_tokens=None):
        if bundle is not None: self._validate_bundle(bundle)
        batch=self.build_inputs([r["neutral"] for r in records])
        limit=max_new_tokens or self.config["generation"]["max_new_tokens"]
        start=time.perf_counter()
        with intervention_hooks(self.adapter,bundle,spec,batch["attention_mask"]):
            ids=self.model.generate(**batch,max_new_tokens=limit,do_sample=False,num_beams=1,
                    num_return_sequences=1,use_cache=True,pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=sorted(self.eos),repetition_penalty=1.0,renormalize_logits=False)
        if self.device.type=="cuda": torch.cuda.synchronize(self.device)
        elapsed=time.perf_counter()-start
        generated=ids[:,batch["input_ids"].shape[1]:].detach().cpu().tolist()
        results=[]
        for row in generated:
            first=next((i for i,x in enumerate(row) if x in self.eos),None)
            content=row if first is None else row[:first]
            results.append({"text":self.tokenizer.decode(content,skip_special_tokens=True),
                            "token_ids":content,"tokens":len(content),"stopped_eos":first is not None,
                            "hit_limit":first is None and len(row)>=limit,
                            "termination":"eos" if first is not None else "length" if len(row)>=limit else "other",
                            "batch_size":len(records),"batch_seconds":elapsed})
        return results

    @torch.inference_mode()
    def capture(self, conversations):
        batch=self.build_inputs(conversations)
        values={}; handles=[]
        def make_hook(l):
            def hook(module,args,out): values[l]=hidden_of(out)[:,-1].detach().float().clone()
            return hook
        try:
            for l,b in enumerate(self.adapter.blocks): handles.append(b.register_forward_hook(make_hook(l)))
            self.model(**self.forward_inputs(batch),use_cache=False)
        finally:
            for h in handles:h.remove()
        # [B,L,D], raw block outputs, including last block before final norm.
        return torch.stack([values[l].cpu() for l in range(len(self.adapter.blocks))],dim=1)

    @torch.inference_mode()
    def trace(self, record, bundle, spec=None, prefix_ids=None):
        self._validate_bundle(bundle)
        batch=self.build_inputs([record["neutral"]])
        if prefix_ids:
            continuation=torch.tensor([prefix_ids],device=self.device,dtype=torch.long)
            batch["input_ids"]=torch.cat((batch["input_ids"],continuation),1)
            batch["attention_mask"]=torch.ones_like(batch["input_ids"])
        if batch["input_ids"].shape[1]>self.config["generation"]["max_input_tokens"]:
            raise LabError("Trace prefix exceeds max_input_tokens")
        values={}; handles=[]
        def make_hook(l):
            def hook(module,args,out):
                x=hidden_of(out)[:,-1].float()
                q=bundle.basis[l].to(x.device)
                values[l]=(x @ q.T).detach()
            return hook
        # Install interventions BEFORE recorders so recorded source states are post-edit.
        from dataclasses import replace
        trace_spec=replace(spec,phase="both") if spec is not None else None
        with intervention_hooks(self.adapter,bundle,trace_spec,batch["attention_mask"]):
            try:
                for l,b in enumerate(self.adapter.blocks): handles.append(b.register_forward_hook(make_hook(l)))
                self.model(**self.forward_inputs(batch),use_cache=False)
            finally:
                for h in handles:h.remove()
        return torch.stack([values[l][0].cpu() for l in range(len(self.adapter.blocks))])

    @torch.inference_mode()
    def reference_nll(self,record,reference_ids,bundle=None,spec=None):
        """Cached teacher forcing on FIXED tokens. Tests distribution drift, not factuality.

        One forward per reference token preserves last-token/phase semantics. A single
        teacher-forced full-sequence pass would incorrectly intervene only on its last token.
        """
        if not reference_ids:return None
        batch=self.build_inputs([record["neutral"]])
        if batch["input_ids"].shape[1]+len(reference_ids)>self.config["generation"]["max_input_tokens"]:
            raise LabError("Reference sequence exceeds max_input_tokens")
        losses=[]
        with intervention_hooks(self.adapter,bundle,spec,batch["attention_mask"]):
            output=self.model(**self.forward_inputs(batch),use_cache=True)
            mask=batch["attention_mask"]
            for i,token in enumerate(reference_ids):
                losses.append(-torch.log_softmax(output.logits[:,-1].float(),dim=-1)[0,int(token)])
                if i+1==len(reference_ids):break
                past=getattr(output,"past_key_values",None)
                if past is None:raise Unsupported("Reference-NLL evaluation requires a supported autoregressive cache")
                mask=torch.cat((mask,torch.ones((1,1),dtype=mask.dtype,device=mask.device)),1)
                output=self.model(input_ids=torch.tensor([[int(token)]],device=self.device),
                                  attention_mask=mask,past_key_values=past,use_cache=True)
        return float(torch.stack(losses).mean().cpu())
