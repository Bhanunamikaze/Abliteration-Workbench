"""Tiny REAL PyTorch decoders for offline mechanical tests, NOT trained language models.
These fixtures exercise attention caches, dense/MoE blocks, hooks and saved checkpoints.
Their generated text and behavioral metrics have no semantic research meaning.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import math
import torch
from torch import nn
from safetensors.torch import save_file, load_file

class ByteTokenizer:
    pad_token_id=0
    eos_token_id=1
    bos_token_id=2
    pad_token="<pad>"
    eos_token="<eos>"
    chat_template="toy-role-lines-v1"
    padding_side="left"
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True,**kwargs):
        text="".join(f"{m['role']}: {m['content']}\n" for m in messages)
        text += "assistant: " if add_generation_prompt else ""
        return self.encode(text) if tokenize else text
    def encode(self,text,add_special_tokens=False):
        return [b+3 for b in text.encode("utf-8")]
    def __call__(self,texts,return_tensors="pt",padding=True,add_special_tokens=False,**kwargs):
        if isinstance(texts,str): texts=[texts]
        rows=[self.encode(t) for t in texts]
        width=max(map(len,rows))
        return {"input_ids":torch.tensor([[0]*(width-len(r))+r for r in rows]),
                "attention_mask":torch.tensor([[0]*(width-len(r))+[1]*len(r) for r in rows])}
    def decode(self,ids,skip_special_tokens=True):
        values=ids.tolist() if torch.is_tensor(ids) else ids
        return bytes(x-3 for x in values if 3<=x<259).decode("utf-8",errors="replace")
    def save_pretrained(self,path):
        Path(path).mkdir(parents=True,exist_ok=True)
        (Path(path)/"toy_tokenizer.json").write_text(json.dumps({"type":"byte-v1"}))

class TinyAttention(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.q_proj=nn.Linear(d,d,bias=False); self.k_proj=nn.Linear(d,d,bias=False)
        self.v_proj=nn.Linear(d,d,bias=False); self.o_proj=nn.Linear(d,d,bias=True)
    def forward(self,x,mask=None,past=None):
        q,k,v=self.q_proj(x),self.k_proj(x),self.v_proj(x)
        offset=0 if past is None else past[0].shape[1]
        if past is not None: k,v=torch.cat((past[0],k),1),torch.cat((past[1],v),1)
        scores=q @ k.transpose(-1,-2) / math.sqrt(x.shape[-1])
        causal=torch.arange(k.shape[1],device=x.device)[None,:] <= (torch.arange(q.shape[1],device=x.device)[:,None]+offset)
        scores=scores.masked_fill(~causal,-1e4)
        if mask is not None: scores=scores.masked_fill(~mask[:,None,:].bool(),-1e4)
        values=scores.softmax(-1) @ v
        return self.o_proj(values),(k,v)

class TinyMLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.up_proj=nn.Linear(d,2*d,bias=False)
        self.gate_proj=nn.Linear(d,2*d,bias=False)
        self.down_proj=nn.Linear(2*d,d,bias=True)
    def forward(self,x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj(x))

class TinyMoE(nn.Module):
    def __init__(self,d):
        super().__init__(); self.gate=nn.Linear(d,3,bias=False)
        self.experts=nn.ModuleList([TinyMLP(d) for _ in range(3)])
    def forward(self,x):
        logits=self.gate(x)
        top=logits.topk(2,dim=-1).indices
        mask=torch.zeros_like(logits).scatter_(-1,top,1).bool()
        weights=logits.masked_fill(~mask,-1e4).softmax(-1)
        result=sum(expert(x)*weights[...,i,None] for i,expert in enumerate(self.experts))
        return result,logits

class TinyBlock(nn.Module):
    def __init__(self,d,moe=False):
        super().__init__(); self.self_attn=TinyAttention(d)
        self.input_layernorm=nn.LayerNorm(d); self.post_attention_layernorm=nn.LayerNorm(d)
        if moe: self.block_sparse_moe=TinyMoE(d)
        else: self.mlp=TinyMLP(d)
    def forward(self,x,mask=None,past=None):
        a,cache=self.self_attn(self.input_layernorm(x),mask,past)
        x=x+a
        if hasattr(self,"block_sparse_moe"): m,_=self.block_sparse_moe(self.post_attention_layernorm(x))
        else: m=self.mlp(self.post_attention_layernorm(x))
        return x+m,cache

class TinyLM(nn.Module):
    def __init__(self,moe=False,d=24,layers=4,seed=17):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model=nn.Module()
            self.model.embed_tokens=nn.Embedding(259,d)
            self.model.layers=nn.ModuleList([TinyBlock(d,moe) for _ in range(layers)])
            self.model.norm=nn.LayerNorm(d)
            self.lm_head=nn.Linear(d,259,bias=False)
        self.config=SimpleNamespace(hidden_size=d,num_hidden_layers=layers,model_type="ablab_toy_moe" if moe else "ablab_toy_dense",
                                    is_encoder_decoder=False,max_position_embeddings=4096,_commit_hash=None)
        self.generation_config=SimpleNamespace(eos_token_id=1,pad_token_id=0)
        self.toy_args={"moe":moe,"d":d,"layers":layers,"seed":seed}
    def get_input_embeddings(self): return self.model.embed_tokens
    def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=True,**kwargs):
        x=self.model.embed_tokens(input_ids)
        caches=[]
        for i,layer in enumerate(self.model.layers):
            x,cache=layer(x,attention_mask,None if past_key_values is None else past_key_values[i])
            caches.append(cache)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(x)),past_key_values=caches)
    @torch.inference_mode()
    def generate(self,input_ids,attention_mask,max_new_tokens,**kwargs):
        ids=input_ids; mask=attention_mask
        alive=torch.ones(len(ids),device=ids.device,dtype=torch.bool)
        cache=None; current=ids
        for _ in range(max_new_tokens):
            out=self(current,attention_mask=mask,past_key_values=cache)
            cache=out.past_key_values
            nxt=out.logits[:,-1].argmax(-1)
            nxt=torch.where(alive,nxt,torch.zeros_like(nxt))
            ids=torch.cat((ids,nxt[:,None]),1)
            alive=alive & (nxt!=1)
            mask=torch.cat((mask,alive[:,None].to(mask.dtype)),1)
            if not alive.any(): break
            current=nxt[:,None]
        return ids
    def save_pretrained(self,path,**kwargs):
        p=Path(path);p.mkdir(parents=True,exist_ok=True)
        (p/"toy_model.json").write_text(json.dumps(self.toy_args))
        save_file({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()},str(p/"model.safetensors"))
    @classmethod
    def load(cls,path):
        p=Path(path);m=cls(**json.loads((p/"toy_model.json").read_text()))
        m.load_state_dict(load_file(str(p/"model.safetensors")))
        return m
