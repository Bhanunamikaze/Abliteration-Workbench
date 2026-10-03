"""Capability-based residual/writer adapters; never assume every model is Llama."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from .util import Unsupported, LabError

STACKS = ["model.layers", "transformer.h", "gpt_neox.layers", "layers", "transformer.blocks",
          "model.decoder.layers", "language_model.model.layers", "model.language_model.layers"]
ATTENTION = ["self_attn.o_proj", "self_attn.out_proj", "self_attn.dense", "attention.dense", "self_attention.dense",
             "attention.out_proj", "attn.c_proj", "attn.out_proj", "attention.wo"]
DENSE = ["mlp.down_proj", "mlp.dense_4h_to_h", "mlp.fc2", "mlp.c_proj", "feed_forward.w2", "fc2"]
MOE = ["block_sparse_moe", "mlp", "feed_forward"]

def resolve(root, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj

def hidden_of(output):
    if torch.is_tensor(output): return output
    if isinstance(output,(tuple,list)) and output and torch.is_tensor(output[0]): return output[0]
    if isinstance(output,dict) and torch.is_tensor(output.get("hidden_states")): return output["hidden_states"]
    raise Unsupported(f"Unsupported output layout: {type(output).__name__}; implement hidden_of/replace_hidden adapter")

def replace_hidden(output, hidden):
    if torch.is_tensor(output): return hidden
    if isinstance(output,tuple): return (hidden,*output[1:])
    if isinstance(output,list): return [hidden,*output[1:]]
    if isinstance(output,dict): return {**output,"hidden_states":hidden}
    raise Unsupported("Cannot replace module output")

@dataclass
class Site:
    layer: int
    name: str
    path: str
    module: torch.nn.Module
    kind: str
    exportable: bool = False

class Adapter:
    def __init__(self, model: torch.nn.Module, mapping: dict | None = None):
        self.model, mapping = model, mapping or {}
        self.stack_path, self.blocks = None, None
        candidates = [mapping["layers"]] if "layers" in mapping else STACKS
        for path in candidates:
            try:
                value = resolve(model,path)
                if len(value) and all(isinstance(x,torch.nn.Module) for x in value):
                    self.stack_path,self.blocks=path,value
                    break
            except (AttributeError, TypeError, KeyError): continue
        if self.blocks is None:
            raise Unsupported("No supported block stack. Supply model.adapter.layers or add an Adapter implementation.")
        self.sites: dict[tuple[int,str],Site] = {}
        self.moe_layers = []
        self.expert_inventory = []
        for l, block in enumerate(self.blocks):
            self.sites[l,"residual"] = Site(l,"residual",f"{self.stack_path}.{l}",block,"residual")
            self._find(l,block,"attention",[mapping["attention"]] if "attention" in mapping else ATTENTION,"writer")
            if "mlp" in mapping:
                self._find(l,block,"mlp",[mapping["mlp"]],"aggregate" if mapping.get("moe") else "writer")
            else:
                is_moe = any("expert" in name.lower() for name,_ in block.named_modules()) or any("expert" in name.lower() for name,_ in block.named_parameters())
                if is_moe:
                    self.moe_layers.append(l)
                    self._find(l,block,"mlp",MOE,"aggregate")
                else: self._find(l,block,"mlp",DENSE,"writer")
            for name,module in block.named_modules():
                if "expert" in name.lower() and (isinstance(module,torch.nn.Linear) or "experts" == name.rsplit(".",1)[-1]):
                    self.expert_inventory.append({"layer":l,"path":f"{self.stack_path}.{l}.{name}","type":type(module).__name__,
                                                  "note":"inventory only; routed token outputs are not batch-aligned residual sites"})
        self.validated = set()
        self.hidden_size = None

    def _find(self,l,block,name,paths,kind):
        for path in paths:
            try:
                module=resolve(block,path)
                if not isinstance(module,torch.nn.Module): continue
                self.sites[l,name]=Site(l,name,f"{self.stack_path}.{l}.{path}",module,kind,
                                        isinstance(module,torch.nn.Linear) and kind=="writer")
                return
            except AttributeError: continue

    def get(self,l,name) -> Site:
        if (l,name) not in self.sites: raise Unsupported(f"Layer {l} has no supported {name} site")
        return self.sites[l,name]

    def probe(self, forward):
        """Validate actual outputs; a name match by itself never grants runtime support."""
        observations,handles={},[]
        def hook_for(key):
            def hook(module,args,out):
                try:
                    h=hidden_of(out)
                    observations[key]=tuple(h.shape)
                except Unsupported:
                    observations[key]=None
            return hook
        try:
            for key,site in self.sites.items(): handles.append(site.module.register_forward_hook(hook_for(key)))
            with torch.inference_mode(): forward()
        finally:
            for h in handles: h.remove()
        residual = [observations.get((l,"residual")) for l in range(len(self.blocks))]
        if any(s is None or len(s)!=3 for s in residual):
            raise Unsupported("Block outputs are not uniform [batch,sequence,hidden] tensors")
        dims={s[-1] for s in residual}
        if len(dims)!=1: raise Unsupported("Variable-width residual streams require a custom adapter")
        self.hidden_size=next(iter(dims))
        self.validated={key for key,shape in observations.items() if shape is not None and len(shape)==3 and shape[-1]==self.hidden_size}
        return self.describe()

    def describe(self):
        return {"stack_path":self.stack_path,"num_layers":len(self.blocks),"hidden_size":self.hidden_size,
                "moe_layers":self.moe_layers,"expert_inventory":self.expert_inventory,
                "sites":[{"layer":s.layer,"name":s.name,"path":s.path,"kind":s.kind,
                          "runtime_validated":k in self.validated,"exportable_linear":s.exportable}
                         for k,s in self.sites.items()],
                "limitations":["MoE support refers to block/aggregate-mixture output, not arbitrary routed/fused expert surgery",
                               "Names and output shapes do not prove a component's natural behavioral role",
                               "last-block hooks are before final model norm; directions use that same site"]}


def export_writers(site: Site) -> list[tuple[str,torch.nn.Linear]]:
    """Known unfused scalar-mixture output writers; zero projection commutes with mixing.

    No generic expert tensor guessing. Fused/stacked expert implementations are rejected.
    The exporter additionally checks full-model fixed-input logits before saving.
    """
    if site.kind=="writer" and type(site.module) is torch.nn.Linear:
        return [(site.path,site.module)]
    if site.kind!="aggregate":raise Unsupported(f"Not an exportable writer: {site.path}")
    allowed={"TinyMoE","MixtralSparseMoeBlock","Qwen2MoeSparseMoeBlock"}
    if type(site.module).__name__ not in allowed:
        raise Unsupported(f"Aggregate export not registered for {type(site.module).__name__}; fused experts need an explicit exporter")
    experts=getattr(site.module,"experts",None)
    if not isinstance(experts,torch.nn.ModuleList):
        raise Unsupported("Only explicit ModuleList experts are exportable; stacked/fused expert tensors are unsupported")
    writers=[]
    for i,expert in enumerate(experts):
        found=[]
        for name in ["w2","down_proj"]:
            module=getattr(expert,name,None)
            if type(module) is torch.nn.Linear:found.append((f"{site.path}.experts.{i}.{name}",module))
        if len(found)!=1:raise Unsupported(f"Expert {i} output writer is ambiguous or quantized")
        writers.extend(found)
    shared=getattr(site.module,"shared_expert",None)
    if shared is not None:
        module=getattr(shared,"down_proj",None)
        if type(module) is not torch.nn.Linear:raise Unsupported("Shared expert output is not a plain linear writer")
        writers.append((f"{site.path}.shared_expert.down_proj",module))
    if not writers:raise Unsupported("No recognized expert output writers")
    return writers
