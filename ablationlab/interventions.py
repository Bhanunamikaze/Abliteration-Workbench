"""Scoped runtime hooks with FP32 math and exact zero-strength no-op."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import torch
from .adapters import hidden_of, replace_hidden
from .util import LabError, Unsupported

@dataclass(frozen=True)
class Intervention:
    layers: tuple[int,...]
    site: str = "residual"
    operation: str = "ablate"    # steer, ablate
    strength: float = 1.0
    reference: str = "zero"      # zero, negative centroid
    scope: str = "last"          # last, all valid current positions
    phase: str = "both"          # prefill, decode, both
    control_seed: int | None = None

    def serial(self): return asdict(self)


def transform(h: torch.Tensor, basis: torch.Tensor, strength: float,
              operation: str, scale: float = 1, center: torch.Tensor | None = None) -> torch.Tensor:
    """Rows of basis must be orthonormal. Input may be [...,hidden]."""
    if strength == 0: return h
    if h.shape[-1] != basis.shape[-1]: raise LabError("Activation / basis dimension mismatch")
    q=basis.to(h.device,dtype=torch.float32)
    x=h.float()
    if operation == "steer":
        y=x + float(strength)*float(scale)*q[0]
    elif operation == "ablate":
        if not 0 <= strength <= 1: raise LabError("Ablation strength must be in [0,1]")
        origin=torch.zeros_like(q[0]) if center is None else center.to(h.device,dtype=torch.float32)
        y=x - float(strength)*((x-origin) @ q.T) @ q
    else: raise LabError(f"Unknown operation {operation}")
    return y.to(h.dtype)


@contextmanager
def intervention_hooks(adapter, bundle, spec: Intervention | None, attention_mask: torch.Tensor):
    if spec is None or spec.strength==0:
        yield
        return
    if any(l in bundle.metadata.get("excluded_layers",[]) for l in spec.layers):
        raise Unsupported("Requested legacy layer has unverified capture coordinates; recapture raw block activations first")
    if spec.site=="both" and spec.operation!="steer":
        raise Unsupported("Split-writer mode is defined for additive steering only")
    if spec.reference == "negative" and spec.site != "residual":
        raise Unsupported("Negative residual centroid is NOT calibrated for writer outputs; use zero for writers")
    if spec.reference == "negative" and not bundle.metadata.get("calibrated",True):
        raise LabError("Negative reference requires captured class centroids; legacy directions are uncalibrated")
    effective=bundle.random_control(spec.control_seed) if spec.control_seed is not None else bundle
    handles=[]
    def factory(layer,site_name,fraction=1.0):
        calls=0
        constant_cache={}
        # Place tiny constants once per hook installation; no host sync every token.
        site=adapter.get(layer,site_name)
        def hook(module,args,out):
            nonlocal calls
            is_prefill=calls==0
            calls+=1
            if spec.phase=="prefill" and not is_prefill: return out
            if spec.phase=="decode" and is_prefill: return out
            h=hidden_of(out)
            if h.ndim!=3 or h.shape[0]!=attention_mask.shape[0]:
                raise Unsupported("Runtime site is not batch/sequence aligned; expert routes require specialized adapters")
            if h.device not in constant_cache:
                constant_cache[h.device]=(effective.basis[layer].to(h.device,dtype=torch.float32),
                                         effective.negative[layer].to(h.device,dtype=torch.float32) if spec.reference=="negative" else None)
            q,center=constant_cache[h.device]
            if spec.scope=="last":
                edited=h.clone()
                edited[:,-1]=transform(h[:,-1],q,spec.strength*fraction,spec.operation,
                                       float(effective.scale[layer]),center)
            else:
                mask=(attention_mask if is_prefill else torch.ones(h.shape[:2],device=h.device)).to(h.device).bool()
                if mask.shape!=h.shape[:2]: raise Unsupported("Prefill mask is incompatible with module sequence layout")
                edited=h.clone()
                edited[mask]=transform(h[mask],q,spec.strength*fraction,spec.operation,
                                      float(effective.scale[layer]),center)
            return replace_hidden(out,edited)
        return hook
    try:
        actual_sites=[("attention",.5),("mlp",.5)] if spec.site=="both" else [(spec.site,1.0)]
        for l in spec.layers:
            for site_name,fraction in actual_sites:
                if (l,site_name) not in adapter.validated: raise Unsupported(f"Unvalidated site: {l}/{site_name}")
                handles.append(adapter.get(l,site_name).module.register_forward_hook(factory(l,site_name,fraction)))
        yield
    finally:
        for handle in handles: handle.remove()
