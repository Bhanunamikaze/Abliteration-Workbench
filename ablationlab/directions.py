"""Calibrated FP32 direction/subspace estimation, no model-specific constants."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from .util import LabError, read_json, write_json, digest, save_tensors

@dataclass
class DirectionBundle:
    basis: torch.Tensor                 # [layers, rank, hidden], orthonormal rows
    positive: torch.Tensor              # mean activation at each layer
    negative: torch.Tensor
    scale: torch.Tensor                # train contrast along first direction
    metadata: dict

    def validate(self):
        q = self.basis.float()
        if q.ndim != 3 or not torch.isfinite(q).all(): raise LabError("Invalid direction basis")
        n, rank, d = q.shape
        if self.positive.shape != (n,d) or self.negative.shape != (n,d) or self.scale.shape != (n,):
            raise LabError("Direction calibration shapes do not match")
        if not torch.allclose(q @ q.transpose(-1,-2), torch.eye(rank).expand(n,rank,rank), atol=1e-4):
            raise LabError("Direction basis must be orthonormal")
        if not torch.isfinite(self.positive).all() or not torch.isfinite(self.negative).all():
            raise LabError("Nonfinite direction centroids")
        if not torch.isfinite(self.scale).all() or (self.scale < 0).any(): raise LabError("Invalid direction scales")
        return self

    def save(self, directory: str | Path):
        p = Path(directory); p.mkdir(parents=True, exist_ok=True)
        self.validate()
        save_tensors(p / "directions.safetensors", {"basis": self.basis.float().contiguous(), "positive": self.positive.float().contiguous(),
                   "negative": self.negative.float().contiguous(), "scale": self.scale.float().contiguous()})
        write_json(p / "directions.json", self.metadata)

    @classmethod
    def load(cls, directory: str | Path):
        p = Path(directory)
        t = load_file(str(p / "directions.safetensors"))
        return cls(**t, metadata=read_json(p / "directions.json")).validate()

    def fingerprint(self):
        import hashlib
        h = hashlib.sha256()
        for tensor in (self.basis, self.positive, self.negative, self.scale):
            h.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
        h.update(digest(self.metadata).encode())
        return h.hexdigest()

    def random_control(self, seed: int):
        if self.basis.shape[1]*2 > self.basis.shape[2]:
            raise LabError("Not enough orthogonal dimensions for a same-rank random control")
        g = torch.Generator().manual_seed(seed)
        bases = []
        for q in self.basis:
            z = torch.randn(q.shape[1], q.shape[0], generator=g)
            z = z - q.T @ (q @ z)
            r, _ = torch.linalg.qr(z)
            bases.append(r.T)
        return DirectionBundle(torch.stack(bases), self.positive, self.negative, self.scale,
                               {**self.metadata, "random_control_seed": seed}).validate()


def estimate(train_positive: torch.Tensor, train_negative: torch.Tensor, rank: int = 1,
             method: str = "mean", metadata: dict | None = None) -> DirectionBundle:
    p, n = train_positive.float(), train_negative.float()
    if p.shape != n.shape or p.ndim != 3 or p.shape[0] < 2:
        raise LabError("Activation arrays must be [paired_examples,layers,hidden], >=2 pairs")
    if not torch.isfinite(p).all() or not torch.isfinite(n).all(): raise LabError("Nonfinite captured activations")
    if rank > min(p.shape[0], p.shape[2]-1): raise LabError("Subspace rank exceeds available paired data/dimension")
    pos, neg = p.mean(0), n.mean(0)
    difference = pos-neg
    lengths = difference.norm(dim=-1)
    if (lengths < 1e-8).any():
        raise LabError("Zero contrast at one or more layers; dataset needs a measurable difference")
    qs = []
    for layer, d in enumerate(difference):
        q0 = d / d.norm()
        basis = [q0]
        if rank > 1:
            if method != "mean_svd": raise LabError("rank>1 needs mean_svd")
            residuals = p[:,layer] - n[:,layer]
            residuals -= (residuals @ q0)[:,None] * q0
            _, singular, v = torch.linalg.svd(residuals, full_matrices=False)
            for vector, singular_value in zip(v, singular):
                if singular_value < 1e-7: continue
                for prev in basis: vector = vector - torch.dot(vector, prev)*prev
                if vector.norm() > 1e-7: basis.append(vector / vector.norm())
                if len(basis) == rank: break
            if len(basis) != rank: raise LabError("Data does not support requested subspace rank")
        qs.append(torch.stack(basis))
    return DirectionBundle(torch.stack(qs), pos, neg, lengths,
                           {"capture_site": "raw_block_output", "coordinate": "residual", "method":method,
                            "rank":rank, **(metadata or {})}).validate()


def validation_metrics(bundle: DirectionBundle, positive: torch.Tensor, negative: torch.Tensor) -> list[dict]:
    rows = []
    for l, q in enumerate(bundle.basis):
        p, n = positive[:,l].float() @ q[0], negative[:,l].float() @ q[0]
        margin = float((p-n).mean())
        pooled = float(torch.sqrt((p.var(unbiased=False) + n.var(unbiased=False))/2))
        midpoint = float(((bundle.positive[l] + bundle.negative[l])/2) @ q[0])
        accuracy = float(((p > midpoint).float().mean()+(n < midpoint).float().mean())/2)
        rows.append({"layer":l, "validation_margin":margin, "pooled_std":pooled,
                     "standardized_separation": margin/max(pooled,1e-6),
                     "paired_positive_fraction":float((p>n).float().mean()), "midpoint_accuracy":accuracy,
                     "train_gap":float(bundle.scale[l]), "negative_coordinate":float(bundle.negative[l] @ q[0]),
                     "positive_coordinate":float(bundle.positive[l] @ q[0])})
    return rows
