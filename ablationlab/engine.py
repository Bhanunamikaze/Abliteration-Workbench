"""Generation caching, batching, retry and declared scoring."""
from __future__ import annotations
import gc
import torch
from .util import digest, LabError, now
from .metrics import Scorer, lexical_checks

class Engine:
    def __init__(self,store):
        self.store=store; self._backend=None
        self.scorer=Scorer(store.config["behavior"])
    @property
    def backend(self):
        if self._backend is None:
            from .backend import Backend
            plugin=self.store.config["model"].get("backend_plugin")
            if plugin:
                import importlib
                mod,name=plugin.split(":",1)
                self._backend=getattr(importlib.import_module(mod),name)(self.store.config)
            else:
                self._backend=Backend(self.store.config)
            if self.store.completed("inspect"):
                expected = self.store.result("inspect").get("model_identity")
                if expected is not None and expected != digest(self._backend.identity):
                    raise LabError("Model/runtime identity changed since inspect. Start a new run or fork model settings; do not reuse mixed-revision activation batches.")
            self.store.event("backend_loaded",identity=self._backend.identity,
                             capabilities=self._backend.capabilities)
        return self._backend

    def _generate_retry(self,records,bundle,spec,limit):
        try:
            return self.backend.generate(records,bundle,spec,limit)
        except torch.OutOfMemoryError:
            if len(records)==1:
                raise LabError("OOM at batch size 1. Reduce context/model size or explicitly enable quantization/offload.")
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            cut=len(records)//2
            self.store.event("oom_batch_split",old_batch=len(records),new_batch=cut)
            return self._generate_retry(records[:cut],bundle,spec,limit)+self._generate_retry(records[cut:],bundle,spec,limit)

    def generate(self,records,bundle=None,spec=None,max_new_tokens=None):
        b=self.backend
        config=self.store.config
        signature={"backend":b.identity,"generation":config["generation"],
                   "max_new_tokens":max_new_tokens or config["generation"]["max_new_tokens"],
                   "directions":bundle.fingerprint() if bundle else None,
                   "intervention":spec.serial() if spec else None}
        batch_size=config["generation"]["batch_size"]
        all_results=[]
        for start in range(0,len(records),batch_size):
            chunk=records[start:start+batch_size]
            # Include batch composition: floating-point greedy generation can differ by batching.
            group=digest([r["neutral"] for r in chunk])
            keys=[digest({**signature,"record":r,"batch":group}) for r in chunk]
            if all(k in self.store.cache for k in keys):
                responses=[self.store.cache[k] for k in keys]
            else:
                self.store.check_budget(len(chunk))
                responses=self._generate_retry(chunk,bundle,spec,signature["max_new_tokens"])
                for r,response,key in zip(chunk,responses,keys):
                    response.update(job_key=key,id=r["id"],split=r["split"],messages=r["neutral"],
                                    request=signature,created_at=now())
                    self.store.put_generation(response)
            for r,response in zip(chunk,responses):
                row={**response,"score":self.scorer(response["text"],response["tokens"],r),
                     **lexical_checks(response["text"],r)}
                all_results.append(row)
            print(f"  generated/cached {min(start+batch_size,len(records))}/{len(records)}",flush=True)
        return all_results
