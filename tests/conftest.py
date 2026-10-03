from pathlib import Path
from copy import deepcopy
import json
import pytest
import torch
from ablationlab.config import DEFAULTS, merge
from ablationlab.store import RunStore
from ablationlab.data import load_dataset

@pytest.fixture(autouse=True,scope="session")
def small_cpu_threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)

@pytest.fixture
def toy_config():
    return merge(DEFAULTS,{"model":{"id":"toy:dense","device":"cpu","dtype":"fp32"},
        "generation":{"batch_size":2,"max_new_tokens":6,"refine_max_new_tokens":8,"max_input_tokens":256},
        "discovery":{"min_pairs":2},"search":{"top_k":2,"max_layer_tests":2,"random_controls":1,"ablations":[.5,1.]},
        "trace":{"prefix_tokens":[0,2]},"budget":{"max_generations":2000,"max_seconds":300}})

@pytest.fixture
def dataset_file(tmp_path):
    rows=[]
    for split,qs in [("train",["one","two","three","four"]),("validation",["five","six","seven","eight"]),("test",["nine","ten"]),("control",["eleven"])]:
        for i,q in enumerate(qs):
            r={"id":f"{split}{i}","split":split,"neutral":q}
            if split in {"train","validation"}:r.update(positive=f"{q} detail",negative=f"{q} brief")
            rows.append(r)
    path=tmp_path/"data.json";path.write_text(json.dumps(rows));return path

@pytest.fixture
def store(tmp_path,toy_config,dataset_file):
    rows,audit=load_dataset(dataset_file,2)
    return RunStore.create(tmp_path/"run",toy_config,rows,audit)
