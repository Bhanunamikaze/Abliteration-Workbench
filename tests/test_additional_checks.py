from copy import deepcopy
import math
import pytest
import torch
from ablationlab.config import validate_config
from ablationlab.util import LabError, BudgetReached, write_json
from ablationlab.store import descendants
from ablationlab.directions import DirectionBundle
from ablationlab.pipeline import Pipeline

@pytest.mark.parametrize('section,key,value',[
 ('generation','batch_size',True),('generation','refine_max_new_tokens',0),
 ('search','strengths',[-.25,.25,.25]),('search','ablations',[.5,.5]),
 ('search','min_effect_fraction',float('nan')),('search','refine_strength_factor',float('inf')),
 ('trace','denominator_floor',0),('trace','denominator_floor',float('nan')),
 ('budget','max_generations',True),('evaluation','reference_tokens',-.5),
 ('evaluation','baseline_include_test',1),
])
def test_strict_numeric_config(toy_config,section,key,value):
    toy_config[section][key]=value
    with pytest.raises(LabError): validate_config(toy_config)

def test_optional_refinement_invalidates_downstream_not_directions():
    affected=descendants('refine')
    assert {'writers','ablate','trace','persistent','evaluate','report'}<=affected
    assert not {'inspect','baseline','directions'} & affected

def test_budget_does_not_double_count_stages(store,monkeypatch):
    import ablationlab.store as module
    store.config['budget']['max_seconds']=100
    store.prior_seconds=20; store.started=100
    state=store.state(); state['seconds_spent']=90
    write_json(store.root/'state.json',state)
    monkeypatch.setattr(module.time,'monotonic',lambda:170)
    store.check_budget() # 20 earlier + 70 current, not 90+70
    monkeypatch.setattr(module.time,'monotonic',lambda:190)
    with pytest.raises(BudgetReached): store.check_budget()

def test_direction_nonfinite_centroid_rejected():
    q=torch.eye(4)[:1].unsqueeze(0)
    bundle=DirectionBundle(q,torch.full((1,4),float('nan')),torch.zeros(1,4),torch.ones(1),{})
    with pytest.raises(LabError,match='Nonfinite'):bundle.validate()

def test_random_subspace_needs_enough_orthogonal_space():
    q=torch.eye(4)[:3].unsqueeze(0)
    bundle=DirectionBundle(q,torch.zeros(1,4),torch.zeros(1,4),torch.ones(1),{}).validate()
    with pytest.raises(LabError,match='orthogonal'): bundle.random_control(12)

@pytest.mark.parametrize('regions',[
    [{'label':'old','layers':[]}],
    [{'label':'old','layers':[1,1]}],
    [{'label':'old','layers':[True]}],
    [{'label':'peaks','layers':[1]}],
])
def test_custom_region_validation(toy_config,regions):
    toy_config['search']['custom_regions']=regions
    with pytest.raises(LabError,match='Custom region'):
        validate_config(toy_config)

@pytest.mark.integration
def test_explicit_persistent_region_uses_declared_layers(store):
    store.config['search']['regions']=[]
    store.config['search']['custom_regions']=[{'label':'old_set','layers':[0,2]}]
    with store.lock():
        Pipeline(store).run_stage('persistent',with_deps=True)
    assert store.result('persistent')['regions']=={'old_set':[0,2]}

@pytest.mark.integration
def test_offline_plots_match_completed_stages(store,tmp_path):
    pytest.importorskip('matplotlib')
    from ablationlab.plotting import plot_run
    p=Pipeline(store)
    store.config['search']['layers']=[1]
    with store.lock():
        p.run_stage('trace',with_deps=True)
        files=plot_run(store,tmp_path/'plots')
    assert len(files)>=5
    from pathlib import Path
    assert all(Path(f).read_bytes().startswith(b'\x89PNG') for f in files)

def test_direct_forward_mask_positions(toy_config):
    from ablationlab.backend import Backend
    b=Backend(toy_config)
    class PositionAware(torch.nn.Module):
        def forward(self,input_ids,attention_mask,position_ids=None):pass
    b.model=PositionAware()
    batch={'input_ids':torch.tensor([[0,0,3,4],[3,4,5,6]]),
           'attention_mask':torch.tensor([[0,0,1,1],[1,1,1,1]])}
    result=b.forward_inputs(batch)
    assert result['position_ids'].tolist()==[[0,0,0,1],[0,1,2,3]]
    assert 'position_ids' not in batch

@pytest.mark.integration
def test_legacy_tensor_import_marks_uncalibrated_and_last_layer(store,tmp_path):
    from ablationlab.importer import import_directions
    from ablationlab.planner import choose_next
    tensor=torch.randn(4,24)
    file=tmp_path/'directions.pt'
    torch.save({'model_id':'toy:dense','directions':tensor,'layer_results':[(i,float(i+1)) for i in range(4)]},file)
    with store.lock():
        result=import_directions(store,file)
        Pipeline(store).run_stage('baseline')
    assert result['calibrated'] is False
    assert store.result('directions')['eligible_layers']==[0,1,2]
    choice=choose_next({s:store.result(s) for s in ['inspect','baseline','capture','directions']})
    assert choice.status=='needs_review'
    bundle=DirectionBundle.load(store.stage_dir('directions'))
    assert bundle.metadata['excluded_layers']==[3]

@pytest.mark.parametrize('edits',[[{'layer':1}], [{'layer':1,'site':'mlp','strength':float('nan')}],
                                 [{'layer':-1,'site':'mlp','strength':.2}]])
def test_bad_export_plan_fails_before_loading(store,tmp_path,edits):
    from ablationlab.export import export_checkpoint
    p=tmp_path/'plan.json';write_json(p,{'schema_version':1,'edits':edits})
    with pytest.raises(LabError):export_checkpoint(store,p,tmp_path/'out')

def test_export_rejects_parameter_tied_to_unselected_module(store,tmp_path,monkeypatch):
    import ablationlab.backend as backend_module
    from ablationlab.export import make_export_plan, export_checkpoint
    from ablationlab.util import Unsupported
    pipe=Pipeline(store)
    with store.lock():
        pipe.run_stage('directions',with_deps=True)
        backend=pipe.engine.backend
        writer=backend.adapter.get(1,'attention').module
        alias=torch.nn.Linear(writer.in_features,writer.out_features,bias=False)
        alias.weight=writer.weight
        backend.model.unselected_alias=alias
        monkeypatch.setattr(backend_module,'Backend',lambda config:backend)
        plan=make_export_plan(store);plan['edits']=[{'layer':1,'site':'attention','strength':.5}]
        file=tmp_path/'plan.json';write_json(file,plan)
        with pytest.raises(Unsupported,match='aliases'):export_checkpoint(store,file,tmp_path/'out')
    assert not (tmp_path/'out').exists()

def test_resume_rejects_changed_model_identity(store,monkeypatch):
    from ablationlab.engine import Engine
    import ablationlab.backend as module
    pipeline=Pipeline(store)
    with store.lock():pipeline.run_stage('inspect')
    original=module.Backend
    def changed(config):
        result=original(config)
        result.identity['resolved_revision']='unexpected-new-revision'
        return result
    monkeypatch.setattr(module,'Backend',changed)
    with pytest.raises(LabError,match='identity changed'):Engine(store).backend
