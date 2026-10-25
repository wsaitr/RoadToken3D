"""Small generated fixtures only: zero real trajectories/checkpoints."""
from types import SimpleNamespace
import copy,io,json,hashlib,random,tempfile
from pathlib import Path
import numpy as np
import torch
from roadtoken3d.current_model import from_parents
from roadtoken3d.dense_model import DenseCoordinateForecaster
from roadtoken3d.route_model_v13 import RoadNativeV13
from roadtoken3d.residual import OBSERVATION_KEYS
from roadtoken3d.optimization import loss_parts,TERMS,pooled_arithmetic_step,capture_rng,restore_rng,rng_equal
from roadtoken3d.data import WindowDataset,DrawStream,LABEL_KEYS
from roadtoken3d.ascend import atomic_save,frozen_guard
from torch.utils.data import default_collate
from roadtoken3d.prefix import scan
def model_batch(length=2000.0):
    graph = SimpleNamespace(points_xyz=np.array([[72000.0, 3279000.0, 0.0], [72000.0 + length, 3279000.0, 0.0]]), s_road=np.array([0.0, length]), road_ptr=np.array([0, 2]), road_length_m=np.array([length]))
    history = torch.zeros(2, 5, 3)
    history[:, :, 0] = torch.arange(-4, 1) * 15.0
    target = torch.zeros(2, 4, 3)
    target[:, :, 0] = torch.arange(1, 5) * 30.0
    batch = {'history_xyz': history, 'history_xy_valid': torch.ones(2, 5, dtype=torch.bool), 'history_z_valid': torch.ones(2, 5, dtype=torch.bool), 'origin_context': torch.zeros(2, 3), 'time_features': torch.zeros(2, 4), 'history_matched_xyz': history.clone(), 'history_road_tangent': torch.tensor([1.0, 0.0]).expand(2, 5, 2).clone(), 'history_road_d': torch.zeros(2, 5), 'history_map_gap_m': torch.zeros(2, 5), 'history_map_confidence': torch.ones(2, 5), 'route_roads': torch.zeros(2, 2, 1, dtype=torch.long), 'route_directions': torch.ones(2, 2, 1, dtype=torch.long), 'route_count': torch.ones(2, 2, dtype=torch.long), 'route_valid': torch.ones(2, 2, dtype=torch.bool), 'route_start_s': torch.zeros(2, 2), 'origin_local': torch.zeros(2, 3), 'target_xyz': target, 'target_xy_valid': torch.ones(2, 4, dtype=torch.bool), 'target_matched_xyz': target.clone(), 'target_map_confidence': torch.ones(2, 4), 'teacher_progress': target[..., 0][:, None].expand(-1, 2, -1).clone(), 'teacher_progress_weight': torch.ones(2, 2, 4), 'teacher_route_coverage': torch.ones(2, 2)}
    return (None, batch)

def graph_fixture(model_class=RoadNativeV13):
    points = np.array([[72000.0, 3279000.0, 0.0], [72100.0, 3279000.0, 5.0], [72100.0, 3279000.0, 5.0], [72300.0, 3279000.0, 15.0], [72300.0, 3279000.0, 15.0], [72700.0, 3279000.0, 15.0]], dtype=np.float32)
    graph = SimpleNamespace(points_xyz=points, s_road=np.array([0.0, 100.0, 0.0, 200.0, 0.0, 400.0], dtype=np.float32), road_ptr=np.array([0, 2, 4, 6]), road_length_m=np.array([100.0, 200.0, 400.0], dtype=np.float32))
    (_, batch) = model_batch()
    batch['route_roads'] = torch.tensor([0, 1, 2]).expand(2, 2, 3).clone()
    batch['route_directions'] = torch.ones(2, 2, 3, dtype=torch.long)
    batch['route_count'].fill_(3)
    batch['history_xyz'][..., 2] = batch['history_xyz'][..., 0] * 0.05
    batch['history_matched_xyz'] = batch['history_xyz'].clone()
    batch['target_xyz'][..., 2] = batch['target_xyz'][..., 0] * 0.05
    batch['target_z_valid'] = batch['target_xy_valid'].clone()
    model = model_class(graph, history_steps=5, future_steps=4, width=16, layers=1, dropout=0.0)
    return (model, batch)

def _model_batch():
    (model, batch) = graph_fixture(RoadNativeV13)
    (nodes, degree) = (8, 2)
    state = torch.tensor([1, 3, 5, 0, 0, 0, 0, 0])
    successors = torch.zeros(nodes, degree, dtype=torch.long)
    predecessors = torch.zeros(nodes, degree, dtype=torch.long)
    (successors[0, 0], successors[1, 0]) = (1, 2)
    (predecessors[1, 0], predecessors[2, 0]) = (0, 1)
    valid_out = torch.zeros(nodes, degree, dtype=torch.bool)
    valid_in = torch.zeros(nodes, degree, dtype=torch.bool)
    valid_out[0, 0] = valid_out[1, 0] = True
    valid_in[1, 0] = valid_in[2, 0] = True
    local = {'local_state': state, 'local_node_valid': torch.tensor([True, True, True, False, False, False, False, False]), 'local_successors': successors, 'local_strict_successor_valid': valid_out, 'local_predecessors': predecessors, 'local_strict_predecessor_valid': valid_in, 'local_start_valid': torch.tensor([True, False, False, False, False, False, False, False]), 'local_start_distance_m': torch.tensor([[0.0, 100.0, 300.0, 10000000.0, 10000000.0, 10000000.0, 10000000.0, 10000000.0], [0.0, 100.0, 300.0, 10000000.0, 10000000.0, 10000000.0, 10000000.0, 10000000.0]]), 'local_start_node_index': torch.tensor([0, 0]), 'local_start_parent': torch.tensor([[-1, 0, 1, -1, -1, -1, -1, -1], [-1, 0, 1, -1, -1, -1, -1, -1]])}
    batch.update({key: value[None].expand(2, *value.shape).clone() for (key, value) in local.items()})
    batch['target_road_id'] = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])
    batch['target_road_s'] = torch.tensor([[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]])
    return (model, batch)

def assert_same(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same(x,y)
    else:assert a==b

def runtime_checks(model,batch):
    """Three tiny synthetic updates: uninterrupted vs save/reload/replay."""
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);rows=[]
        for i in range(3):
            path=root/f'window-{i}.npz'
            np.savez(path,**{k:batch[k][i%2].detach().numpy() for k in OBSERVATION_KEYS+LABEL_KEYS})
            rows.append(dict(sample_id=f'generated-{i}',file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest=root/'train.json'
        manifest.write_text(json.dumps(dict(split='train',windows=rows)))
        dataset=WindowDataset(manifest,labels=True)
        observations=WindowDataset(manifest,labels=False)[0]
        assert not any(k.startswith('target_') for k in observations)
        for k in OBSERVATION_KEYS+LABEL_KEYS:assert torch.equal(dataset[0][k],batch[k][0])
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001)
        stream=DrawStream(len(dataset),17);device=torch.device('cpu')
        def update(m,o,s):
            indices=s.take(128)
            batches=[default_collate([dataset[j] for j in indices[i:i+16]]) for i in range(0,128,16)]
            cache={}
            record=pooled_arithmetic_step(m,o,batches,lambda mm,b:loss_parts(mm,b,cache),device)
            assert record['count_replay_values_and_RNG_verified']
            frozen_guard(m)
            return record,indices
        before={k:v.clone() for k,v in model.state_dict().items()}
        first,_=update(model,opt,stream)
        assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        checkpoint=root/'step-0001.pt'
        atomic_save(checkpoint,dict(step=1,model=model.state_dict(),optimizer=opt.state_dict(),
            rng=capture_rng(device),draw_stream=stream.state_dict()))
        expected,draws=update(model,opt,stream)
        expected_model=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
        expected_rng=capture_rng(device);expected_draw=copy.deepcopy(stream.state_dict())
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
        replay=copy.deepcopy(model);replay.load_state_dict(loaded['model'],strict=True)
        replay_opt=torch.optim.AdamW([p for p in replay.parameters() if p.requires_grad],lr=.0003,weight_decay=.001)
        replay_opt.load_state_dict(loaded['optimizer'])
        replay_stream=DrawStream(len(dataset),999);replay_stream.load_state_dict(loaded['draw_stream'])
        random.random();np.random.random();torch.rand(1)
        restore_rng(loaded['rng'],device)
        actual,replay_draws=update(replay,replay_opt,replay_stream)
        assert draws==replay_draws and expected==actual
        assert_same(expected_model,replay.state_dict());assert_same(expected_opt,replay_opt.state_dict())
        assert_same(expected_draw,replay_stream.state_dict())
        assert rng_equal(expected_rng,capture_rng(device))
        try:atomic_save(checkpoint,{})
        except FileExistsError:pass
        else:raise AssertionError('Checkpoint overwrite was not rejected')
        path=root/rows[0]['file'];path.write_bytes(path.read_bytes()+b'changed')
        try:WindowDataset(manifest,labels=True)
        except ValueError:pass
        else:raise AssertionError('Input hash mismatch was not rejected')
    return dict(prepared_input_roundtrip=True,observation_only_input=True,input_hash_guard=True,
        pooled_128_rows_and_eight_losses=True,parameters_actually_updated=True,
        optimizer_model_draw_and_RNG_resume_exact=True,immutable_checkpoint_guard=True,
        synthetic_optimizer_updates=3,real_data_rows=0)

def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    road,batch=_model_batch()
    direct=DenseCoordinateForecaster(history_steps=5,future_steps=4,width=16,layers=1,modes=1,dropout=0.,architecture='attention')
    batch['sample_id']=['synthetic-0','synthetic-1']
    model=from_parents(direct,road,hops=4,message_width=8,checkpoint_chunk=0)
    model.eval()
    with torch.no_grad():
        out=model(*(batch[k] for k in OBSERVATION_KEYS))
        assert torch.equal(out['trajectories'],out['cartesian_backbone_xyz'][:,None].expand_as(out['trajectories']))
        # Open the actual two gates and motion heads; a zero-initialization-only
        # test would miss the historical height-broadcast failure.
        model.height_gate[-1].bias.fill_(1.)
        model.time_gate[-1].bias.fill_(.2)
        model.correction[-1].bias[:4].copy_(torch.tensor([.1,.05,.05,.02]))
        first=model(*(batch[k] for k in OBSERVATION_KEYS))
        changed={k:(v+1000 if k.startswith('target_')else v)for k,v in batch.items()}
        second=model(*(changed[k] for k in OBSERVATION_KEYS))
        assert torch.equal(first['trajectories'],second['trajectories'])
        assert torch.equal(first['mode_logits'],second['mode_logits'])
        assert first['height_anchor_gate'].max()>0
        assert torch.equal(first['height_anchor_displacement_xyz'][...,:2],torch.zeros_like(first['height_anchor_displacement_xyz'][...,:2]))
        assert torch.allclose(first['trajectories'],first['learned_XYZ_branch_xyz']+first['road_residual_displacement_xyz']+first['height_anchor_displacement_xyz'])
        assert first['road_residual_displacement_xyz'][...,:2].norm(dim=-1).max()<=20.0001
        assert first['road_residual_displacement_xyz'][...,2].abs().max()<=2.000001
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        model.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
        reloaded=model(*(batch[k] for k in OBSERVATION_KEYS))
        assert torch.equal(first['trajectories'],reloaded['trajectories'])
    model.train();model.zero_grad(set_to_none=True)
    parts=loss_parts(model,batch)
    assert set(parts['reductions'])==set(TERMS) and len(TERMS)==8
    loss=sum(parts['weights'][k]*parts['reductions'][k]['numerator']/parts['reductions'][k]['denominator'].clamp_min(1) for k in TERMS)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.correction[-1].weight.grad is not None
    assert model.height_gate[-1].weight.grad is not None
    assert model.time_gate[-1].weight.grad is not None
    assert all(p.grad is None for p in model.cartesian.parameters())
    x=torch.tensor([[1.5,2.5,0.,0.]],requires_grad=True)
    y=scan(x,-1)
    assert torch.equal(y,torch.tensor([[1.5,4.,4.,4.]]))
    y.sum().backward();assert torch.equal(x.grad,torch.tensor([[4.,3.,2.,1.]]))
    runtime=runtime_checks(model,batch)
    print(json.dumps(dict(passed=True,zero_correction_identity=True,nonzero_height_anchor_is_Z_only=True,
        future_label_poisoning_invariance=True,current_eight_loss_terms=True,three_heads_receive_gradients=True,
        same_state_reload_exact=True,prefix_zero_tail_and_gradient=True,real_data_rows=0,optimizer_updates=3,
        NPU_device_execution=False,runtime_checks=runtime),indent=2))
if __name__=='__main__':main()
