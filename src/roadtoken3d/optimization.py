"""Current v6 gate supervision and exact pooled reduction arithmetic."""
import random
import numpy as np
import torch
from torch.nn import functional as F
from .current_model import LAMBDA_MODE, GATE_DEADBAND_M
from .current_loss import joint_residual_loss, DEFAULT_WEIGHTS
from .residual import OBSERVATION_KEYS
from types import SimpleNamespace
GATE_AUX_WEIGHT = 0.5
TIME_GATE_AUX_WEIGHT = 0.5
base = SimpleNamespace(GATE_AUX_WEIGHT=GATE_AUX_WEIGHT)
rt = SimpleNamespace(OBSERVATION_KEYS=OBSERVATION_KEYS, joint_residual_loss=joint_residual_loss)
TERMS = tuple(DEFAULT_WEIGHTS) + ('height_gate_aux', 'time_gate_aux')
def balanced_gate_loss(gate_values, target_values):
    """Class-balanced smooth L1 so the gate is not pushed open by a 73/27 prior.

    Measured on the v5 family: the hard label-oracle target is positive for
    about 73% of the usable (candidate, horizon) entries, and v5's gate then
    saturates near 1.0 within 50 steps (DEV gate mean 0.90 at step 50).  The
    lambda sweep on the same weights shows that lambda = 1 is the regime where
    the high-heterogeneity gain disappears (interaction +9.5), while
    lambda = 0.5-0.75 is where it is largest and interval-separated
    (het_hi -15.5 [-29.6, -5.2] and -19.4 [-34.5, -5.8]).  Balancing the two
    classes keeps the gate's operating point near the informative middle
    instead of letting the class prior dominate; the labels and the deadband
    are unchanged, and only the loss weights change.
    """
    positive = (target_values > 0.5).to(gate_values.dtype)
    fraction = positive.mean().clamp(1e-3, 1.0 - 1e-3)
    weight = torch.where(positive > 0.5,
                         0.5 / fraction, 0.5 / (1.0 - fraction))
    return (F.smooth_l1_loss(gate_values, target_values, reduction='none')
            * weight).mean()

def time_gate_aux(model, graph, batch, mode_valid, cache=None, key=None):
    """Hard label-oracle target for the occupancy gate (training only).

    ``graph`` is the road-graph encoder output captured by a forward hook.
    The route position at each horizon is the hard segment position implied by
    the predicted traversal times; the target is 1 where that position is
    closer to the recorded future than the uniform-occupancy position and 0
    otherwise, inside a 1 m deadband.  All targets are detached and cached per
    micro-batch so the pooled replay pass reuses them bit-exactly.
    """
    if graph is None:
        zero = batch['target_xyz'].sum() * 0.
        return zero, 0.
    output = graph
    gate = output.get('horizon_time_gate')
    entry = output.get('segment_entry_time_s')
    exit_ = output.get('segment_exit_time_s')
    seg_valid = output.get('segment_valid')
    node_xy = output.get('route_entry_xy')
    if gate is None or entry is None or node_xy is None:
        zero = batch['target_xyz'].sum() * 0.
        return zero, 0.
    if LAMBDA_MODE[model.time_gate_arm] != 'learned':
        return gate.sum() * 0., 0.
    if cache is not None and key in cache:
        usable, target = cache[key]
        return (balanced_gate_loss(gate[usable], target[usable])
                if usable.any() else gate.sum() * 0.), TIME_GATE_AUX_WEIGHT
    steps = gate.shape[-1]
    interval = float(model.road.road_graph.interval_seconds)
    with torch.no_grad():
        horizon = torch.arange(1, steps + 1, device=gate.device,
                               dtype=gate.dtype) * interval
        sv = seg_valid.to(gate.dtype)
        entry_xy = node_xy.detach().to(gate.dtype)
        exit_xy = graph['route_exit_xy'].detach().to(gate.dtype)
        count = sv.sum(dim=-1).clamp_min(1.0)
        inside = ((entry[:, :, None, :] <= horizon[None, None, :, None])
                  & (horizon[None, None, :, None] <= exit_[:, :, None, :])
                  & (sv[:, :, None, :] > 0))
        any_inside = inside.any(dim=-1)
        index = inside.float().argmax(dim=-1)
        last = (sv.long().sum(dim=-1) - 1).clamp_min(0)
        index = torch.where(any_inside, index, last[:, :, None])
        duration = (exit_ - entry).clamp_min(1e-6)
        fraction = ((horizon[None, None, :]
                     - entry.gather(-1, index)) /
                    duration.gather(-1, index)).clamp(0.0, 1.0)
        fraction = torch.where(any_inside, fraction, torch.ones_like(fraction))
        gather_index = index[..., None].expand(-1, -1, -1, 2)
        start = entry_xy.gather(2, gather_index)
        stop = exit_xy.gather(2, gather_index)
        p_time = start + fraction[..., None] * (stop - start)
        p_uniform = (entry_xy * sv[..., None]).sum(dim=2) / count[..., None]
        truth = batch['target_xyz'][:, :steps, :2].to(gate.dtype)
        declared = batch['target_xy_valid'][:, :steps].bool()
        d_time = torch.linalg.vector_norm(p_time - truth[:, None], dim=-1)
        d_uniform = torch.linalg.vector_norm(p_uniform[:, :, None] - truth[:, None],
                                             dim=-1)
        # Hard label-oracle choice with a deadband around the tie.
        target = (d_uniform > d_time).to(gate.dtype)
        usable = (mode_valid.bool()[..., None] & declared[:, None, :]
                  & (count[:, :, None] > 1.0)
                  & ((d_uniform - d_time).abs() > GATE_DEADBAND_M))
    if cache is not None and key is not None:
        cache[key] = (usable, target)
    if usable.any():
        aux = balanced_gate_loss(gate[usable], target[usable])
    else:
        aux = gate.sum() * 0.
    return aux, TIME_GATE_AUX_WEIGHT

def loss_parts(m, batch, target_cache=None):
    target_cache = {} if target_cache is None else target_cache
    captured = {}

    def hook(module, inputs, output):
        if isinstance(output, dict):
            captured['graph'] = output
    handle = m.road.road_graph.register_forward_hook(hook)
    try:
        output = m(*(batch[k] for k in rt.OBSERVATION_KEYS))
    finally:
        handle.remove()
    assert output['trajectories'].device == batch['history_xyz'].device
    part = rt.joint_residual_loss(output, batch, return_parts=True, split='train')[1]
    gate = output.get('height_anchor_gate')
    dz = output.get('height_anchor_delta_z')
    if gate is not None and dz is not None:
        gate = gate.float()
        dz = dz.detach().float()
        steps = output['trajectories'].shape[-2]
        truth = batch['target_xyz'][:, :steps].float()
        declared = batch['target_xy_valid'][:, :steps].bool() & batch['target_z_valid'][:, :steps].bool()
        valid = output['mode_valid'].bool()[..., None] & declared[:, None, :]
        usable = valid & (dz.abs() > 0.5)
        with torch.no_grad():
            base_z = output['trajectories'].detach()[..., 2] - gate.detach() * dz
            best = ((truth[:, None, :, 2] - base_z) / torch.where(usable, dz, torch.ones_like(dz))).clamp(0.0, 1.0)
            best = torch.where(usable, best, torch.zeros_like(best))
        aux = F.smooth_l1_loss(gate[usable], best[usable]) if usable.any() else gate.sum() * 0.0
        weight = base.GATE_AUX_WEIGHT
    else:
        aux = output['trajectories'].sum() * 0.0
        weight = 0.0
    part['reductions']['height_gate_aux'] = dict(numerator=aux, denominator=aux.new_ones(()), unit='label_oracle_trust_gate_smooth_l1')
    part['weights']['height_gate_aux'] = weight
    (t_aux, t_weight) = time_gate_aux(m, captured.get('graph'), batch, output['mode_valid'], target_cache, id(batch))
    part['reductions']['time_gate_aux'] = dict(numerator=t_aux, denominator=t_aux.new_ones(()), unit='hard_label_oracle_occupancy_gate_smooth_l1')
    part['weights']['time_gate_aux'] = t_weight
    return part

def capture_rng(device):
    state = dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state().clone(), device=None)
    if device.type == 'npu':
        state['device'] = [x.cpu().clone() for x in torch.npu.get_rng_state_all()]
    return state

def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'npu':
        torch.npu.set_rng_state_all(state['device'])

def rng_equal(a, b):
    return (a['python'] == b['python'] and a['numpy'][0] == b['numpy'][0]
        and np.array_equal(a['numpy'][1], b['numpy'][1]) and a['numpy'][2:] == b['numpy'][2:]
        and torch.equal(a['torch'], b['torch']) and
        ((a['device'] is None and b['device'] is None) or
         (a['device'] is not None and b['device'] is not None and len(a['device']) == len(b['device'])
          and all(torch.equal(x,y) for x,y in zip(a['device'],b['device'])))))


def pooled_arithmetic_step(model,optimizer,batches,parts_fn,device,*,expected_global_rows=128):
    """Pure callback component; production wrapper below enforces model/family."""
    device=torch.device(device)
    if not batches or sum(len(b["sample_id"]) for b in batches)!=expected_global_rows or expected_global_rows!=128:
        raise ValueError("Exactly 128 global TRAIN draws required before any optimizer update")
    parameters=[p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.device!=device for p in parameters):raise ValueError("Trainable parameter device mismatch")
    model.train();optimizer.zero_grad(set_to_none=True)
    counted=[];rngs=[];den=torch.zeros(len(TERMS),device=device)
    weight=None
    for batch in batches:
        begin=capture_rng(device)
        with torch.no_grad():parts=parts_fn(model,batch)
        num=torch.stack([parts["reductions"][k]["numerator"] for k in TERMS])
        localden=torch.stack([parts["reductions"][k]["denominator"] for k in TERMS])
        current=num.new_tensor([parts["weights"][k] for k in TERMS])
        if weight is None:weight=current
        elif not torch.equal(weight,current):raise ValueError("Microbatch objective weights differ")
        if num.device!=device or localden.device!=device or not torch.isfinite(num).all() or not torch.isfinite(localden).all() or (localden<0).any():raise FloatingPointError("Invalid count reductions")
        counted.append((num.detach(),localden.detach()));rngs.append((begin,capture_rng(device)));den+=localden
    summed=torch.zeros_like(den)
    for i,batch in enumerate(batches):
        restore_rng(rngs[i][0],device);parts=parts_fn(model,batch)
        num=torch.stack([parts["reductions"][k]["numerator"] for k in TERMS])
        localden=torch.stack([parts["reductions"][k]["denominator"] for k in TERMS])
        current=num.new_tensor([parts["weights"][k] for k in TERMS])
        if (not torch.equal(current,weight) or not torch.equal(localden.detach(),counted[i][1]) or
            not torch.allclose(num.detach(),counted[i][0],rtol=2e-5,atol=1e-6) or
            not rng_equal(capture_rng(device),rngs[i][1])):raise RuntimeError("Count/replay value, denominator, weight or device RNG differs")
        loss=(num*weight/torch.where(den>0,den,torch.ones_like(den))).sum()
        loss=loss+sum(p.sum()*0 for p in parameters)
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite pooled micro loss")
        loss.backward();summed+=num.detach()
    restore_rng(rngs[-1][1],device)
    if any(p.grad is None or p.grad.device!=device or not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError("Invalid gradient before optimizer update")
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step()
    for p in parameters:
        for v in (p,*optimizer.state[p].values()):
            if isinstance(v,torch.Tensor) and not torch.isfinite(v).all():raise FloatingPointError("Nonfinite parameter/Adam state after update")
    mean=summed/torch.where(den>0,den,torch.ones_like(den))
    return dict(loss=float((mean*weight).sum()),numerators={k:float(summed[i]) for i,k in enumerate(TERMS)},
        denominators={k:float(den[i]) for i,k in enumerate(TERMS)},weights={k:float(weight[i]) for i,k in enumerate(TERMS)},
        actual_global_rows=128,actual_microbatch_rows=[len(b["sample_id"]) for b in batches],
        gradient_norm=float(norm),count_replay_values_and_RNG_verified=True,device=str(device),world_size=1)
