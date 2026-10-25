"""Current paper v6 model; portable tensor core, no training scheduler."""
from .prefix import cumsum as _prefix_sum
import sys
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .endpoint import EndpointStableRoadDisplacementResidualFusion as Base
from .route_model_v11 import TravelTimeRoadGraphEncoder
jm = sys.modules[__name__]
ARMS = ('joint-full', 'joint-XYZ-only', 'joint-fixed-hop', 'joint-no-Z-coupling')
FAMILIES = {a: a + '-20261018-v1' for a in ARMS}
EQUATIONS = {a: '(C + learned_XYZ_delta) + smooth_bound_road_delta' for a in ARMS}
XY_LIMIT = 20.0
Z_LIMIT = 2.0

def smooth_bound(d):
    xy = d[..., :2] * torch.rsqrt(1.0 + d[..., :2].square().sum(-1, keepdim=True) / 400.0)
    v = torch.cat((xy, 2.0 * torch.tanh(d[..., 2:3] / 2.0)), -1)
    previous = torch.cat((torch.zeros_like(v[..., :1, :]), v[..., :-1, :]), -2)
    following = torch.cat((v[..., 1:, :], v[..., -1:, :]), -2)
    return 0.25 * previous + 0.5 * v + 0.25 * following

def transform_output(original, arm):
    o = dict(original)
    raw = original['correction_branch_raw_delta']
    assert raw.shape[-1] == 7
    xyz = raw[..., 4:7] * raw.new_tensor([1000.0, 1000.0, 20.0])
    road = original['correction_branch_displacement_xyz']
    if arm == 'joint-no-Z-coupling':
        dz = original['correction_branch_height_residual_m'] - original['correction_branch_reference_height_residual_m']
        road = torch.cat((road[..., :2], dz[..., None]), -1)
    applied = torch.zeros_like(road) if arm == 'joint-XYZ-only' else smooth_bound(road)
    o['learned_XYZ_displacement_xyz'] = xyz
    o['learned_XYZ_branch_xyz'] = original['cartesian_backbone_xyz'][:, None] + xyz
    o['road_residual_displacement_xyz'] = applied
    o['diagnostic_unbounded_road_delta_xyz'] = road
    o['diagnostic_original_coupled_road_delta_xyz'] = original['correction_branch_displacement_xyz']
    o['trajectories'] = o['learned_XYZ_branch_xyz'] + applied
    o.pop('correction_branch_displacement_xyz')
    for which in ('reference', 'corrected'):
        o['diagnostic_road_' + which + '_xyz'] = o.pop('correction_branch_' + which + '_xyz')
    o['metadata'] = {**original['metadata'], 'model_family': FAMILIES[arm], 'arm': arm, 'final_equation': EQUATIONS[arm], 'learned_XYZ_branch_present': True, 'road_residual_present': arm != 'joint-XYZ-only', 'physical_delayed_messages_present': arm != 'joint-fixed-hop', 'progress_height_coupling_present': arm not in ('joint-no-Z-coupling', 'joint-XYZ-only'), 'same_seven_channel_parameterization_across_arms': True, 'road_horizontal_vector_norm_limit_m': 20.0, 'road_vertical_absolute_limit_m': 2.0, 'XYZ_branch_is_not_subject_to_road_residual_cap': True}
    return o

def validate_output(o):
    arm = o['metadata']['arm']
    assert arm in ARMS and o['metadata']['model_family'] == FAMILIES[arm]
    c = o['cartesian_backbone_xyz']
    xyz = o['learned_XYZ_displacement_xyz']
    b = o['learned_XYZ_branch_xyz']
    road = o['road_residual_displacement_xyz']
    assert xyz.shape == b.shape == road.shape == o['trajectories'].shape
    assert torch.equal(b, c[:, None] + xyz) and torch.equal(o['trajectories'], b + road)
    assert torch.linalg.vector_norm(road[..., :2], dim=-1).max() <= 20.0 + 0.0001
    assert road[..., 2].abs().max() <= 2.0 + 1e-06
    if arm == 'joint-XYZ-only':
        assert torch.equal(road, torch.zeros_like(road))
    else:
        assert torch.equal(road, smooth_bound(o['diagnostic_unbounded_road_delta_xyz']))
    valid = o['mode_valid']
    assert torch.equal(o['forecast_valid'], valid.any(-1))
    assert torch.isfinite(o['trajectories'][valid]).all()

class JointXYZRoad(Base):

    def __init__(self, initialized_anchor, arm):
        torch.nn.Module.__init__(self)
        assert type(initialized_anchor) is Base and arm in ARMS
        self.arm = arm
        for name in ('cartesian', 'road', 'correction'):
            self.add_module(name, getattr(initialized_anchor, name))
        for name in ('future_steps', 'dt', 'frozen_road_parent_parameter_names'):
            setattr(self, name, getattr(initialized_anchor, name))
        old = self.correction[-1]
        assert old.out_features == 4
        self.correction[-1] = torch.nn.Linear(old.in_features, 7, device=old.weight.device, dtype=old.weight.dtype)
        torch.nn.init.zeros_(self.correction[-1].weight)
        torch.nn.init.zeros_(self.correction[-1].bias)
        if arm == 'joint-fixed-hop':
            self.road.proposer.delayed_messages.mode = 'fixed-hop'
            self.road.proposer.suppress_arrival_feature = True
            self.road.proposer.time_aware = False
            self.road.road_graph.time_aware = False
        self.train(initialized_anchor.training)

    def forward(self, *observations):
        o = transform_output(super().forward(*observations), self.arm)
        validate_output(o)
        return o
ARMS_V3 = ('joint-timegate-anchor', 'joint-timegate-off-anchor', 'joint-timegate-fixedhop-anchor', 'joint-timegate-shuffled-anchor', 'joint-timegate-decoupled-anchor', 'joint-timegate-independent-z-anchor')
LAMBDA_MODE = {'joint-timegate-anchor': 'learned', 'joint-timegate-off-anchor': 'off', 'joint-timegate-fixedhop-anchor': 'off', 'joint-timegate-shuffled-anchor': 'learned', 'joint-timegate-decoupled-anchor': 'learned', 'joint-timegate-independent-z-anchor': 'learned'}
CONFIG_V3 = {'joint-timegate-anchor': dict(time_gate=False, anchor=True, independent_z=False, fixed_hop=False, shuffle=False), 'joint-timegate-off-anchor': dict(time_gate=False, anchor=True, independent_z=False, fixed_hop=False, shuffle=False), 'joint-timegate-fixedhop-anchor': dict(time_gate=False, anchor=True, independent_z=False, fixed_hop=True, shuffle=False), 'joint-timegate-shuffled-anchor': dict(time_gate=False, anchor=True, independent_z=False, fixed_hop=False, shuffle=True), 'joint-timegate-decoupled-anchor': dict(time_gate=False, anchor=False, independent_z=False, fixed_hop=False, shuffle=False), 'joint-timegate-independent-z-anchor': dict(time_gate=False, anchor=False, independent_z=True, fixed_hop=False, shuffle=False)}
ARMS_V6 = ('joint-timegate6-anchor', 'joint-timegate6-off-anchor', 'joint-timegate6-fixedhop-anchor', 'joint-timegate6-shuffled-anchor', 'joint-timegate6-decoupled-anchor', 'joint-timegate6-independent-z-anchor')
_SOURCE = {'joint-timegate6-anchor': 'joint-timegate-anchor', 'joint-timegate6-off-anchor': 'joint-timegate-off-anchor', 'joint-timegate6-fixedhop-anchor': 'joint-timegate-fixedhop-anchor', 'joint-timegate6-shuffled-anchor': 'joint-timegate-shuffled-anchor', 'joint-timegate6-decoupled-anchor': 'joint-timegate-decoupled-anchor', 'joint-timegate6-independent-z-anchor': 'joint-timegate-independent-z-anchor'}
GATE_GAIN = 10.0
GATE_DEADBAND_M = 1.0
ARMS_V2 = ARMS_V6
CONFIG = {a: dict(CONFIG_V3[s]) for (a, s) in _SOURCE.items()}
LAMBDA_MODE.update({a: LAMBDA_MODE[s] for (a, s) in _SOURCE.items()})
ARMS = tuple(ARMS) + ARMS_V6
ANCHOR_EQUATION = '(C + learned_XYZ_delta + g*(road_Z - C_Z)) + smooth_bound_road_delta'
for arm in ARMS_V6:
    FAMILIES[arm] = arm + '-20261023-v6'
    EQUATIONS[arm] = ANCHOR_EQUATION
DELAY_SHUFFLE_SEED = 20261022

class AnchoredVariant(jm.JointXYZRoad):

    def __init__(self, initialized_anchor, arm):
        super().__init__(initialized_anchor, arm)
        cfg = CONFIG[arm]
        if cfg['fixed_hop']:
            self.road.proposer.delayed_messages.mode = 'fixed-hop'
            self.road.proposer.suppress_arrival_feature = True
            self.road.proposer.time_aware = False
            self.road.road_graph.time_aware = False
        if not cfg['time_gate']:
            self.road.proposer.time_aware = False
            self.road.road_graph.time_aware = False
        else:
            self.road.road_graph.time_aware = True
        if cfg['shuffle']:
            messenger = self.road.proposer.delayed_messages
            original = messenger.propagate

            def shuffled_propagate(features, travel_seconds, *args, **kwargs):
                nodes = travel_seconds.shape[1]
                generator = torch.Generator(device=travel_seconds.device)
                generator.manual_seed(DELAY_SHUFFLE_SEED + nodes)
                perm = torch.randperm(nodes, generator=generator, device=travel_seconds.device)
                return original(features, travel_seconds[:, perm], *args, **kwargs)
            messenger.propagate = shuffled_propagate
        self._cfg = cfg
        head = self.cartesian.xyz_head
        width = head.in_features
        reference = head.weight
        (device, dtype) = (reference.device, reference.dtype)
        if cfg['anchor'] or arm == 'joint-decoupled-anchor':
            self.height_gate = torch.nn.Sequential(torch.nn.Linear(width + 3, width, device=reference.device, dtype=reference.dtype), torch.nn.GELU(), torch.nn.Linear(width, 1, device=reference.device, dtype=reference.dtype))
            torch.nn.init.zeros_(self.height_gate[-1].weight)
            torch.nn.init.zeros_(self.height_gate[-1].bias)
        if cfg['independent_z']:
            self.z_head = torch.nn.Sequential(torch.nn.Linear(width, width, device=reference.device, dtype=reference.dtype), torch.nn.GELU(), torch.nn.Linear(width, 1, device=reference.device, dtype=reference.dtype))
            torch.nn.init.zeros_(self.z_head[-1].weight)
            torch.nn.init.zeros_(self.z_head[-1].bias)

    def forward(self, *observations):
        output = _PARENT_FORWARD(self, *observations)
        cfg = self._cfg
        if cfg['independent_z']:
            with torch.no_grad():
                latent = self.cartesian(*observations[:5])['latent_horizon']
            raw_z = self.z_head(latent).squeeze(-1).float()
            unbounded = output['diagnostic_unbounded_road_delta_xyz'].clone()
            unbounded[..., 2] = 2.0 * raw_z
            road = jm.smooth_bound(unbounded)
            output['diagnostic_unbounded_road_delta_xyz'] = unbounded
            output['road_residual_displacement_xyz'] = road
            output['height_anchor_displacement_xyz'] = torch.zeros_like(road)
            output['trajectories'] = output['learned_XYZ_branch_xyz'] + road
            output['metadata'] = {**output['metadata'], 'progress_height_coupling_present': False, 'independent_z_head': True, 'independent_z_head_reads_progress': False, 'independent_z_head_input': 'cartesian_latent_horizon_only'}
        elif cfg['anchor']:
            corrected = output['diagnostic_road_corrected_xyz'].float()
            base = output['learned_XYZ_branch_xyz'].float()
            road = output['road_residual_displacement_xyz'].float()
            with torch.no_grad():
                latent = self.cartesian(*observations[:5])['latent_horizon']
            if latent.dim() == 4:
                latent = latent[:, 0]
            horizon = torch.arange(corrected.shape[-2], device=corrected.device, dtype=corrected.dtype)
            horizon = (horizon / max(corrected.shape[-2] - 1, 1))[None, None, :, None]
            horizon = horizon.expand(corrected.shape[0], corrected.shape[1], -1, 1)
            dz = corrected[..., 2:3] - base[..., 2:3] - road[..., 2:3]
            dxy = torch.linalg.vector_norm(corrected[..., :2] - base[..., :2], dim=-1, keepdim=True) / 50.0
            features = torch.cat((latent[:, None].expand(-1, corrected.shape[1], -1, -1), dz / 20.0, dxy, horizon), dim=-1)
            raw_gate = self.height_gate(features).squeeze(-1)
            gate = (torch.sigmoid(raw_gate) - 0.5).clamp_min(0.0) * 2.0
            anchor = gate[..., None] * dz
            anchor_xyz = torch.zeros_like(output['trajectories'])
            anchor_xyz[..., 2:3] = anchor
            output['height_anchor_displacement_xyz'] = anchor_xyz
            output['height_anchor_gate'] = gate
            output['height_anchor_delta_z'] = dz[..., 0]
            output['trajectories'] = output['trajectories'] + anchor_xyz
            output['metadata'] = {**output['metadata'], 'progress_height_coupling_present': True, 'height_anchor_present': True, 'height_anchor_gate_learned': True, 'height_anchor_initially_zero': True, 'height_anchor_source': 'road_center_Z_at_corrected_progress_plus_bounded_residual', 'height_anchor_inputs': 'cartesian_latent+height_delta+XY_gap+horizon', 'height_residual_is_bounded_m': 2.0}
        else:
            output['height_anchor_displacement_xyz'] = torch.zeros_like(output['trajectories'])
            output['metadata'] = {**output['metadata'], 'height_anchor_present': False, 'height_anchor_gate_present_but_zero': True, 'progress_height_coupling_present': False}
        arm = output['metadata']['arm']
        assert arm in jm.ARMS and output['metadata']['model_family'] == jm.FAMILIES[arm]
        road = output['road_residual_displacement_xyz']
        base = output['learned_XYZ_branch_xyz']
        anchor = output['height_anchor_displacement_xyz']
        assert output['trajectories'].shape == base.shape == road.shape == anchor.shape
        assert torch.allclose(output['trajectories'], base + road + anchor)
        assert torch.linalg.vector_norm(road[..., :2], dim=-1).max() <= 20.0 + 0.0001
        assert road[..., 2].abs().max() <= 2.0 + 1e-06
        gate = output.get('height_anchor_gate')
        if gate is not None:
            assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0 + 1e-06
        valid = output['mode_valid']
        assert torch.equal(output['forecast_valid'], valid.any(-1))
        assert torch.isfinite(output['trajectories'][valid]).all()
        return output
_PARENT_FORWARD = JointXYZRoad.forward

def validate_output_v2(o):
    """Variant-aware version of joint_model.validate_output.

    The anchor is a separate additive term: trajectories = branch + road
    residual + anchor.  The road residual itself keeps the original +-2 m
    contract; the anchor has no cap because it is the road base height, not a
    correction.
    """
    arm = o['metadata']['arm']
    assert arm in ARMS_V2 and o['metadata']['model_family'] == jm.FAMILIES[arm]
    c = o['cartesian_backbone_xyz']
    xyz = o['learned_XYZ_displacement_xyz']
    b = o['learned_XYZ_branch_xyz']
    road = o['road_residual_displacement_xyz']
    anchor = o.get('height_anchor_displacement_xyz', torch.zeros_like(o['trajectories']))
    assert xyz.shape == b.shape == road.shape == anchor.shape == o['trajectories'].shape
    assert torch.equal(b, c[:, None] + xyz)
    assert torch.equal(o['trajectories'], b + road + anchor)
    assert torch.linalg.vector_norm(road[..., :2], dim=-1).max() <= 20.0 + 0.0001
    assert road[..., 2].abs().max() <= 2.0 + 1e-06
    assert torch.equal(road, jm.smooth_bound(o['diagnostic_unbounded_road_delta_xyz']))
    if arm == 'joint-decoupled-anchor':
        assert torch.equal(anchor, torch.zeros_like(anchor))
    valid = o['mode_valid']
    assert torch.equal(o['forecast_valid'], valid.any(-1))
    assert torch.isfinite(o['trajectories'][valid]).all()
validate_output = validate_output_v2
base = SimpleNamespace(AnchoredVariant=AnchoredVariant)
v3 = SimpleNamespace(nn=nn)

def _occupancy_gate_forward(self, entry_xyz, exit_xyz, segment_length_m, valid, history_state, observed_speed_mps):
    """Frozen encoder forward with a learned time-uniform occupancy gate.

    Everything up to ``occupancy`` is byte-for-byte the frozen V11 forward.
    The gate is per candidate and horizon; at ``lambda = 0`` the result is the
    uniform occupancy of the matched no-delay arm.
    """
    displacement = exit_xyz - entry_xyz
    chord = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(0.1)
    unit = displacement / chord[..., None]
    grade = displacement[..., 2] / torch.linalg.vector_norm(displacement[..., :2], dim=-1).clamp_min(0.1)
    features = torch.cat((entry_xyz[..., :2] / 1000.0, entry_xyz[..., 2:3] / 20.0, unit, grade[..., None].clamp(-2.0, 2.0), torch.log1p(segment_length_m)[..., None] / 10.0, chord[..., None] / 1000.0, valid[..., None]), dim=-1)
    hidden = self.node_projection(features) * valid[..., None]
    for layer in self.layers:
        hidden = layer(hidden, valid)
    state = history_state[:, None, None, :].expand_as(hidden)
    speed_adjustment = 10.0 * self.traversal_speed(torch.cat((hidden, state), dim=-1)).squeeze(-1)
    traversal_speed = F.softplus(observed_speed_mps[:, None, None] + speed_adjustment).clamp(0.5, 60.0)
    duration = segment_length_m / traversal_speed
    duration = duration * valid
    exit_time = _prefix_sum(duration, dim=-1)
    entry_time = exit_time - duration
    horizon = torch.arange(1, self.horizon_embedding.shape[0] + 1, device=hidden.device, dtype=hidden.dtype) * self.interval_seconds
    temperature = self.interval_seconds
    time_occupancy = torch.sigmoid((horizon[None, None, :, None] - entry_time[:, :, None, :]) / temperature) - torch.sigmoid((horizon[None, None, :, None] - exit_time[:, :, None, :]) / temperature)
    time_occupancy = time_occupancy.clamp_min(0.0) * valid[:, :, None, :]
    uniform_occupancy = valid[:, :, None, :].expand(-1, -1, len(horizon), -1)
    mode = getattr(self, 'time_gate_mode', 'off')
    gate_module = getattr(self, 'time_gate_ref', None)
    if mode == 'learned' and gate_module is not None:
        count = valid.sum(dim=-1).clamp_min(1.0)
        candidate = (hidden * valid[..., None]).sum(dim=-2) / count[..., None]
        mean_speed = (traversal_speed * valid).sum(dim=-1) / count
        mean_length = (segment_length_m * valid).sum(dim=-1) / count
        spread = ((segment_length_m - mean_length[..., None]) ** 2 * valid).sum(dim=-1).clamp_min(0.0).sqrt()
        spread = spread / mean_length.clamp_min(1.0)
        stats = torch.stack((torch.log1p((duration * valid).sum(dim=-1)) / 5.0, mean_speed / 60.0, mean_length / 1000.0, spread), dim=-1)
        (batch, modes) = hidden.shape[:2]
        steps = len(horizon)
        gate_input = torch.cat((candidate[:, :, None, :].expand(-1, -1, steps, -1), stats[:, :, None, :].expand(-1, -1, steps, -1), self.horizon_embedding[None, None].expand(batch, modes, -1, -1), observed_speed_mps[:, None, None, None].expand(batch, modes, steps, 1).to(hidden.dtype)), dim=-1)
        raw = gate_module(gate_input).squeeze(-1)
        raw = raw * getattr(self, 'time_gate_gain', 1.0)
        gate = (torch.sigmoid(raw) - 0.5).clamp_min(0.0) * 2.0
    else:
        gate = torch.zeros_like(time_occupancy[..., 0])
    occupancy = gate[..., None] * time_occupancy + (1.0 - gate[..., None]) * uniform_occupancy
    denominator = occupancy.sum(dim=-1, keepdim=True)
    last = (valid.long().sum(dim=-1) - 1).clamp_min(0)
    endpoint = F.one_hot(last, valid.shape[-1]).to(hidden.dtype)
    endpoint = endpoint[:, :, None, :].expand(-1, -1, len(horizon), -1)
    weights = torch.where(denominator > 1e-06, occupancy / denominator.clamp_min(1e-06), endpoint)
    temporal = torch.einsum('bmtp,bmpw->bmtw', weights, hidden)
    horizon_feature = self.horizon_embedding[None, None].expand(temporal.shape[0], temporal.shape[1], -1, -1)
    temporal = self.time_fusion(torch.cat((temporal, horizon_feature), dim=-1))
    route_summary = temporal.mean(dim=-2)
    return {'temporal': temporal, 'route_summary': route_summary, 'traversal_speed_mps': traversal_speed, 'segment_duration_s': duration, 'segment_entry_time_s': entry_time, 'segment_exit_time_s': exit_time, 'horizon_road_weight': weights, 'horizon_time_gate': gate, 'route_entry_xy': entry_xyz[..., :2], 'route_exit_xy': exit_xyz[..., :2], 'segment_valid': valid}

class TimeGateVariant(base.AnchoredVariant):
    """Height anchor plus a matched-capacity occupancy trust gate."""

    def __init__(self, initialized_anchor, arm):
        super().__init__(initialized_anchor, arm)
        cfg = self._cfg
        if arm == 'joint-timegate-decoupled-anchor' and (not hasattr(self, 'height_gate')):
            head = self.cartesian.xyz_head
            width = head.in_features
            reference = head.weight
            self.height_gate = nn.Sequential(nn.Linear(width + 3, width, device=reference.device, dtype=reference.dtype), nn.GELU(), nn.Linear(width, 1, device=reference.device, dtype=reference.dtype))
            nn.init.zeros_(self.height_gate[-1].weight)
            nn.init.zeros_(self.height_gate[-1].bias)
        encoder = self.road.road_graph
        width = encoder.horizon_embedding.shape[1]
        gate = nn.Sequential(nn.Linear(2 * width + 5, width, device=encoder.horizon_embedding.device, dtype=encoder.horizon_embedding.dtype), nn.GELU(), nn.Linear(width, 1, device=encoder.horizon_embedding.device, dtype=encoder.horizon_embedding.dtype))
        nn.init.zeros_(gate[-1].weight)
        nn.init.zeros_(gate[-1].bias)
        self.time_gate = gate
        object.__setattr__(encoder, 'time_gate_ref', gate)
        object.__setattr__(encoder, 'time_gate_mode', LAMBDA_MODE[arm])
        self.time_gate_arm = arm

class TimeGateVariant6(TimeGateVariant):
    """v3 variant with a reachable gate and the matched-capacity height gate."""

    def __init__(self, initialized_anchor, arm):
        super().__init__(initialized_anchor, arm)
        object.__setattr__(self.road.road_graph, 'time_gate_gain', GATE_GAIN)
        if arm == 'joint-timegate6-decoupled-anchor' and (not hasattr(self, 'height_gate')):
            head = self.cartesian.xyz_head
            width = head.in_features
            reference = head.weight
            self.height_gate = v3.nn.Sequential(v3.nn.Linear(width + 3, width, device=reference.device, dtype=reference.dtype), v3.nn.GELU(), v3.nn.Linear(width, 1, device=reference.device, dtype=reference.dtype))
            v3.nn.init.zeros_(self.height_gate[-1].weight)
            v3.nn.init.zeros_(self.height_gate[-1].bias)

class OccupancyEncoder(TravelTimeRoadGraphEncoder):
    forward = _occupancy_gate_forward

class CurrentRoadToken3D(TimeGateVariant6):
    """Same named modules/parameter keys as the current paper's v6 family."""

    def __init__(self, initialized_anchor, arm='joint-timegate6-anchor'):
        if arm not in ARMS_V6:
            raise ValueError('Only the six current v6 configurations are supported')
        super().__init__(initialized_anchor, arm)
        self.road.road_graph.__class__ = OccupancyEncoder

def from_parents(cartesian, road, *, arm='joint-timegate6-anchor', hops=48, message_width=32, checkpoint_chunk=8, seed=17):
    torch.manual_seed(seed)
    anchor = Base(cartesian, road, hops=hops, message_width=message_width, checkpoint_chunk=checkpoint_chunk)
    model = CurrentRoadToken3D(anchor, arm)
    torch.manual_seed(seed)
    return model
