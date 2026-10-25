"""Road-relative 3D decoding with travel-time-aware road-graph reasoning.

V11 keeps the audited V10 output contract: every accepted XYZ point is decoded
from a supplied three-dimensional road polyline.  Its new component represents
each directed road piece as a graph node, passes messages along predecessor and
successor relations, and reads the graph at the time horizon when a vehicle is
expected to reach each piece.  No future observation or teacher tensor enters
the forward pass.
"""
from __future__ import annotations
from .prefix import cumsum as _prefix_sum
import torch
from torch import nn
from torch.nn import functional as F
from .road_model import AscendCausalEncoder
from .route_model_v9 import ResidualMLP
from .route_model_v10 import StrictRoadGeometry

class DirectedRoadGraphLayer(nn.Module):
    """Dense, Ascend-friendly message passing on directed route subgraphs."""

    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.self_projection = nn.Linear(width, width)
        self.predecessor_projection = nn.Linear(width, width, bias=False)
        self.successor_projection = nn.Linear(width, width, bias=False)
        self.normalization = nn.LayerNorm(width)

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        pieces = hidden.shape[-2]
        shift = min(self.dilation, pieces)
        zero = hidden[..., :shift, :] * 0.0
        if shift == pieces:
            predecessor = successor = torch.zeros_like(hidden)
        else:
            predecessor = torch.cat((zero, hidden[..., :-shift, :]), dim=-2)
            successor = torch.cat((hidden[..., shift:, :], zero), dim=-2)
        predecessor_valid = torch.cat((valid[..., :shift] & False, valid[..., :-shift]), dim=-1) if shift < pieces else torch.zeros_like(valid)
        successor_valid = torch.cat((valid[..., shift:], valid[..., :shift] & False), dim=-1) if shift < pieces else torch.zeros_like(valid)
        message = self.self_projection(hidden)
        message = message + self.predecessor_projection(predecessor) * predecessor_valid[..., None]
        message = message + self.successor_projection(successor) * successor_valid[..., None]
        return self.normalization(hidden + F.gelu(message)) * valid[..., None]

class TravelTimeRoadGraphEncoder(nn.Module):
    """Encode a directed road sequence and query it by physical arrival time.

    Graph hops are used only for topology.  Horizon relevance is determined by
    cumulative learned traversal time, so one hop is not assumed to represent a
    fixed number of seconds.
    """

    def __init__(self, width: int, future_steps: int, interval_seconds: float, time_aware: bool=True) -> None:
        super().__init__()
        self.interval_seconds = float(interval_seconds)
        self.time_aware = bool(time_aware)
        self.node_projection = nn.Sequential(nn.Linear(10, width), nn.GELU(), nn.LayerNorm(width))
        self.layers = nn.ModuleList((DirectedRoadGraphLayer(width, dilation) for dilation in (1, 2, 4)))
        self.traversal_speed = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.traversal_speed[-1].weight)
        nn.init.zeros_(self.traversal_speed[-1].bias)
        self.horizon_embedding = nn.Parameter(torch.randn(future_steps, width) * 0.02)
        self.time_fusion = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), ResidualMLP(width))

    def forward(self, entry_xyz: torch.Tensor, exit_xyz: torch.Tensor, segment_length_m: torch.Tensor, valid: torch.Tensor, history_state: torch.Tensor, observed_speed_mps: torch.Tensor) -> dict[str, torch.Tensor]:
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
        if self.time_aware:
            occupancy = torch.sigmoid((horizon[None, None, :, None] - entry_time[:, :, None, :]) / temperature) - torch.sigmoid((horizon[None, None, :, None] - exit_time[:, :, None, :]) / temperature)
            occupancy = occupancy.clamp_min(0.0) * valid[:, :, None, :]
        else:
            occupancy = valid[:, :, None, :].expand(-1, -1, len(horizon), -1)
        denominator = occupancy.sum(dim=-1, keepdim=True)
        last = (valid.long().sum(dim=-1) - 1).clamp_min(0)
        endpoint = F.one_hot(last, valid.shape[-1]).to(hidden.dtype)
        endpoint = endpoint[:, :, None, :].expand(-1, -1, len(horizon), -1)
        weights = torch.where(denominator > 1e-06, occupancy / denominator.clamp_min(1e-06), endpoint)
        temporal = torch.einsum('bmtp,bmpw->bmtw', weights, hidden)
        horizon_feature = self.horizon_embedding[None, None].expand(temporal.shape[0], temporal.shape[1], -1, -1)
        temporal = self.time_fusion(torch.cat((temporal, horizon_feature), dim=-1))
        route_summary = temporal.mean(dim=-2)
        return {'temporal': temporal, 'route_summary': route_summary, 'traversal_speed_mps': traversal_speed, 'segment_duration_s': duration, 'segment_entry_time_s': entry_time, 'segment_exit_time_s': exit_time, 'horizon_road_weight': weights}

class RoadNativeV11(nn.Module):
    """V10-compatible road-only forecaster with time-aware graph context."""

    def __init__(self, graph, history_steps: int=41, future_steps: int=20, width: int=128, layers: int=2, dropout: float=0.1, interval_seconds: float=15.0, maximum_anchor_gap_m: float=25.0, connection_tolerance_m: float=0.001, time_aware_graph: bool=True) -> None:
        super().__init__()
        if history_steps < 5 or width % 4 or interval_seconds <= 0:
            raise ValueError('Need >=5 history steps, width divisible by 4, positive cadence')
        (self.api_version, self.future_steps) = (11, future_steps)
        self.dt = float(interval_seconds)
        self.maximum_anchor_gap_m = float(maximum_anchor_gap_m)
        self.connection_tolerance_m = float(connection_tolerance_m)
        self.geometry = StrictRoadGeometry(graph)
        for name in ('xyz', 'arc', 'ptr', 'length'):
            self.geometry.register_buffer(name, getattr(self.geometry, name), persistent=False)
        self.history_project = nn.Linear(15, width)
        self.position = nn.Parameter(torch.randn(history_steps, width) * 0.02)
        self.temporal = AscendCausalEncoder(width, 4, layers, width * 4, dropout, history_steps)
        self.history_summary = nn.Sequential(nn.Linear(width * 2 + 4, width), nn.GELU(), ResidualMLP(width))
        self.road_graph = TravelTimeRoadGraphEncoder(width, future_steps, interval_seconds, time_aware=time_aware_graph)
        self.fusion = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), ResidualMLP(width))
        self.motion_fusion = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), ResidualMLP(width))
        self.speed_head = nn.Linear(width, 1)
        self.height_head = nn.Linear(width, 1)
        self.route_head = nn.Linear(width, 1)
        nn.init.zeros_(self.speed_head.weight)
        nn.init.zeros_(self.speed_head.bias)

    def forward(self, history_xyz, history_xy_valid, history_z_valid, origin_context, time_features, matched_xyz, road_tangent, road_d, map_gap, map_confidence, route_roads, route_directions, route_count, route_valid, route_start_s, origin_local):
        hv = history_xy_valid.bool()
        zv = (history_z_valid & history_xy_valid).bool()
        xy = torch.where(hv[..., None], history_xyz[..., :2], 0.0)
        z = torch.where(zv, history_xyz[..., 2], 0.0)
        pair = hv & torch.cat((hv[:, :1], hv[:, :-1]), dim=1)
        zpair = zv & torch.cat((zv[:, :1], zv[:, :-1]), dim=1)
        delta = torch.diff(torch.cat((xy[:, :1], xy), dim=1), dim=1) * pair[..., None]
        dz = torch.diff(torch.cat((z[:, :1], z), dim=1), dim=1) * zpair
        heading = xy[:, -1] - xy[:, -5]
        norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
        default = torch.zeros_like(heading)
        default[:, 0] = 1.0
        heading = torch.where((norm > 0.1) & hv[:, -5:].all(-1, keepdim=True), heading / norm.clamp_min(0.1), default)

        def rotate(value):
            direction = heading.reshape(len(heading), *[1] * (value.ndim - 2), 2)
            return torch.stack(((value * direction).sum(-1), value[..., 1] * direction[..., 0] - value[..., 0] * direction[..., 1]), dim=-1)
        confidence = torch.where(hv, map_confidence, 0.0).clamp(0, 1)
        matched_xy = torch.where(hv[..., None], matched_xyz[..., :2], 0.0)
        matched_z = torch.where(zv, matched_xyz[..., 2], 0.0)
        tangent = torch.where(hv[..., None], road_tangent, 0.0)
        lateral = torch.where(hv, road_d, 0.0)
        history_features = torch.cat((rotate(xy) / 1000.0, rotate(delta) / 100.0, z[..., None] / 20.0, dz[..., None] / 5.0, hv[..., None], zv[..., None], rotate(matched_xy) / 1000.0 * confidence[..., None], matched_z[..., None] / 20.0 * confidence[..., None], rotate(tangent) * confidence[..., None], confidence[..., None], lateral[..., None] / 25.0), dim=-1)
        encoded = self.temporal(self.history_project(history_features) + self.position[None])
        pooled = (encoded * hv[..., None]).sum(1) / hv.sum(1, keepdim=True).clamp_min(1)
        history_state = self.history_summary(torch.cat((encoded[:, -1], pooled, time_features), dim=-1))
        (count, join_gap) = self.geometry.continuous_prefix(route_roads, route_directions, route_count, route_start_s, self.connection_tolerance_m)
        recent = pair[:, -4:]
        observed_speed = ((torch.linalg.vector_norm(delta[:, -4:], dim=-1) * recent).sum(-1) / (self.dt * recent.sum(-1).clamp_min(1))).clamp(0, 60)
        (start, end, entry, exit, _, _, cumulative) = self.geometry.route_parts(route_roads, route_directions, count, route_start_s)
        segment_length = (end - start).abs()
        piece_valid = torch.arange(route_roads.shape[-1], device=route_roads.device) < count[..., None]
        entry_local = entry - origin_local[:, None, None]
        exit_local = exit - origin_local[:, None, None]
        graph = self.road_graph(entry_local, exit_local, segment_length, piece_valid, history_state, observed_speed)
        length = cumulative[..., -1]
        anchor = torch.linalg.vector_norm(entry_local[:, :, 0, :2], dim=-1)
        feasible = route_valid & (count > 0) & (length > 0.1) & (anchor <= self.maximum_anchor_gap_m)
        joint = self.fusion(torch.cat((history_state[:, None].expand_as(graph['route_summary']), graph['route_summary']), dim=-1))
        motion = self.motion_fusion(torch.cat((joint[:, :, None].expand_as(graph['temporal']), graph['temporal']), dim=-1))
        requested_speed = (observed_speed[:, None, None] + 10.0 * self.speed_head(motion).squeeze(-1).float()).clamp(0, 60)
        progress = _prefix_sum(requested_speed * self.dt, -1)
        decoded = self.geometry(route_roads, route_directions, count, route_start_s, progress)
        trajectories = decoded['xyz'] - origin_local[:, None, None]
        decoded_progress = torch.minimum(progress, length[..., None])
        realized_speed = torch.diff(F.pad(decoded_progress, (1, 0)), dim=-1) / self.dt
        velocity_xyz = torch.diff(torch.cat((torch.zeros_like(trajectories[:, :, :1]), trajectories), dim=2), dim=2) / self.dt
        logits = self.route_head(joint).squeeze(-1).float().masked_fill(~feasible, -10000.0)
        auxiliary_height = self.height_head(motion).squeeze(-1).float() * 20.0
        zeros = torch.zeros_like(progress)
        return {'trajectories': trajectories, 'mode_logits': logits, 'mode_valid': feasible, 'forecast_valid': feasible.any(-1), 'effective_route_valid': feasible, 'progress_m': progress, 'decoded_progress_m': decoded_progress, 'speed_mps': realized_speed, 'requested_speed_mps': requested_speed, 'velocity_xyz_mps': velocity_xyz, 'raw_speed': requested_speed, 'route_length_m': length, 'anchor_gap_m': anchor, 'strict_route_count': count, 'route_prefix_truncated': count < route_count, 'point_road_id': decoded['road'], 'point_road_s': decoded['s'], 'on_connector': decoded['on_connector'], 'clamped_at_route_end': decoded['clamped_at_route_end'], 'auxiliary_height_m': auxiliary_height, 'interval_seconds': self.dt, 'road_relative_progress_m': decoded_progress, 'road_relative_lateral_m': zeros, 'road_relative_height_residual_m': zeros, 'graph_traversal_speed_mps': graph['traversal_speed_mps'], 'graph_segment_duration_s': graph['segment_duration_s'], 'graph_segment_entry_time_s': graph['segment_entry_time_s'], 'graph_segment_exit_time_s': graph['segment_exit_time_s'], 'graph_horizon_road_weight': graph['horizon_road_weight'], 'join_gap_m': join_gap, '_history_state': history_state, '_motion_feature': motion}
