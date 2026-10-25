"""V12 road-relative motion with bounded offsets and a learned XY route guide.

The route bank remains observation/map-only. Future labels enter only v12_loss.
The geometric guarantee is proximity to a source 3D centerline, not certified
lane occupancy: the available map has no lane-width/traffic-legality labels.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .route_model_v11 import RoadNativeV11
from .route_model_v9 import ResidualMLP

class RoadNativeV12(RoadNativeV11):

    def __init__(self, graph, *args, maximum_lateral_m: float=5.0, maximum_height_residual_m: float=10.0, **kwargs) -> None:
        super().__init__(graph, *args, **kwargs)
        if maximum_lateral_m <= 0 or maximum_height_residual_m <= 0:
            raise ValueError('Road-relative offset bounds must be positive')
        self.api_version = 12
        self.maximum_lateral_m = float(maximum_lateral_m)
        self.maximum_height_residual_m = float(maximum_height_residual_m)
        width = self.speed_head.in_features
        self.lateral_head = nn.Linear(width, 1)
        self.height_residual_head = nn.Linear(width, 1)
        nn.init.zeros_(self.lateral_head.weight)
        nn.init.zeros_(self.lateral_head.bias)
        nn.init.zeros_(self.height_residual_head.weight)
        nn.init.zeros_(self.height_residual_head.bias)
        self.guide_position = nn.Parameter(torch.randn(self.future_steps, width) * 0.02)
        self.guide_fusion = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), ResidualMLP(width))
        self.guide_head = nn.Linear(width, 2)
        nn.init.zeros_(self.guide_head.weight)
        nn.init.zeros_(self.guide_head.bias)
        self.guide_rank_strength = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('guide_scale_m', (torch.arange(1, self.future_steps + 1).float() * self.dt / 60.0).sqrt().clamp_min(0.5) * 100.0)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        output = super().forward(*args, **kwargs)
        route_roads = args[10] if len(args) > 10 else kwargs['route_roads']
        route_directions = args[11] if len(args) > 11 else kwargs['route_directions']
        route_start_s = args[14] if len(args) > 14 else kwargs['route_start_s']
        history_state = output.pop('_history_state')
        output['_road_history_state'] = history_state
        motion = output.pop('_motion_feature')
        output['_road_motion_feature'] = motion
        progress = output['progress_m']
        count = output['strict_route_count']
        before = self.geometry(route_roads, route_directions, count, route_start_s, (progress - 1.0).clamp_min(0))['xyz'][..., :2]
        after = self.geometry(route_roads, route_directions, count, route_start_s, progress + 1.0)['xyz'][..., :2]
        tangent = after - before
        tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(0.01)
        normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
        lateral = self.maximum_lateral_m * self.lateral_head(motion).float().tanh().squeeze(-1)
        height_residual = self.maximum_height_residual_m * self.height_residual_head(motion).float().tanh().squeeze(-1)
        centerline = output['trajectories'].float()
        trajectory = torch.cat((centerline[..., :2] + lateral[..., None] * normal, centerline[..., 2:3] + height_residual[..., None]), dim=-1)
        output['trajectories'] = trajectory
        output['road_relative_lateral_m'] = lateral
        output['road_relative_height_residual_m'] = height_residual
        output['centerline_trajectories'] = centerline
        output['road_xy_distance_bound_m'] = self.maximum_lateral_m
        output['road_z_residual_bound_m'] = self.maximum_height_residual_m
        output['velocity_xyz_mps'] = torch.diff(torch.cat((torch.zeros_like(trajectory[:, :, :1]), trajectory), dim=2), dim=2) / self.dt
        horizon = self.guide_position[None].expand(len(history_state), -1, -1)
        guide_hidden = self.guide_fusion(torch.cat((history_state[:, None].expand_as(horizon), horizon), dim=-1))
        guide_xy = self.guide_head(guide_hidden).float() * self.guide_scale_m[None, :, None]
        guide_distance = torch.linalg.vector_norm(trajectory[..., :2] - guide_xy[:, None], dim=-1).mean(-1)
        output['guide_xy'] = guide_xy
        output['guide_distance_m'] = guide_distance
        output['mode_logits'] = output['mode_logits'] - F.softplus(self.guide_rank_strength) * guide_distance / 100.0
        return output
