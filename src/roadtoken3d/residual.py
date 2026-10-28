"""Shared PyTorch decoder for Cartesian and road-relative displacements.

The forecast is C + (Q_corrected - Q_reference). Only the correction branch
is decoded on road geometry; final XYZ coordinates need not lie on a road.
Geometry uses the map's stored s_road coordinate and strict XYZ joins.
Checkpoint loading requires the same parent construction and immutable
geometry, including nonpersistent geometry buffers. The current loss module
validates the output contract before supervising final coordinates.
"""
from __future__ import annotations
from .prefix import cumsum as _prefix_sum
import torch
from torch import nn
from torch.nn import functional as F
from roadtoken3d.road_progress_prior import intrinsic_progress_prior
OBSERVATION_KEYS = ('history_xyz', 'history_xy_valid', 'history_z_valid', 'origin_context', 'time_features', 'history_matched_xyz', 'history_road_tangent', 'history_road_d', 'history_map_gap_m', 'history_map_confidence', 'route_roads', 'route_directions', 'route_count', 'route_valid', 'route_start_s', 'origin_local', 'local_state', 'local_node_valid', 'local_successors', 'local_strict_successor_valid', 'local_predecessors', 'local_strict_predecessor_valid', 'local_start_valid', 'local_start_distance_m', 'local_start_node_index', 'local_start_parent')

class RoadDisplacementResidualFusion(nn.Module):
    """Keep the direct forecast at initialization, learn intrinsic corrections.

    Supply the two parent *modules*, before adding a delayed graph. Their
    existing parameters are frozen. Only the newly attached travel-time graph,
    its node-speed adjustment, and the four-output correction head are trained.
    Every candidate uses the same single road forward for reference/correction;
    no forward hooks, mutable cross-forward cache, or future labels are used.
    """

    def __init__(self, cartesian: nn.Module, road: nn.Module, *, hops: int=48, message_width: int=32, checkpoint_chunk: int=8) -> None:
        super().__init__()
        if cartesian.modes != 1 or cartesian.future_steps != road.future_steps:
            raise ValueError('Single-mode Cartesian parent and equal horizons required')
        if cartesian.architecture != 'attention':
            raise ValueError('The candidate needs observation/horizon attention features')
        if cartesian.xyz_head.in_features != road.speed_head.in_features:
            raise ValueError('Cartesian and road feature widths must match')
        if road.proposer.delayed_messages is not None:
            raise ValueError('Supply the frozen road parent before adding a delayed graph')
        if hops < 1 or message_width < 1 or checkpoint_chunk < 0:
            raise ValueError('Invalid delayed graph capacity')
        if float(road.dt) != 15.0 or float(cartesian.interval_seconds) != float(road.dt):
            raise ValueError('The inherited V13 graph requires matched 15-second cadence')
        (self.cartesian, self.road) = (cartesian, road)
        (self.future_steps, self.dt) = (road.future_steps, float(road.dt))
        self.frozen_road_parent_parameter_names = tuple(dict(road.named_parameters()))
        self.cartesian.requires_grad_(False)
        self.road.requires_grad_(False)
        road.proposer.enable_delayed_messages('travel-time', hops, message_width)
        road.proposer.delayed_messages.checkpoint_chunk = checkpoint_chunk
        road.proposer.time_aware = True
        road.proposer.suppress_arrival_feature = False
        road.road_graph.time_aware = True
        width = cartesian.xyz_head.in_features
        self.correction = nn.Sequential(nn.Linear(width * 3 + 5, width), nn.GELU(), nn.Linear(width, 4))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.train()

    def train(self, mode: bool=True):
        super().train(mode)
        self.cartesian.eval()
        self.road.eval()
        self.road.proposer.delayed_messages.train(mode)
        return self

    def _decode(self, roads, directions, count, start_s, origin, speed, lateral, height):
        """Integrate in stored map arc coordinates, then sample actual 3D XYZ."""
        progress = _prefix_sum(speed * self.dt, -1)
        geometry = self.road.geometry
        center = geometry(roads, directions, count, start_s, progress)
        center_xyz = center['xyz'].float() - origin[:, None, None].float()
        before = geometry(roads, directions, count, start_s, (progress - 1.0).clamp_min(0.0))['xyz'][..., :2].float()
        after = geometry(roads, directions, count, start_s, progress + 1.0)['xyz'][..., :2].float()
        tangent = F.normalize(after - before, dim=-1, eps=0.01)
        normal = torch.stack((-tangent[..., 1], tangent[..., 0]), -1)
        xyz = torch.cat((center_xyz[..., :2] + lateral[..., None] * normal, center_xyz[..., 2:3] + height[..., None]), -1)
        return (xyz, progress, center)

    def forward(self, *road_inputs: torch.Tensor) -> dict:
        if len(road_inputs) != len(OBSERVATION_KEYS):
            raise ValueError('Exactly the 26 observation/map-only V13 inputs are required')
        origin = road_inputs[15]
        with torch.no_grad():
            direct_output = self.cartesian(*road_inputs[:5])
            direct = direct_output['trajectories'][:, 0].float()
            direct_speed = torch.linalg.vector_norm(torch.diff(torch.cat((torch.zeros_like(direct[:, :1, :2]), direct[..., :2]), 1), dim=1), dim=-1).div(self.dt).clamp(0.0, 60.0)
        if direct.shape != (len(road_inputs[0]), self.future_steps, 3):
            raise ValueError('Cartesian output differs from the declared horizon')
        road = self.road(*road_inputs, motion_prior_speed=direct_speed.mean(-1))
        roads = road['generated_route_roads']
        directions = road['generated_route_directions']
        count = road['strict_route_count']
        start_s = road['generated_route_start_s']
        with torch.no_grad():
            intrinsic = intrinsic_progress_prior(self.road.geometry, roads, directions, count, start_s, road['route_length_m'], origin, direct, interval_seconds=self.dt)
        base_speed = intrinsic['speed_mps'].clamp(0.0, 60.0)
        base_lateral = road['road_relative_lateral_m'].float().clamp(-self.road.maximum_lateral_m, self.road.maximum_lateral_m)
        base_height = road['road_relative_height_residual_m'].float().clamp(-self.road.maximum_height_residual_m, self.road.maximum_height_residual_m)
        motion = road['_road_motion_feature']
        latent = direct_output['latent_horizon'].expand_as(motion)
        time_context = road['time_message_context'][:, None].expand_as(motion)
        direct_context = direct[:, None].expand(-1, motion.shape[1], -1, -1)
        direct_context = direct_context / direct.new_tensor([1000.0, 1000.0, 20.0])
        motion_scalar = torch.stack((direct_speed[:, None].expand_as(base_speed) / 10.0, base_speed / 10.0), -1)
        raw = self.correction(torch.cat((latent, motion, time_context, direct_context, motion_scalar), -1)).float()
        speed = (base_speed + 10.0 * raw[..., 0].tanh()).clamp(0.0, 60.0)
        lateral = (base_lateral + 5.0 * raw[..., 1].tanh()).clamp(-self.road.maximum_lateral_m, self.road.maximum_lateral_m)
        height = (base_height + 10.0 * raw[..., 2].tanh()).clamp(-self.road.maximum_height_residual_m, self.road.maximum_height_residual_m)
        (reference, reference_progress, reference_center) = self._decode(roads, directions, count, start_s, origin, base_speed, base_lateral, base_height)
        (corrected, progress, center) = self._decode(roads, directions, count, start_s, origin, speed, lateral, height)
        displacement = corrected - reference
        trajectories = direct[:, None] + displacement
        guide_distance = torch.linalg.vector_norm(reference[..., :2] - direct[:, None, :, :2], dim=-1).mean(-1)
        logits = -guide_distance / 100.0 + raw[..., 3].mean(-1)
        valid = road['mode_valid'].bool()
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        output = {'trajectories': trajectories, 'mode_logits': logits, 'mode_valid': valid, 'forecast_valid': valid.any(-1), 'cartesian_backbone_xyz': direct, 'interval_seconds': self.dt, 'road_geometry_constraint': False, 'metadata': {'model_family': 'road-displacement-residual-candidate-v1', 'preregistered': False, 'development_only': True, 'final_road_geometry_constraint': False, 'certified_traffic_legality': False, 'correction_branch_geometry_is_final_output_geometry': False, 'old_v13_v17_loss_compatible': False, 'final_equation': 'C + (Q_corrected - Q_reference)', 'no_candidate_policy': 'explicit_abstain_via_forecast_valid'}, 'correction_branch_reference_xyz': reference, 'correction_branch_corrected_xyz': corrected, 'correction_branch_displacement_xyz': displacement, 'correction_branch_raw_delta': raw, 'correction_branch_reference_progress_m': reference_progress, 'correction_branch_reference_speed_mps': base_speed, 'correction_branch_reference_lateral_m': base_lateral, 'correction_branch_reference_height_residual_m': base_height, 'correction_branch_progress_m': progress, 'correction_branch_decoded_progress_m': torch.minimum(progress, road['route_length_m'][..., None]), 'correction_branch_requested_speed_mps': speed, 'correction_branch_lateral_m': lateral, 'correction_branch_height_residual_m': height, 'correction_branch_point_road_id': center['road'], 'correction_branch_point_road_s': center['s'], 'correction_branch_on_connector': center['on_connector'], 'correction_branch_clamped_at_route_end': center['clamped_at_route_end'], 'correction_branch_reference_clamped_at_route_end': reference_center['clamped_at_route_end'], 'correction_branch_intrinsic_endpoint_saturated': intrinsic['endpoint_saturated'], 'correction_branch_route_roads': roads, 'correction_branch_route_directions': directions, 'correction_branch_strict_route_count': count, 'correction_branch_route_start_s': start_s}
        for key in ('time_message_context', 'time_message_arrival_mass', 'time_message_traversal_s', 'time_message_speed_mps', 'graph_temporal_node_logits', 'graph_node_reachable', 'graph_earliest_arrival_s', 'graph_motion_prior_speed_mps'):
            output[key] = road[key]
        return output

def road_displacement_forward(model: RoadDisplacementResidualFusion, batch: dict[str, torch.Tensor]) -> dict:
    """Explicit observation whitelist: target/teacher entries are never read."""
    return model(*(batch[key] for key in OBSERVATION_KEYS))
