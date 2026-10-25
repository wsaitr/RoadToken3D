"""Unregistered DEV revision of the road-displacement residual decoder.

The frozen v1 module and its 1600-update study are not changed or replaced.
This revision fixes only the loss of a lateral frame when requested progress
exceeds a candidate's endpoint. It does NOT extend a road, relax strict XYZ
joins, restore longitudinal sensitivity beyond a short prefix, repair map
heights, or certify traffic legality. A new study/adapter is required to train
or score this distinct model family; the v1 production admission must reject it.
"""
from __future__ import annotations
from .prefix import cumsum as _prefix_sum
import torch
from torch.nn import functional as F
from .residual import OBSERVATION_KEYS, RoadDisplacementResidualFusion
FAMILY = 'road-displacement-endpoint-stable-candidate-v2'

def endpoint_stable_decode(geometry, roads, directions, count, start_s, origin, speed, lateral, height, *, dt):
    """Sample the lateral frame at decoded arc, not unbounded requested arc.

    Centerline geometry still clamps at the actual prefix endpoint. The frame
    is sampled inside that SAME prefix, including reverse and partial roads.
    Zero-length/degenerate XY geometry has a zero frame, never an invented one.
    No host graph search, projection, future label or mutable cache is used.
    """
    progress = _prefix_sum(speed * dt, -1)
    center = geometry(roads, directions, count, start_s, progress)
    decoded_progress = torch.minimum(progress.float().clamp_min(0.0), center['total_length'].float())
    before = geometry(roads, directions, count, start_s, (decoded_progress - 1.0).clamp_min(0.0))['xyz'][..., :2].float()
    after = geometry(roads, directions, count, start_s, decoded_progress + 1.0)['xyz'][..., :2].float()
    chord = after - before
    tangent = F.normalize(chord, dim=-1, eps=0.01)
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), -1)
    center_xyz = center['xyz'].float() - origin[:, None, None].float()
    xyz = torch.cat((center_xyz[..., :2] + lateral[..., None] * normal, center_xyz[..., 2:3] + height[..., None]), -1)
    diagnostics = {**center, 'lateral_normal_xy': normal, 'lateral_frame_non_degenerate': torch.linalg.vector_norm(chord, dim=-1) >= 0.01, 'lateral_frame_sample_progress_m': decoded_progress}
    return (xyz, progress, diagnostics)

class EndpointStableRoadDisplacementResidualFusion(RoadDisplacementResidualFusion):
    """Same frozen parents, 3D relative correction and actual delayed graph."""

    def _decode(self, roads, directions, count, start_s, origin, speed, lateral, height):
        return endpoint_stable_decode(self.road.geometry, roads, directions, count, start_s, origin, speed, lateral, height, dt=self.dt)

    def forward(self, *road_inputs):
        output = super().forward(*road_inputs)
        output['metadata'] = {**output['metadata'], 'model_family': FAMILY, 'endpoint_lateral_frame': 'sampled_at_clamped_arc_inside_same_strict_prefix', 'short_prefix_longitudinal_saturation_repaired': False, 'strict_XYZ_connectivity_or_elevation_repaired': False, 'registered_v1_study_compatible': False}
        return output

def endpoint_stable_residual_forward(model, batch):
    """Same 26 observation/map-only inputs, no target/teacher selection."""
    return model(*(batch[key] for key in OBSERVATION_KEYS))
