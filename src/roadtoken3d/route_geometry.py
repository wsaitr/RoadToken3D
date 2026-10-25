"""Continuous source-road routes and a tensor-only arc-length decoder.

Connectivity is geometric, not certified traffic legality. The preliminary
offline route proposal uses ONLY the observed endpoint and public map. Coverage
must be measured before using it as a prediction model's feasible domain.
"""
from __future__ import annotations
from .prefix import cumsum as _prefix_sum
import numpy as np
import torch
from torch import nn

class NativeRouteGeometry(nn.Module):
    """Sample full polylines by distance, without CPU graph search or projection.

    All road curves are shared device buffers; each sample stores only road IDs
    and directions. Bisection uses gather/compare, not unsupported searchsorted.
    The output includes connector flags so tiny map gaps cannot masquerade as
    verified drivable geometry. Traversing a curve never draws an endpoint chord.
    """

    def __init__(self, graph, reference=(72000.0, 3279000.0, 0.0)):
        super().__init__()
        reference = np.asarray(reference, np.float64)
        self.register_buffer('reference', torch.tensor(reference, dtype=torch.float32))
        self.register_buffer('xyz', torch.from_numpy((np.asarray(graph.points_xyz) - reference).astype(np.float32)))
        self.register_buffer('arc', torch.from_numpy(np.asarray(graph.s_road).astype(np.float32)))
        self.register_buffer('ptr', torch.from_numpy(np.asarray(graph.road_ptr).astype(np.int64)))
        self.register_buffer('length', torch.from_numpy(np.asarray(graph.road_length_m).astype(np.float32)))
        self.bisection_iterations = max(1, int(np.ceil(np.log2(max(int(np.diff(graph.road_ptr).max()), 2)))))

    def sample_road(self, roads, arc):
        shape = roads.shape
        (roads, arc) = (roads.reshape(-1).long(), arc.reshape(-1).float())
        (low, high) = (self.ptr[roads], self.ptr[roads + 1] - 1)
        (first, last) = (low.clone(), high.clone())
        arc = torch.minimum(torch.maximum(arc, self.arc[first]), self.arc[last])
        for _ in range(self.bisection_iterations):
            middle = (low + high + 1) // 2
            before = self.arc[middle] <= arc
            low = torch.where(before, middle, low)
            high = torch.where(before, high, middle - 1)
        left = torch.minimum(low, torch.maximum(first, last - 1))
        right = torch.minimum(left + 1, last)
        fraction = ((arc - self.arc[left]) / (self.arc[right] - self.arc[left]).clamp_min(1e-08)).clamp(0, 1)
        xyz = self.xyz[left] + fraction[:, None] * (self.xyz[right] - self.xyz[left])
        return xyz.reshape(*shape, 3)

    def route_parts(self, roads, directions, count, start_s):
        p = roads.shape[-1]
        mask = torch.arange(p, device=roads.device) < count[..., None]
        full = self.length[roads.long()]
        start = torch.where(directions > 0, torch.zeros_like(full), full)
        start = torch.cat((start_s[..., None], start[..., 1:]), -1)
        end = torch.where(directions > 0, full, torch.zeros_like(full))
        road_length = (end - start).abs() * mask
        (entry, exit) = (self.sample_road(roads, start), self.sample_road(roads, end))
        next_entry = torch.cat((entry[..., 1:, :], exit[..., -1:, :]), -2)
        connector = torch.linalg.vector_norm(next_entry[..., :2] - exit[..., :2], dim=-1)
        connector_mask = torch.arange(p, device=roads.device) + 1 < count[..., None]
        connector = connector * connector_mask
        lengths = torch.stack((road_length, connector), -1).flatten(-2)
        cumulative = _prefix_sum(lengths, -1)
        return (start, end, entry, exit, next_entry, lengths, cumulative)

    def forward(self, roads, directions, count, start_s, progress):
        (start, end, entry, exit, next_entry, lengths, cumulative) = self.route_parts(roads, directions, count, start_s)
        total = cumulative[..., -1:]
        distance = torch.minimum(progress.float().clamp_min(0), total)
        part = (distance[..., None] > cumulative[..., None, :]).long().sum(-1).clamp_max(lengths.shape[-1] - 1)
        base = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), -1).gather(-1, part)
        part_length = lengths.gather(-1, part)
        fraction = ((distance - base) / part_length.clamp_min(1e-08)).clamp(0, 1)
        road_part = part // 2
        chosen = roads.gather(-1, road_part)
        (arc_start, arc_end) = (start.gather(-1, road_part), end.gather(-1, road_part))
        arc = arc_start + fraction * (arc_end - arc_start)
        curve = self.sample_road(chosen, arc)
        gather_xyz = road_part[..., None].expand(*road_part.shape, 3)
        connector_start = exit.gather(-2, gather_xyz)
        connector_end = next_entry.gather(-2, gather_xyz)
        connector_xyz = connector_start + fraction[..., None] * (connector_end - connector_start)
        is_connector = part % 2 == 1
        xyz = torch.where(is_connector[..., None], connector_xyz, curve)
        return {'xyz': xyz, 'road': chosen, 's': arc, 'on_connector': is_connector, 'total_length': total, 'clamped_at_route_end': progress > total}
