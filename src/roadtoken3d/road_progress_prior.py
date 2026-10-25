"""Observation-prediction-conditioned intrinsic progress on connected roads.

A Cartesian guide is not a speed sensor: its point-to-point Euclidean distance
is not the arc coordinate on a curved candidate road. This tensor-only prior
expresses the guide in each candidate's road coordinate before the learned
3D-motion correction. It is also an explicit projection-only control, not a
standalone novelty or proof that learned road/graph components help.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def intrinsic_progress_prior(geometry, roads, directions, count, start_s,
                             length, origin_local, guide_xyz, *, samples=257,
                             interval_seconds=15., maximum_speed_mps=60.,
                             vertical_metric_weight=.1):
    if samples < 3 or interval_seconds <= 0 or maximum_speed_mps <= 0:
        raise ValueError("Positive cadence/speed and at least three curve samples required")
    if vertical_metric_weight < 0:
        raise ValueError("Vertical metric weight cannot be negative")
    grid = torch.linspace(0., 1., samples, device=guide_xyz.device,
                          dtype=torch.float32)[None, None] * length[..., None].float()
    curve = geometry(roads, directions, count, start_s, grid)["xyz"].float()
    curve = curve - origin_local[:, None, None].float()
    start = curve[..., :-1, :]
    segment = curve[..., 1:, :] - start
    metric = guide_xyz.new_tensor([1., 1., vertical_metric_weight]).float()
    displacement = guide_xyz[:, None, :, None].float() - start[:, :, None]
    dot = (displacement * segment[:, :, None] * metric).sum(-1)
    norm = (segment.square() * metric).sum(-1).clamp_min(1e-8)
    fraction = (dot / norm[:, :, None]).clamp(0., 1.)
    horizon = torch.arange(1, guide_xyz.shape[1] + 1,
                           device=guide_xyz.device, dtype=torch.float32)
    reachable = horizon * interval_seconds * maximum_speed_mps
    interval = (grid[..., 1:] - grid[..., :-1]).clamp_min(1e-8)
    upper_fraction = ((reachable[None, None, :, None] - grid[:, :, None, :-1]) /
                      interval[:, :, None]).clamp(0., 1.)
    fraction = torch.minimum(fraction, upper_fraction)
    closest = start[:, :, None] + fraction[..., None] * segment[:, :, None]
    distance = ((guide_xyz[:, None, :, None].float() - closest).square() * metric).sum(-1)
    distance = distance.masked_fill(grid[:, :, None, :-1] > reachable[None, None, :, None], float("inf"))
    chosen = distance.argmin(-1)
    chosen_fraction = fraction.gather(-1, chosen[..., None]).squeeze(-1)
    selected_start = grid[..., :-1].gather(-1, chosen)
    selected_interval = interval.gather(-1, chosen)
    raw = selected_start + chosen_fraction * selected_interval
    # Explicit small time loop uses accelerator tensor maxima. No cummax/CPU
    # graph search, and a nearest point on a later loop cannot jump >60 m/s.
    previous = torch.zeros_like(raw[..., 0])
    progress = []
    for point in raw.unbind(-1):
        current = torch.minimum(torch.maximum(point, previous),
                                previous + interval_seconds * maximum_speed_mps)
        current = torch.minimum(current, length)
        progress.append(current)
        previous = current
    progress = torch.stack(progress, -1)
    speed = torch.diff(F.pad(progress, (1, 0)), dim=-1) / interval_seconds
    end_tangent = F.normalize(curve[..., -1, :2] - curve[..., -2, :2], dim=-1, eps=.01)
    beyond_end = ((guide_xyz[:, None, :, :2] - curve[:, :, None, -1, :2]) *
                  end_tangent[:, :, None]).sum(-1)
    saturated = (progress >= length[..., None] - 1e-3) & (beyond_end > 5.)
    return {"progress_m": progress, "speed_mps": speed,
            "endpoint_saturated": saturated, "beyond_endpoint_m": beyond_end,
            "raw_progress_m": raw}
