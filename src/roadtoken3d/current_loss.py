"""Shared trajectory and graph objectives for the current RoadToken3D model.

This module supplies six loss terms; optimization.py adds the height- and
time-gate objectives. Detached teacher assignment supervises final XY/Z
coordinates, while graph supervision remains independent of candidate
acceptance. The explicit TRAIN split keeps hindsight teachers out of
validation and prediction. Shape, geometry, and model-family checks enforce
the interface contract; they are not an experiment-approval mechanism.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from roadtoken3d.time_delayed_graph import UNREACHED_SECONDS
from .current_model import FAMILIES, EQUATIONS
from .loss_helpers import (
    DEFAULT_WEIGHTS as _V1_DEFAULT_WEIGHTS, validate_loss_weights,
    _ratio, _masked_logsumexp, _finite_on,
)

DEFAULT_WEIGHTS = dict(_V1_DEFAULT_WEIGHTS)


def _check_v2_contract(output, batch):
    meta = output.get("metadata", {})
    if (not isinstance(meta, Mapping) or meta.get("model_family") != FAMILIES.get(meta.get("arm")) or
            meta.get("endpoint_lateral_frame") != "sampled_at_clamped_arc_inside_same_strict_prefix" or
            meta.get("short_prefix_longitudinal_saturation_repaired") is not False or
            meta.get("strict_XYZ_connectivity_or_elevation_repaired") is not False or
            meta.get("final_road_geometry_constraint") is not False or
            meta.get("correction_branch_geometry_is_final_output_geometry") is not False or
            meta.get("final_equation") != EQUATIONS.get(meta.get("arm")) or
            meta.get("old_v13_v17_loss_compatible") is not False or
            output.get("road_geometry_constraint") is not False):
        raise ValueError("The current RoadToken3D objective requires matching model and geometry metadata")
    from .current_model import validate_output
    validate_output(output)
    # A legacy geometry alias would make the output semantics ambiguous. This
    # loss has no arc/speed/road-proximity supervision for final coordinates.
    legacy = {"clamped_at_route_end", "point_road_id", "point_road_s", "on_connector",
              "progress_m", "decoded_progress_m", "requested_speed_mps", "speed_mps",
              "route_length_m", "road_relative_progress_m", "road_relative_lateral_m",
              "road_relative_height_residual_m", "centerline_trajectories",
              "road_xy_distance_bound_m", "road_z_residual_bound_m"}
    if legacy.intersection(output):
        raise ValueError("Legacy final-geometry aliases are not a residual-candidate contract")
    forecast = output["trajectories"]
    if (forecast.ndim != 4 or not forecast.is_floating_point() or
            forecast.shape[-1] != 3 or min(forecast.shape[:3]) < 1):
        raise ValueError("Nonempty B,K,T,3 final trajectories required")
    batch_size, modes, steps, _ = forecast.shape
    if (output["mode_valid"].shape != (batch_size, modes) or
            output["mode_valid"].dtype != torch.bool or
            output["mode_logits"].shape != (batch_size, modes) or
            not output["mode_logits"].is_floating_point()):
        raise ValueError("Candidate masks/logits must match final trajectories")
    if (output["forecast_valid"].shape != (batch_size,) or
            output["forecast_valid"].dtype != torch.bool or not torch.equal(
            output["forecast_valid"].bool(), output["mode_valid"].bool().any(-1))):
        raise ValueError("forecast_valid must retain the explicit candidate-abstention mask")
    _finite_on(forecast, output["mode_valid"][:, :, None, None], "accepted predicted XYZ")
    truth = batch["target_xyz"]
    if (truth.ndim != 3 or not truth.is_floating_point() or truth.shape[0] != batch_size or
            truth.shape[1] < steps or truth.shape[2] != 3):
        raise ValueError("B,T,3 supervised coordinates must cover the forecast horizon")
    for name in ("target_xy_valid", "target_z_valid", "target_road_id", "target_map_confidence"):
        value = batch[name]
        if value.ndim != 2 or value.shape[0] != batch_size or value.shape[1] < steps:
            raise ValueError(f"Mismatched horizon for {name}")
    if any(batch[name].dtype != torch.bool for name in ("target_xy_valid", "target_z_valid")):
        raise ValueError("Authoritative target validity masks must be boolean")
    integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    if batch["target_road_id"].dtype not in integer_types or not batch["target_map_confidence"].is_floating_point():
        raise ValueError("Integer road labels and floating graph confidence required")
    state = batch["local_state"]
    if state.ndim != 2 or state.dtype not in integer_types or state.shape[0] != batch_size or state.shape[1] < 1:
        raise ValueError("A padded local graph with at least one node slot is required")
    for value in (batch["local_node_valid"], output["graph_node_reachable"],
                  output["graph_earliest_arrival_s"]):
        if value.shape != state.shape:
            raise ValueError("Local graph masks/arrival times must match node slots")
    if (batch["local_node_valid"].dtype != torch.bool or
            output["graph_node_reachable"].dtype != torch.bool or
            not output["graph_earliest_arrival_s"].is_floating_point()):
        raise ValueError("Boolean graph masks and floating physical arrival times required")
    for name in ("graph_temporal_node_logits", "time_message_arrival_mass"):
        if output[name].shape != (batch_size, steps, state.shape[1]) or not output[name].is_floating_point():
            raise ValueError(f"Mismatched graph horizon for {name}")
    interval = output["interval_seconds"]
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval != 15.0:
        raise ValueError("The residual candidate requires its inherited 15-second cadence")
    return batch_size, modes, steps


def joint_residual_loss(output, batch, weights=None, return_parts=False, *, split="train"):
    """Supervise final coordinates and delayed graph; no future-input routing.

    The detached teacher set contains valid modes within one metre of the
    minimum XY ADE. A detached minimum (XY/100 + height SmoothL1/10) within that
    set selects the spatial training assignment. The set is also the target of
    a true masked log-probability loss; expected XY risk trains the distribution
    without claiming that hindsight assignment is forecast performance.
    """
    if split != "train":
        raise ValueError("Endpoint-stable teachers are TRAIN-only; split must be train")
    weights = validate_loss_weights(weights)
    batch_size, _, steps = _check_v2_contract(output, batch)
    forecast = output["trajectories"].float()
    valid_mode = output["mode_valid"].bool()
    accepted = valid_mode.any(-1)
    truth = batch["target_xyz"][:, :steps].float()
    declared_xy = batch["target_xy_valid"][:, :steps].bool()
    declared_z = batch["target_z_valid"][:, :steps].bool()
    xy_valid = declared_xy
    z_valid = declared_xy & declared_z
    _finite_on(truth[..., :2], xy_valid[..., None], "labeled target XY")
    _finite_on(truth[..., 2], z_valid, "labeled target height")
    xy_points = xy_valid.float().sum(-1)
    z_points = z_valid.float().sum(-1)
    available = accepted & (xy_points > 0)
    height_available = accepted & (z_points > 0)

    xy_mask = valid_mode[:, :, None, None] & xy_valid[:, None, :, None]
    _finite_on(forecast[..., :2], xy_mask, "predicted XY")
    xy_prediction = torch.where(xy_mask, forecast[..., :2], 0.)
    xy_truth = torch.where(xy_valid[..., None], truth[..., :2], 0.)
    xy_delta = torch.where(xy_mask, xy_prediction - xy_truth[:, None], 0.)
    xy_cost = _ratio(torch.linalg.vector_norm(xy_delta, dim=-1).sum(-1),
                     xy_points[:, None]) / 100.
    z_mask = valid_mode[:, :, None] & z_valid[:, None]
    _finite_on(forecast[..., 2], z_mask, "predicted height")
    z_prediction = torch.where(z_mask, forecast[..., 2], 0.)
    z_truth = torch.where(z_valid, truth[..., 2], 0.)
    z_point = F.smooth_l1_loss(z_prediction / 10.,
        (z_truth[:, None] / 10.).expand_as(z_prediction), reduction="none")
    height_cost = _ratio(torch.where(z_mask, z_point, 0.).sum(-1), z_points[:, None])

    detached_xy = xy_cost.detach()
    best = detached_xy.masked_fill(~valid_mode, float("inf")).amin(-1, keepdim=True)
    teacher = valid_mode & available[:, None] & (detached_xy <= best + .01)
    assignment_cost = (detached_xy + height_cost.detach()).masked_fill(~teacher, float("inf"))
    assignment = assignment_cost.argmin(-1)
    rows = torch.arange(batch_size, device=forecast.device)
    logits = output["mode_logits"].float()
    _finite_on(logits, valid_mode, "candidate logit")
    logits = torch.where(valid_mode, logits, 0.)
    all_log_mass = _masked_logsumexp(logits, valid_mode)
    teacher_log_mass = _masked_logsumexp(logits, teacher)
    probability = torch.exp(torch.where(valid_mode,
        logits - all_log_mass[:, None], torch.full_like(logits, float("-inf"))))

    reductions = {}
    parts = {}

    def reduce_term(name, values, weight, unit):
        safe_values = torch.where(weight > 0., values, torch.zeros_like(values))
        numerator = (safe_values * weight).sum()
        denominator = weight.sum().detach()
        parts[name] = _ratio(numerator, denominator)
        reductions[name] = {"numerator": numerator, "denominator": denominator,
                            "unit": unit}

    window_weight = available.float()
    reduce_term("xy", xy_cost[rows, assignment], window_weight,
                "accepted_window_with_valid_xy_time_mean")
    reduce_term("height", height_cost[rows, assignment], height_available.float(),
                "accepted_window_with_valid_height_time_mean")
    reduce_term("route_set_classification", all_log_mass - teacher_log_mass,
                window_weight, "accepted_window_with_valid_xy")
    reduce_term("expected_xy_risk", (xy_cost * probability).sum(-1), window_weight,
                "accepted_window_with_valid_xy_time_mean")

    # These targets supervise the graph only, independently of whether a beam
    # produced an accepted final candidate. No Cartesian fallback is emitted.
    node_valid = (batch["local_node_valid"].bool() &
                  output["graph_node_reachable"].bool() &
                  (batch["local_state"] >= 0))
    node_road = batch["local_state"].long() // 2
    truth_road = batch["target_road_id"][:, :steps].long()
    labeled = xy_valid & (truth_road >= 0)
    raw_confidence = batch["target_map_confidence"][:, :steps].float()
    confidence = torch.where(torch.isfinite(raw_confidence), raw_confidence.clamp(0., 1.), 0.)
    matching = node_valid[:, None] & (node_road[:, None] == truth_road[..., None])
    mass = output["time_message_arrival_mass"].float()
    _finite_on(mass, node_valid[:, None], "graph arrival mass")
    arrived = node_valid[:, None] & (mass > 0.)
    scored = matching & arrived
    graph_weight = (labeled & scored.any(-1)).float() * confidence
    graph_logits = output["graph_temporal_node_logits"].float()
    _finite_on(graph_logits, arrived, "arrived graph logit")
    graph_logits = torch.where(arrived, graph_logits, 0.)
    graph_nll = (_masked_logsumexp(graph_logits, arrived) -
                 _masked_logsumexp(graph_logits, scored))
    reduce_term("graph_road_nll", graph_nll, graph_weight,
                "confidence_weighted_arrived_labeled_graph_timepoint")

    earliest = output["graph_earliest_arrival_s"].float()
    _finite_on(earliest, node_valid, "reachable graph arrival time")
    first = earliest[:, None].expand_as(matching).masked_fill(~matching, UNREACHED_SECONDS).amin(-1)
    finite_path = matching.any(-1) & (first < UNREACHED_SECONDS)
    arrival_weight = (labeled & finite_path).float() * confidence
    interval = float(output["interval_seconds"])
    times = torch.arange(1, steps + 1, device=forecast.device, dtype=torch.float32) * interval
    # Remove absent paths before arithmetic, rather than multiplying a large
    # sentinel penalty (or nonfinite value) by zero afterwards.
    safe_first = torch.where(finite_path, first, times[None])
    late = F.relu((safe_first - times[None]) / interval)
    reduce_term("arrival_upper_bound", late, arrival_weight,
                "confidence_weighted_finite_path_labeled_graph_timepoint")

    total = sum(weights[name] * parts[name] for name in DEFAULT_WEIGHTS)
    diagnostics = {
        "window_count": forecast.new_tensor(batch_size),
        "accepted_window_count": accepted.sum(),
        "coordinate_labeled_window_count": (xy_points > 0).sum(),
        "coordinate_supervised_window_count": available.sum(),
        "coordinate_rejected_labeled_window_count": (~accepted & (xy_points > 0)).sum(),
        "coordinate_unlabeled_window_count": (xy_points == 0).sum(),
        "height_supervised_window_count": height_available.sum(),
        "ignored_unlabeled_nonfinite_xy_point_count": (~xy_valid & ~torch.isfinite(truth[..., :2]).all(-1)).sum(),
        "ignored_unlabeled_nonfinite_height_point_count": (~z_valid & ~torch.isfinite(truth[..., 2])).sum(),
        "graph_labeled_point_count": labeled.sum(),
        "graph_missing_path_labeled_point_count": (labeled & ~finite_path).sum(),
        "graph_unarrived_finite_path_labeled_point_count": (labeled & finite_path & ~scored.any(-1)).sum(),
        "graph_nonfinite_confidence_labeled_point_count": (labeled & ~torch.isfinite(raw_confidence)).sum(),
        "graph_nll_weight_in_rejected_windows": (graph_weight * (~accepted)[:, None]).sum(),
        "graph_arrival_weight_in_rejected_windows": (arrival_weight * (~accepted)[:, None]).sum(),
    }
    parts.update({
        "reductions": reductions, "diagnostics": diagnostics, "weights": weights,
        "training_teacher_set": teacher.detach(),
        "training_assigned_mode": torch.where(available, assignment, torch.full_like(assignment, -1)).detach(),
        "aggregation": "sum_each_term_numerator_and_denominator_before_dividing; not_mean_of_microbatch_means",
        "graph_supervision_independent_of_candidate_acceptance": True,
        "model_family": output["metadata"]["model_family"],
        "teacher_scope": "training_objective_only_not_forecast_or_evaluation",
        "declared_split": split,
        "data_split_provenance_verified": False,
    })
    return (total, parts) if return_parts else total
