"""Original Cartesian and road-parent training objectives."""
from __future__ import annotations
import torch
from torch.nn import functional as F

def dense_loss(output, batch, output_xy_scale, z_weight=.25):
    prediction = output["trajectories"]
    target = batch["target_xyz"][:, :prediction.shape[2]]
    xy_valid = batch["target_xy_valid"][:, :prediction.shape[2]].float()
    z_valid = batch["target_z_valid"][:, :prediction.shape[2]].float()
    # Clear masked values BEFORE arithmetic, not NaN * zero afterwards.
    xy_target = torch.where(xy_valid[..., None].bool(), target[..., :2], torch.zeros_like(target[..., :2]))
    z_target = torch.where(z_valid.bool(), target[..., 2], torch.zeros_like(target[..., 2]))
    xy = F.smooth_l1_loss(prediction[..., :2] / output_xy_scale[None, None, :, None],
                          (xy_target / output_xy_scale[None, :, None])[:, None].expand_as(prediction[..., :2]), reduction="none").mean(-1)
    xy = (xy * xy_valid[:, None]).sum(-1) / xy_valid.sum(-1).clamp_min(1)[:, None]
    z = F.smooth_l1_loss(prediction[..., 2] / 20., (z_target / 20.)[:, None].expand_as(prediction[..., 2]), reduction="none")
    z = (z * z_valid[:, None]).sum(-1) / z_valid.sum(-1).clamp_min(1)[:, None]
    per_mode = xy + z_weight * z
    winner = per_mode.detach().argmin(-1)
    fit = per_mode.gather(1, winner[:, None]).mean()
    confidence = F.cross_entropy(output["mode_logits"], winner)
    return fit + .1 * confidence

def v13_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
             z_weight: float = 1.0, return_parts: bool = False,
             rank_teacher: str = "coverage"):
    """Future labels supervise, but never enter graph construction/forward."""
    forecast = output["trajectories"]
    batch_size, modes, steps, _ = forecast.shape
    valid_xy = batch["target_xy_valid"][:, :steps].bool()
    valid_z = batch["target_z_valid"][:, :steps].bool() & valid_xy
    target = batch["target_xyz"][:, :steps].float()
    safe_xy = torch.where(valid_xy[..., None], target[..., :2], 0.)
    safe_z = torch.where(valid_z, target[..., 2], 0.)
    xy_point = torch.linalg.vector_norm(forecast[..., :2] - safe_xy[:, None], dim=-1)
    xy_cost = (xy_point * valid_xy[:, None]).sum(-1) / valid_xy.sum(-1, keepdim=True).clamp_min(1) / 100.
    z_point = F.smooth_l1_loss(forecast[..., 2] / 10.,
                               (safe_z / 10.)[:, None].expand(batch_size, modes, steps), reduction="none")
    z_cost = (z_point * valid_z[:, None]).sum(-1) / valid_z.sum(-1, keepdim=True).clamp_min(1)
    auxiliary_z_point = F.smooth_l1_loss(output["auxiliary_height_m"] / 10.,
                                         (safe_z / 10.)[:, None].expand(batch_size, modes, steps), reduction="none")
    auxiliary_z_cost = (auxiliary_z_point * valid_z[:, None]).sum(-1) / valid_z.sum(-1, keepdim=True).clamp_min(1)
    feasible = output["mode_valid"].bool()
    road = output["generated_route_roads"]
    count = output["generated_route_count"]
    piece_valid = torch.arange(road.shape[-1], device=road.device) < count[..., None]
    future_road = batch["target_road_id"][:, :steps].long()
    future_s = batch["target_road_s"][:, :steps].float()
    labeled = valid_xy & (future_road >= 0)
    road_hit = ((road[:, :, None] == future_road[:, None, :, None]) &
                piece_valid[:, :, None]).any(-1)
    coverage = (road_hit * labeled[:, None]).sum(-1) / labeled.sum(-1, keepdim=True).clamp_min(1)
    coverage = coverage.masked_fill(~feasible, -1.)
    available = feasible.any(-1) & valid_xy.any(-1)
    if rank_teacher == "coverage":
        teacher_score = -coverage
        tolerance = 1e-6
    elif rank_teacher in ("hybrid", "xy-error"):
        # The detached localization error is an assignment target, not a
        # forward input. Coverage is a soft 50 m equivalent in hybrid mode;
        # it must not overrule hundreds of metres of actual XY error.
        teacher_score = xy_cost.detach()
        if rank_teacher == "hybrid":
            teacher_score = teacher_score + .5 * (1. - coverage.clamp_min(0.))
        tolerance = .01
    else:
        raise ValueError(f"Unknown route rank teacher: {rank_teacher}")
    best_teacher_score = teacher_score.masked_fill(~feasible, float("inf")).amin(-1, keepdim=True)
    teacher_set = feasible & (teacher_score <= best_teacher_score + tolerance)
    assigned = (xy_cost + z_weight * z_cost).detach().masked_fill(
        ~teacher_set, float("inf")).argmin(-1)
    rows = torch.arange(batch_size, device=forecast.device)
    sample_weight = available.float()
    denominator = sample_weight.sum().clamp_min(1)
    xy = (xy_cost[rows, assigned] * sample_weight).sum() / denominator
    z = (z_cost[rows, assigned] * sample_weight).sum() / denominator
    auxiliary_z = (auxiliary_z_cost[rows, assigned] * sample_weight).sum() / denominator
    logits = output["mode_logits"].float()
    all_mass = torch.logsumexp(logits.masked_fill(~feasible, -1e4), dim=-1)
    teacher_mass = torch.logsumexp(logits.masked_fill(~teacher_set, -1e4), dim=-1)
    ranking = ((all_mass - teacher_mass) * sample_weight).sum() / denominator
    # If the matched future road appears exactly once on a candidate, its
    # observed arc coordinate gives a fixed, model-independent route progress
    # target. Exclude ambiguous loops and targets requiring >60 m/s.
    unique_piece = ((road[:, :, None] == future_road[:, None, :, None]) &
                    piece_valid[:, :, None])
    unique_match = unique_piece.sum(-1) == 1
    piece_start = output["generated_piece_start_s"]
    piece_length = output["generated_piece_length_m"] * piece_valid
    piece_prefix = piece_length.cumsum(-1) - piece_length
    directed_offset = (future_s[:, None, :, None] - piece_start[:, :, None]).abs()
    teacher_progress = ((piece_prefix[:, :, None] + directed_offset) * unique_piece).sum(-1)
    target_in_piece = ((directed_offset <= piece_length[:, :, None] + 1e-3) &
                       unique_piece).any(-1)
    current_limit = (torch.arange(1, steps + 1, device=forecast.device, dtype=teacher_progress.dtype)
                     * 15.0 * 60.0 + 25.0)
    reliable_progress = (unique_match & target_in_piece & labeled[:, None] &
                         (teacher_progress <= current_limit[None, None]))
    chosen_progress = teacher_progress[rows, assigned]
    chosen_reliable = reliable_progress[rows, assigned]
    chosen_confidence = batch["target_map_confidence"][:, :steps].float().clamp(0., 1.)
    progress_weight = chosen_reliable.float() * chosen_confidence * sample_weight[:, None]
    progress_target = torch.where(chosen_reliable, chosen_progress, 0.0)
    arc_point = F.smooth_l1_loss(output["progress_m"][rows, assigned] / 100.,
                                 progress_target / 100., reduction="none")
    arc = (arc_point * progress_weight).sum() / progress_weight.sum().clamp_min(1)
    pair_weight = progress_weight[:, 1:] * progress_weight[:, :-1]
    target_speed = (progress_target[:, 1:] - progress_target[:, :-1]) / 15.
    plausible_speed = (target_speed >= 0) & (target_speed <= 60.)
    pair_weight = pair_weight * plausible_speed
    speed_point = F.smooth_l1_loss(output["requested_speed_mps"][rows, assigned, 1:] / 10.,
                                   target_speed.clamp(0., 60.) / 10., reduction="none")
    speed_fit = (speed_point * pair_weight).sum() / pair_weight.sum().clamp_min(1)
    guide_point = F.smooth_l1_loss(output["guide_xy"] / 100., safe_xy / 100., reduction="none").sum(-1)
    guide = (guide_point * valid_xy).sum() / valid_xy.sum().clamp_min(1)
    # Dense future-road supervision teaches the branch policy even when the
    # current discrete beam misses every useful complete route. It is a loss
    # target only; target_road_id is excluded from v13_forward.
    node_road = batch["local_state"].long() // 2
    graph_node_valid = batch["local_node_valid"].bool()
    matching_node = ((node_road[:, None] == future_road[:, :, None]) &
                     graph_node_valid[:, None] &
                     output["graph_node_reachable"][:, None])
    label_present = matching_node.any(-1) & labeled
    confidence = batch["target_map_confidence"][:, :steps].float().clamp(0., 1.)
    graph_weight = label_present.float() * confidence
    node_score = output["graph_temporal_node_logits"].float()
    all_node_log_mass = torch.logsumexp(node_score.masked_fill(~graph_node_valid[:, None], -1e4), -1)
    matching_log_mass = torch.logsumexp(
        node_score.masked_fill(~matching_node, -1e4), -1)
    graph_road = (((all_node_log_mass - matching_log_mass) * graph_weight).sum() /
                  graph_weight.sum().clamp_min(1))
    lateral_prior = output["road_relative_lateral_m"].square().mean() / 25.
    height_prior = output["road_relative_height_residual_m"].square().mean() / 100.
    speed = output["requested_speed_mps"]
    smoothness = torch.diff(speed, dim=-1).abs().mean() / 10.
    loss = (xy + z_weight * (z + .1 * auxiliary_z) + ranking + guide + graph_road +
            .5 * arc + .2 * speed_fit +
            .02 * (lateral_prior + height_prior) + .01 * smoothness)
    parts = {"xy": xy, "height": z, "auxiliary_height": auxiliary_z,
             "route_set_classification": ranking,
             "cartesian_guide": guide, "graph_road_nll": graph_road,
             "arc_progress": arc, "observed_speed": speed_fit,
             "lateral_prior": lateral_prior,
             "height_prior": height_prior, "speed_smoothness": smoothness,
             "candidate_coverage": (coverage.clamp_min(0).max(-1).values * sample_weight).sum() / denominator}
    return (loss, parts) if return_parts else loss
