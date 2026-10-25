"""Observation-only branching road graph proposer with 3D road-relative decoder.

The graph input is prebuilt from history and map only. Beam expansion, graph
message passing and curve decoding use tensor operations; neither future labels
nor host-side graph search enter inference. This is an experimental prototype,
not a certified road-legal or validated paper model.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .route_model_v12 import RoadNativeV12
from .route_model_v9 import ResidualMLP
from .local_graph_distance import UNREACHABLE_M
from .time_delayed_graph import TravelTimeDelayedGraph

class DirectedUnionLayer(nn.Module):

    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(width, width)
        self.in_projection = nn.Linear(width, width, bias=False)
        self.out_projection = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    @staticmethod
    def gather(hidden: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        (batch, nodes, degree) = indices.shape
        width = hidden.shape[-1]
        neighbor = hidden.gather(1, indices.reshape(batch, -1)[..., None].expand(-1, -1, width))
        neighbor = neighbor.reshape(batch, nodes, degree, width) * valid[..., None]
        return neighbor.sum(2) / valid.sum(2, keepdim=True).clamp_min(1)

    def forward(self, hidden: torch.Tensor, node_valid: torch.Tensor, predecessors: torch.Tensor, predecessor_valid: torch.Tensor, successors: torch.Tensor, successor_valid: torch.Tensor) -> torch.Tensor:
        incoming = self.gather(hidden, predecessors, predecessor_valid)
        outgoing = self.gather(hidden, successors, successor_valid)
        combined = self.self_projection(hidden) + self.in_projection(incoming) + self.out_projection(outgoing)
        return self.norm(hidden + F.gelu(combined)) * node_valid[..., None]

class LocalGraphRouteProposer(nn.Module):
    """Select K graph walks using history-conditioned directed message passing."""

    def __init__(self, graph, width: int=64, beams: int=8, maximum_roads: int=48, future_seconds: float=300.0, graph_layers: int=2, time_aware: bool=True, proposal_mode: str='beam', time_message_mode: str | None=None, time_message_hops: int=48, time_message_width: int=32) -> None:
        super().__init__()
        if width < 8 or beams < 1 or maximum_roads < 2 or (graph_layers < 1):
            raise ValueError('Invalid graph proposer size')
        (self.beams, self.maximum_roads) = (int(beams), int(maximum_roads))
        self.future_seconds = float(future_seconds)
        self.time_aware = bool(time_aware)
        if proposal_mode not in ('beam', 'endpoint'):
            raise ValueError('Unknown graph route proposal mode')
        self.proposal_mode = proposal_mode
        ptr = np.asarray(graph.road_ptr, dtype=np.int64)
        xyz = np.asarray(graph.points_xyz, dtype=np.float32)
        (start, end) = (xyz[ptr[:-1]], xyz[ptr[1:] - 1])
        reference = np.asarray([72000.0, 3279000.0, 0.0], dtype=np.float32)
        self.register_buffer('state_entry', torch.from_numpy(np.stack((end, start), axis=1).reshape(-1, 3) - reference), persistent=False)
        self.register_buffer('state_exit', torch.from_numpy(np.stack((start, end), axis=1).reshape(-1, 3) - reference), persistent=False)
        self.register_buffer('state_length', torch.from_numpy(np.repeat(np.asarray(graph.road_length_m, dtype=np.float32), 2)), persistent=False)
        self.history_projection = nn.Sequential(nn.Linear(13, width), nn.GELU(), ResidualMLP(width))
        self.node_projection = nn.Sequential(nn.Linear(11, width), nn.GELU(), nn.LayerNorm(width))
        self.layers = nn.ModuleList((DirectedUnionLayer(width) for _ in range(graph_layers)))
        self.node_score = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
        self.horizon_embedding = nn.Parameter(torch.randn(round(self.future_seconds / 15.0), width) * 0.02)
        self.temporal_query = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width))
        self.traversal_speed_adjustment = nn.Linear(width, 1)
        nn.init.zeros_(self.node_score[-1].weight)
        nn.init.zeros_(self.node_score[-1].bias)
        nn.init.zeros_(self.traversal_speed_adjustment.weight)
        nn.init.zeros_(self.traversal_speed_adjustment.bias)
        self.delayed_messages = None
        self.suppress_arrival_feature = False
        if time_message_mode is not None:
            self.enable_delayed_messages(time_message_mode, time_message_hops, time_message_width)

    def enable_delayed_messages(self, mode: str='travel-time', hops: int=48, message_width: int=32) -> None:
        """Add the V17 delay operator after loading a legacy checkpoint.

        Legacy V13/V16 runs remain unchanged unless this is explicitly enabled.
        Both policies have identical learned parameters. ``fixed-hop`` replaces
        physical edge delays with one cadence per hop, not a different network.
        """
        if self.delayed_messages is not None:
            raise ValueError('Time-delayed graph was already configured')
        width = self.traversal_speed_adjustment.in_features
        self.delayed_messages = TravelTimeDelayedGraph(width, message_width, len(self.horizon_embedding), 15.0, hops, mode)
        self.node_traversal_speed_adjustment = nn.Linear(width * 2, 1)
        nn.init.zeros_(self.node_traversal_speed_adjustment.weight)
        nn.init.zeros_(self.node_traversal_speed_adjustment.bias)

    def forward(self, history_xyz: torch.Tensor, history_xy_valid: torch.Tensor, history_z_valid: torch.Tensor, time_features: torch.Tensor, origin_local: torch.Tensor, route_start_s: torch.Tensor, state: torch.Tensor, node_valid: torch.Tensor, successors: torch.Tensor, successor_valid: torch.Tensor, predecessors: torch.Tensor, predecessor_valid: torch.Tensor, start_valid: torch.Tensor, start_distance_m: torch.Tensor, start_node_index: torch.Tensor, start_parent: torch.Tensor, motion_prior_speed: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        (batch, nodes) = state.shape
        if nodes < self.beams:
            raise ValueError('Graph padding capacity must exceed beam count')
        state = state.long()
        successors = successors.long()
        predecessors = predecessors.long()
        valid_out = successor_valid.bool() & node_valid[..., None].bool()
        valid_in = predecessor_valid.bool() & node_valid[..., None].bool()
        xy_valid = history_xy_valid.bool()
        z_valid = history_z_valid.bool() & xy_valid
        recent = history_xyz[:, -5:].float()
        recent_xy = torch.where(xy_valid[:, -5:, None], recent[..., :2], 0.0)
        recent_z = torch.where(z_valid[:, -5:], recent[..., 2], 0.0)
        pair = xy_valid[:, -4:] & xy_valid[:, -5:-1]
        z_pair = z_valid[:, -4:] & z_valid[:, -5:-1]
        delta_xy = recent_xy[:, 1:] - recent_xy[:, :-1]
        delta_z = recent_z[:, 1:] - recent_z[:, :-1]
        velocity_xy = (delta_xy * pair[..., None]).sum(1) / (15.0 * pair.sum(1).clamp_min(1))[..., None]
        velocity_z = (delta_z * z_pair).sum(1) / (15.0 * z_pair.sum(1).clamp_min(1))
        velocity = torch.cat((velocity_xy, velocity_z[:, None]), -1)
        velocity = velocity.clamp(-60, 60)
        speed = torch.linalg.vector_norm(velocity[..., :2], dim=-1).clamp(0, 60)
        current = torch.cat((torch.where(xy_valid[:, -1, None], history_xyz[:, -1, :2].float(), 0.0), torch.where(z_valid[:, -1, None], history_xyz[:, -1, 2:].float(), 0.0)), -1)
        history_feature = torch.cat((current / current.new_tensor([1000.0, 1000.0, 20.0]), velocity / 10.0, time_features.float(), xy_valid[:, -1, None].float(), z_valid[:, -1, None].float(), (speed / 60.0)[:, None]), dim=-1)
        context = self.history_projection(history_feature)
        entry = self.state_entry[state] - origin_local[:, None]
        exit = self.state_exit[state] - origin_local[:, None]
        segment = exit - entry
        length = self.state_length[state]
        start_arc = route_start_s[:, 0].float()
        start_code = state.gather(1, start_node_index.long())
        start_length = length.gather(1, start_node_index.long())
        oriented_start = torch.where(start_code % 2 > 0, start_arc[:, None], start_length - start_arc[:, None])
        distance_from_observation = torch.where(start_distance_m < UNREACHABLE_M, (start_distance_m - oriented_start[..., None]).clamp_min(0.0), torch.full_like(start_distance_m, UNREACHABLE_M)).amin(1)
        motion_speed = speed if motion_prior_speed is None else motion_prior_speed.float().clamp(0.0, 60.0)
        traversal_speed = F.softplus(motion_speed + 10.0 * self.traversal_speed_adjustment(context).squeeze(-1)).clamp(2.0 if motion_prior_speed is None else 0.5, 60.0)
        earliest_time = distance_from_observation / traversal_speed[:, None]
        arrival_feature = earliest_time.clamp_max(3000.0) / 300.0
        if self.suppress_arrival_feature or (self.delayed_messages is not None and self.delayed_messages.mode == 'fixed-hop'):
            arrival_feature = torch.zeros_like(arrival_feature)
        node_feature = torch.cat((entry[..., :2] / 1000.0, entry[..., 2:3] / 20.0, segment / 1000.0, torch.log1p(length)[..., None] / 10.0, (state % 2 * 2 - 1).float()[..., None], start_valid.float()[..., None], distance_from_observation.clamp_max(10000.0)[..., None] / 1000.0, arrival_feature[..., None]), dim=-1)
        hidden = self.node_projection(node_feature) * node_valid[..., None]
        for layer in self.layers:
            hidden = layer(hidden, node_valid, predecessors, valid_in, successors, valid_out)
        context_nodes = context[:, None].expand(-1, nodes, -1)
        static_score = self.node_score(torch.cat((hidden, context_nodes), dim=-1)).squeeze(-1)
        horizon = torch.arange(1, self.horizon_embedding.shape[0] + 1, device=hidden.device, dtype=hidden.dtype) * 15.0
        queries = self.temporal_query(torch.cat((context[:, None].expand(-1, len(horizon), -1), self.horizon_embedding[None].expand(batch, -1, -1)), -1))
        delay_diagnostics = {}
        if self.delayed_messages is None:
            temporal_logits = torch.einsum('btw,bnw->btn', queries, hidden) / math.sqrt(hidden.shape[-1])
            temporal_logits = temporal_logits + static_score[:, None]
            if self.time_aware:
                temporal_logits = temporal_logits - F.relu(earliest_time[:, None] - horizon[None, :, None]) / 15.0
        else:
            node_speed = F.softplus(traversal_speed[:, None] + 10.0 * self.node_traversal_speed_adjustment(torch.cat((hidden, context_nodes), -1)).squeeze(-1)).clamp(0.5, 60.0)
            node_oriented_start = torch.where(state % 2 > 0, start_arc[:, None], length - start_arc[:, None])
            remaining_length = torch.where(start_valid.bool(), (length - node_oriented_start).clamp_min(0), length)
            delayed = self.delayed_messages(hidden, context, remaining_length / node_speed, node_valid, predecessors, valid_in, valid_out.sum(-1), start_valid)
            temporal_logits = torch.einsum('btw,btnw->btn', queries, delayed['temporal_hidden']) / math.sqrt(hidden.shape[-1])
            temporal_logits = temporal_logits + static_score[:, None]
            temporal_logits = temporal_logits.masked_fill(delayed['arrival_mass'] <= 0.0, -10000.0)
            earliest_time = delayed['earliest_arrival_s']
            context_weight = temporal_logits.softmax(-1) * (delayed['arrival_mass'] > 0)
            context_weight = context_weight / context_weight.sum(-1, keepdim=True).clamp_min(1e-06)
            delay_diagnostics = {'time_message_arrival_mass': delayed['arrival_mass'], 'time_message_traversal_s': delayed['effective_traversal_seconds'], 'time_message_speed_mps': node_speed, 'time_message_context': torch.einsum('btn,btnw->btw', context_weight, delayed['temporal_hidden'])}
        temporal_logits = temporal_logits.masked_fill((~node_valid | (distance_from_observation >= UNREACHABLE_M))[:, None], -10000.0)
        score = torch.logsumexp(temporal_logits, dim=1) - math.log(len(horizon))
        score = score.masked_fill(~node_valid, -1000000000.0)
        if self.proposal_mode == 'endpoint':
            endpoint_valid = node_valid
            if self.delayed_messages is not None:
                endpoint_valid = node_valid & (delayed['arrival_mass'][:, -1] > 0)
            paths = self.endpoint_paths(state, endpoint_valid, temporal_logits[:, -1], start_distance_m, start_node_index, start_parent, start_arc, length, motion_speed)
            return {**paths, 'node_score': score, 'temporal_node_logits': temporal_logits, 'earliest_graph_arrival_s': earliest_time, 'graph_node_reachable': distance_from_observation < UNREACHABLE_M, 'graph_hidden': hidden, 'estimated_start_speed_mps': speed, 'graph_motion_prior_speed_mps': motion_speed, **delay_diagnostics}
        initial = score.masked_fill(~start_valid.bool(), -1000000000.0)
        (beam_score, current_node) = initial.topk(self.beams, dim=-1)
        beam_valid = start_valid.gather(1, current_node).bool()
        beam_active = beam_valid
        beam_nodes = current_node[..., None]
        count = beam_valid.long()
        initial_state = state.gather(1, current_node)
        oriented_start = torch.where(initial_state % 2 > 0, start_arc[:, None], length.gather(1, current_node) - start_arc[:, None])
        accumulated = (length.gather(1, current_node) - oriented_start).clamp_min(0)
        required_length = (motion_speed * self.future_seconds * 1.25 + 250.0).clamp(250.0, 6000.0)
        degree = successors.shape[-1]
        for _ in range(1, self.maximum_roads):
            outgoing = successors.gather(1, current_node[..., None].expand(-1, -1, degree))
            outgoing_valid = valid_out.gather(1, current_node[..., None].expand(-1, -1, degree))
            can_extend = outgoing_valid.any(-1) & beam_active
            stop_valid = beam_valid & ((accumulated >= required_length[:, None]) | ~can_extend)
            candidate_node = torch.cat((current_node[..., None], outgoing), dim=-1)
            candidate_valid = torch.cat((stop_valid[..., None], outgoing_valid & beam_active[..., None]), -1)
            candidate_score = beam_score[..., None] + torch.cat((torch.zeros_like(current_node[..., None], dtype=score.dtype), score.gather(1, outgoing.reshape(batch, -1)).reshape(batch, self.beams, degree)), -1)
            candidate_score = candidate_score.masked_fill(~candidate_valid, -1000000000.0)
            (best, flat_index) = candidate_score.flatten(1).topk(self.beams, dim=-1)
            parent = flat_index // (degree + 1)
            option = flat_index % (degree + 1)
            chosen_node = candidate_node.flatten(1).gather(1, flat_index)
            chosen_valid = candidate_valid.flatten(1).gather(1, flat_index)
            old_count = count.gather(1, parent)
            extended = (option > 0) & chosen_valid
            count = old_count + extended.long()
            beam_nodes = beam_nodes.gather(1, parent[..., None].expand(-1, -1, beam_nodes.shape[-1]))
            beam_nodes = torch.cat((beam_nodes, chosen_node[..., None]), dim=-1)
            accumulated = accumulated.gather(1, parent) + length.gather(1, chosen_node) * extended
            beam_active = extended
            (current_node, beam_score, beam_valid) = (chosen_node, best, chosen_valid)
        path_state = state.gather(1, beam_nodes.reshape(batch, -1)).reshape(batch, self.beams, self.maximum_roads)
        route_roads = path_state // 2
        route_directions = (path_state % 2 * 2 - 1).to(torch.int8)
        return {'route_roads': route_roads, 'route_directions': route_directions, 'route_count': count, 'route_valid': beam_valid, 'route_start_s': start_arc[:, None].expand(-1, self.beams), 'beam_score': beam_score, 'beam_local_node': beam_nodes, 'node_score': score, 'temporal_node_logits': temporal_logits, 'earliest_graph_arrival_s': earliest_time, 'graph_node_reachable': distance_from_observation < UNREACHABLE_M, 'graph_hidden': hidden, 'estimated_start_speed_mps': speed, 'graph_motion_prior_speed_mps': motion_speed, **delay_diagnostics}

    def endpoint_paths(self, state: torch.Tensor, node_valid: torch.Tensor, endpoint_score: torch.Tensor, start_distance_m: torch.Tensor, start_node_index: torch.Tensor, start_parent: torch.Tensor, start_arc: torch.Tensor, length: torch.Tensor, speed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Choose distinct predicted road endpoints and backtrack baked trees.

        Topology search and shortest-path construction happened in the offline
        cache. The accelerator only gathers tensor parents; no future label or
        host-side branch selection enters inference.
        """
        (batch, nodes) = state.shape
        score = endpoint_score.masked_fill(~node_valid, -1000000000.0)
        score = score.masked_fill((start_distance_m >= UNREACHABLE_M).all(1), -1000000000.0)
        start_code = state.gather(1, start_node_index.long())
        start_length = length.gather(1, start_node_index.long())
        oriented_start = torch.where(start_code % 2 > 0, start_arc[:, None], start_length - start_arc[:, None])
        from_observation = torch.where(start_distance_m < UNREACHABLE_M, (start_distance_m - oriented_start[..., None]).clamp_min(0.0), torch.full_like(start_distance_m, UNREACHABLE_M))
        node_distance = from_observation.amin(1) + length * 0.5
        expected = (speed * self.future_seconds).clamp(100.0, 6000.0)
        targets = torch.linspace(0.0, 1.75, self.beams, device=state.device)
        (chosen, chosen_score) = ([], [])
        remaining = score
        for ratio in targets:
            target = expected * ratio
            distance_penalty = (node_distance - target[:, None]).abs() / (expected[:, None] * 0.25 + 150.0) * 2.0
            (value, index) = (remaining - distance_penalty).max(-1)
            chosen.append(index)
            chosen_score.append(value)
            road = state.gather(1, index[:, None]) // 2
            remaining = remaining.masked_fill(state // 2 == road, -1000000000.0)
        endpoint = torch.stack(chosen, -1)
        endpoint_value = torch.stack(chosen_score, -1)
        valid_endpoint = endpoint_value > -100000000.0
        endpoint_distance = from_observation.gather(2, endpoint[:, None].expand(-1, 2, -1))
        start_slot = endpoint_distance.argmin(1)
        selected_root = start_node_index.long().gather(1, start_slot)
        parents = start_parent.long().gather(1, start_slot[..., None].expand(-1, -1, nodes))
        active = valid_endpoint
        current = endpoint
        reversed_path = []
        count = torch.zeros_like(endpoint)
        for _ in range(self.maximum_roads):
            reversed_path.append(current)
            count = count + active.long()
            previous = parents.gather(2, current[..., None]).squeeze(-1)
            active = active & (previous >= 0)
            current = previous.clamp_min(0)
        reversed_path = torch.stack(reversed_path, -1)
        positions = (count[..., None] - 1 - torch.arange(self.maximum_roads, device=state.device)).clamp(0, self.maximum_roads - 1)
        path_node = reversed_path.gather(-1, positions)
        valid_path = valid_endpoint & ~active & (count > 0) & (path_node[..., 0] == selected_root)
        path_state = state.gather(1, path_node.reshape(batch, -1)).reshape(batch, self.beams, self.maximum_roads)
        return {'route_roads': path_state // 2, 'route_directions': (path_state % 2 * 2 - 1).to(torch.int8), 'route_count': count, 'route_valid': valid_path, 'route_start_s': start_arc[:, None].expand(-1, self.beams), 'beam_score': torch.zeros_like(endpoint_value), 'beam_local_node': path_node}

class RoadNativeV13(RoadNativeV12):

    def __init__(self, graph, *args, beams: int=8, maximum_roads: int=48, include_fixed_routes: bool=True, proposal_mode: str='beam', initial_heading_guard: bool=False, time_message_mode: str | None=None, time_message_hops: int=48, time_message_width: int=32, **kwargs) -> None:
        super().__init__(graph, *args, **kwargs)
        self.api_version = 13
        self.proposer = LocalGraphRouteProposer(graph, width=self.speed_head.in_features, beams=beams, maximum_roads=maximum_roads, future_seconds=self.dt * self.future_steps, time_aware=self.road_graph.time_aware, proposal_mode=proposal_mode, time_message_mode=time_message_mode, time_message_hops=time_message_hops, time_message_width=time_message_width)
        self.include_fixed_routes = bool(include_fixed_routes)
        self.initial_heading_guard = bool(initial_heading_guard)
        self.beam_rank_weight = nn.Parameter(torch.tensor(1.0))

    def _guard_start_direction(self, roads: torch.Tensor, directions: torch.Tensor, start_s: torch.Tensor, valid: torch.Tensor, history_xyz: torch.Tensor, history_xy_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Disallow an immediate reverse start when recent movement is clear.

        This only uses observed XY and the mapped route geometry. If no valid
        forward candidate exists, preserve the original set and flag that the
        guard was unresolved rather than turning a prediction into rejection.
        """
        recent = history_xyz[:, -1, :2].float() - history_xyz[:, -4, :2].float()
        recent_length = torch.linalg.vector_norm(recent, dim=-1)
        reliable_history = history_xy_valid[:, -4:].bool().all(-1) & (recent_length >= 40.0)
        first_road = roads[..., 0].long()
        first_direction = directions[..., 0].float()
        first = self.geometry.sample_road(first_road, start_s.float())[..., :2]
        next_point = self.geometry.sample_road(first_road, start_s.float() + first_direction * 5.0)[..., :2]
        travel = next_point - first
        usable_tangent = torch.linalg.vector_norm(travel, dim=-1) >= 1.0
        opposed = (travel * recent[:, None]).sum(-1) < 0.0
        allowed = valid & ~(reliable_history[:, None] & usable_tangent & opposed)
        applied = reliable_history & allowed.any(-1)
        unresolved = reliable_history & valid.any(-1) & ~allowed.any(-1)
        filtered = torch.where(applied[:, None], allowed, valid)
        rejected = (valid & ~filtered).to(torch.int32).sum(-1)
        return (filtered, applied, rejected, unresolved)

    def forward(self, history_xyz, history_xy_valid, history_z_valid, origin_context, time_features, matched_xyz, road_tangent, road_d, map_gap, map_confidence, route_roads, route_directions, route_count, route_valid, route_start_s, origin_local, local_state, local_node_valid, local_successors, local_strict_successor_valid, local_predecessors, local_strict_predecessor_valid, local_start_valid, local_start_distance_m, local_start_node_index, local_start_parent, motion_prior_speed=None):
        proposed = self.proposer(history_xyz, history_xy_valid, history_z_valid, time_features, origin_local, route_start_s, local_state, local_node_valid, local_successors, local_strict_successor_valid, local_predecessors, local_strict_predecessor_valid, local_start_valid, local_start_distance_m, local_start_node_index, local_start_parent, motion_prior_speed=motion_prior_speed)
        if self.include_fixed_routes:
            pad = proposed['route_roads'].shape[-1] - route_roads.shape[-1]
            if pad < 0:
                raise ValueError('Dynamic route capacity must cover fixed route length')
            candidate_roads = torch.cat((F.pad(route_roads, (0, pad)), proposed['route_roads']), dim=1)
            candidate_directions = torch.cat((F.pad(route_directions, (0, pad)), proposed['route_directions'].long()), dim=1)
            candidate_count = torch.cat((route_count, proposed['route_count']), dim=1)
            candidate_valid = torch.cat((route_valid, proposed['route_valid']), dim=1)
            candidate_start_s = torch.cat((route_start_s, proposed['route_start_s']), dim=1)
            rank_prior = torch.cat((torch.zeros_like(route_start_s), proposed['beam_score']), dim=1)
            route_source = torch.cat((torch.zeros_like(route_valid, dtype=torch.int8), torch.ones_like(proposed['route_valid'], dtype=torch.int8)), dim=1)
            beam_node = torch.cat((torch.full((*route_roads.shape[:2], proposed['beam_local_node'].shape[-1]), -1, dtype=torch.long, device=route_roads.device), proposed['beam_local_node']), dim=1)
        else:
            candidate_roads = proposed['route_roads']
            candidate_directions = proposed['route_directions'].long()
            candidate_count = proposed['route_count']
            candidate_valid = proposed['route_valid']
            candidate_start_s = proposed['route_start_s']
            rank_prior = proposed['beam_score']
            route_source = torch.ones_like(candidate_valid, dtype=torch.int8)
            beam_node = proposed['beam_local_node']
        if self.initial_heading_guard:
            (candidate_valid, heading_guard_applied, heading_guard_rejected, heading_guard_unresolved) = self._guard_start_direction(candidate_roads, candidate_directions, candidate_start_s, candidate_valid, history_xyz, history_xy_valid)
        else:
            heading_guard_applied = torch.zeros(candidate_valid.shape[0], dtype=torch.bool, device=candidate_valid.device)
            heading_guard_rejected = torch.zeros(candidate_valid.shape[0], dtype=torch.int32, device=candidate_valid.device)
            heading_guard_unresolved = heading_guard_applied.clone()
        output = super().forward(history_xyz, history_xy_valid, history_z_valid, origin_context, time_features, matched_xyz, road_tangent, road_d, map_gap, map_confidence, candidate_roads, candidate_directions, candidate_count, candidate_valid, candidate_start_s, origin_local)
        output['mode_logits'] = output['mode_logits'] + self.beam_rank_weight * rank_prior
        generated_roads = candidate_roads
        full_length = self.geometry.length[generated_roads.long()]
        piece_start = torch.where(candidate_directions > 0, torch.zeros_like(full_length), full_length)
        piece_start = torch.cat((candidate_start_s[..., None], piece_start[..., 1:]), -1)
        piece_end = torch.where(candidate_directions > 0, full_length, torch.zeros_like(full_length))
        piece_length = (piece_end - piece_start).abs()
        output['generated_route_roads'] = candidate_roads
        output['generated_route_directions'] = candidate_directions
        output['generated_route_count'] = candidate_count
        output['generated_route_start_s'] = candidate_start_s
        output['generated_beam_local_node'] = beam_node
        output['candidate_source'] = route_source
        output['generated_piece_start_s'] = piece_start
        output['generated_piece_length_m'] = piece_length
        output['graph_node_score'] = proposed['node_score']
        output['graph_temporal_node_logits'] = proposed['temporal_node_logits']
        output['graph_earliest_arrival_s'] = proposed['earliest_graph_arrival_s']
        output['graph_node_reachable'] = proposed['graph_node_reachable']
        output['estimated_start_speed_mps'] = proposed['estimated_start_speed_mps']
        output['graph_motion_prior_speed_mps'] = proposed['graph_motion_prior_speed_mps']
        for key in ('time_message_arrival_mass', 'time_message_traversal_s', 'time_message_speed_mps', 'time_message_context'):
            if key in proposed:
                output[key] = proposed[key]
        output['initial_heading_guard_applied'] = heading_guard_applied
        output['initial_heading_guard_rejected_candidates'] = heading_guard_rejected
        output['initial_heading_guard_unresolved'] = heading_guard_unresolved
        return output
