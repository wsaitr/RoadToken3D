"""Finite-horizon, travel-time-delayed messages on a directed road graph.

Unlike a static GNN followed by an arrival-time attention mask, each message
reads its predecessor's state at ``t - traversal_time(predecessor)``. Along a
chain the delays therefore accumulate. Start nodes represent the observed
position; their first traversal uses only the remaining road length.

Signals are sampled at a fixed cadence and linearly interpolated. A separate
per-walk-length earliest-arrival guard prevents interpolation from creating
activity before any path of that length can arrive. This is a discretized,
bounded-walk model, not exact continuous-time traffic simulation. Cycles and
long paths are bounded by ``maximum_hops``; maps and seeds must use history
only. All computation uses tensor gathers; no online CPU graph search occurs.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
UNREACHED_SECONDS = 10000000.0

class TravelTimeDelayedGraph(nn.Module):

    def __init__(self, input_width: int, message_width: int=32, future_steps: int=20, interval_seconds: float=15.0, maximum_hops: int=48, mode: str='travel-time', checkpoint_chunk: int=8) -> None:
        super().__init__()
        if min(input_width, message_width, future_steps, maximum_hops) < 1:
            raise ValueError('Positive graph dimensions and walk budget required')
        if interval_seconds <= 0 or mode not in ('travel-time', 'fixed-hop'):
            raise ValueError('Positive cadence and known time-delay policy required')
        self.future_steps = int(future_steps)
        self.interval_seconds = float(interval_seconds)
        self.maximum_hops = int(maximum_hops)
        self.mode = mode
        if checkpoint_chunk < 0:
            raise ValueError('Checkpoint chunk cannot be negative')
        self.checkpoint_chunk = int(checkpoint_chunk)
        self.node_embedding = nn.Linear(input_width, message_width)
        self.history_embedding = nn.Linear(input_width, message_width)
        self.message_update = nn.Linear(message_width, message_width, bias=False)
        self.road_update = nn.Linear(message_width, message_width, bias=False)
        self.output_projection = nn.Linear(message_width, input_width, bias=False)

    def _delay_plan(self, travel_seconds: torch.Tensor, predecessors: torch.Tensor, edge_valid: torch.Tensor):
        (batch, nodes, degree) = predecessors.shape
        samples = self.future_steps + 1
        delay = travel_seconds.gather(1, predecessors.reshape(batch, -1)).reshape(batch, nodes, degree)
        ticks = torch.arange(samples, device=travel_seconds.device, dtype=torch.float32) * self.interval_seconds
        query = (ticks[None, None, :, None] - delay[:, :, None]) / self.interval_seconds
        floor = query.floor()
        fraction = query - floor
        lower = floor.long().clamp(0, samples - 1)
        upper = (floor.long() + 1).clamp(0, samples - 1)
        offset = predecessors[:, :, None] * samples
        allowed = edge_valid[:, :, None] & (query >= 0.0)
        return ((offset + lower).reshape(batch, -1), (offset + upper).reshape(batch, -1), fraction, allowed, delay, ticks)

    @staticmethod
    def _pull(signal: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
        """Gather B,N,T,D,C delayed source signals, preserving delay gradients."""
        (batch, nodes, samples, channels) = signal.shape
        flat = signal.reshape(batch, nodes * samples, channels)
        low = flat.gather(1, lower[..., None].expand(-1, -1, channels))
        high = flat.gather(1, upper[..., None].expand(-1, -1, channels))
        shape = (*fraction.shape, channels)
        return low.reshape(shape) * (1.0 - fraction[..., None]) + high.reshape(shape) * fraction[..., None]

    def propagate(self, features: torch.Tensor, travel_seconds: torch.Tensor, node_valid: torch.Tensor, predecessors: torch.Tensor, predecessor_valid: torch.Tensor, outgoing_degree: torch.Tensor, start_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """Propagate caller-supplied message features; useful for causal audits.

        A unit step is injected at observed start states, not at all roads.
        Outgoing branch splitting is uniform here, while learned content and
        downstream node scores determine route choice. Multiple walks are
        combined using their nonnegative transported mass.
        """
        (batch, nodes, width) = features.shape
        if travel_seconds.shape != (batch, nodes) or node_valid.shape != (batch, nodes):
            raise ValueError('Road durations/validity must match graph nodes')
        predecessors = predecessors.long()
        source_valid = node_valid.gather(1, predecessors.reshape(batch, -1)).reshape(*predecessors.shape)
        edges = predecessor_valid.bool() & source_valid.bool() & node_valid[..., None].bool()
        if self.mode == 'fixed-hop':
            duration = torch.full_like(travel_seconds.float(), self.interval_seconds)
        else:
            duration = travel_seconds.float().clamp_min(0.001)
        (lower, upper, fraction, allowed, edge_delay, ticks) = self._delay_plan(duration, predecessors, edges)
        degree = outgoing_degree.gather(1, predecessors.reshape(batch, -1)).reshape(*predecessors.shape).float().clamp_min(1)
        branch_weight = degree.reciprocal()[:, :, None, :, None]
        starts = start_valid.bool() & node_valid.bool()
        start_mass = starts.float() / starts.float().sum(-1, keepdim=True).clamp_min(1)
        samples = self.future_steps + 1
        mass = start_mass[:, :, None, None].expand(-1, -1, samples, 1)
        content = features[:, :, None] * mass
        (summed_mass, summed_content) = (mass, content)
        earliest = torch.where(starts, torch.zeros_like(duration), torch.full_like(duration, UNREACHED_SECONDS))
        first_arrival = earliest
        road_feature = self.road_update(features)[:, :, None]
        chunk = self.checkpoint_chunk or self.maximum_hops
        for begin in range(0, self.maximum_hops, chunk):
            iterations = min(chunk, self.maximum_hops - begin)

            def advance_block(mass, content, earliest, summed_mass, summed_content, first_arrival, _iterations=iterations):
                for _ in range(_iterations):
                    arrival = earliest.gather(1, predecessors.reshape(batch, -1)).reshape(*predecessors.shape) + edge_delay
                    arrival = arrival.masked_fill(~edges, UNREACHED_SECONDS)
                    causal = allowed & (ticks[None, None, :, None] >= arrival[:, :, None])
                    transported = self._pull(torch.cat((mass, content), -1), lower, upper, fraction) * causal[..., None] * branch_weight
                    incoming = transported.sum(3)
                    mass = incoming[..., :1]
                    mean_content = incoming[..., 1:] / mass.clamp_min(1e-06)
                    hidden = F.gelu(self.message_update(mean_content) + road_feature)
                    content = hidden * mass
                    summed_mass = summed_mass + mass
                    summed_content = summed_content + content
                    earliest = arrival.amin(-1)
                    first_arrival = torch.minimum(first_arrival, earliest)
                return (mass, content, earliest, summed_mass, summed_content, first_arrival)
            values = (mass, content, earliest, summed_mass, summed_content, first_arrival)
            if self.checkpoint_chunk and torch.is_grad_enabled():
                values = checkpoint(advance_block, *values, use_reentrant=False, preserve_rng_state=False)
            else:
                values = advance_block(*values)
            (mass, content, earliest, summed_mass, summed_content, first_arrival) = values
        temporal = summed_content / summed_mass.clamp_min(1e-06)
        return {'temporal_features': temporal[:, :, 1:].permute(0, 2, 1, 3), 'arrival_mass': summed_mass[:, :, 1:, 0].permute(0, 2, 1), 'earliest_arrival_s': first_arrival, 'effective_traversal_seconds': duration}

    def forward(self, node_features: torch.Tensor, history_state: torch.Tensor, travel_seconds: torch.Tensor, node_valid: torch.Tensor, predecessors: torch.Tensor, predecessor_valid: torch.Tensor, outgoing_degree: torch.Tensor, start_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = F.gelu(self.node_embedding(node_features) + self.history_embedding(history_state)[:, None])
        result = self.propagate(feature, travel_seconds, node_valid, predecessors, predecessor_valid, outgoing_degree, start_valid)
        result['temporal_hidden'] = self.output_projection(result['temporal_features'])
        return result
