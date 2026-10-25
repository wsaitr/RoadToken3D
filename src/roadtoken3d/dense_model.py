"""Dense-coordinate CONTROLS for the road model, not a claimed road constraint.

Native tensor attention, explicit horizon queries, no positive movement floor,
and no legacy road-motion fusion. A forecast is never passed a future road or
future Cartesian label. Road-native decoding is a separate component.
"""
from __future__ import annotations
import torch
from torch import nn
from .road_model import AscendCausalEncoder

class DenseCoordinateForecaster(nn.Module):

    def __init__(self, history_steps=41, future_steps=240, interval_seconds=15, width=128, layers=2, modes=1, dropout=0.1, architecture='attention'):
        super().__init__()
        if architecture not in ('attention', 'linear'):
            raise ValueError('Unknown dense control architecture')
        (self.future_steps, self.modes, self.architecture) = (future_steps, modes, architecture)
        self.interval_seconds = float(interval_seconds)
        self.register_buffer('xyz_scale', torch.tensor([100.0, 100.0, 20.0]), persistent=True)
        self.register_buffer('future_seconds', torch.arange(1, future_steps + 1).float() * interval_seconds, persistent=True)
        output_scale = torch.sqrt(self.future_seconds / 60.0).clamp_min(0.5) * 100.0
        self.register_buffer('output_xy_scale', output_scale, persistent=True)
        if architecture == 'linear':
            if modes != 1:
                raise ValueError('Linear control is single-mode')
            self.linear = nn.Linear(history_steps * 8 + 7, future_steps * 3)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            return
        self.project = nn.Linear(8, width)
        self.context = nn.Linear(7, width)
        self.position = nn.Parameter(torch.randn(history_steps, width) * 0.02)
        self.encoder = AscendCausalEncoder(width, 4, layers, 4 * width, dropout, history_steps)
        self.summary = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.LayerNorm(width))
        self.horizon = nn.Parameter(torch.randn(future_steps, width))
        self.mode = nn.Parameter(torch.randn(modes, width) * 0.2)
        self.decode = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width), nn.GELU())
        self.xyz_head = nn.Linear(width, 3)
        self.confidence = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, history_xyz, history_xy_valid, history_z_valid, origin_context, time_features):
        valid = history_xy_valid.float()
        zvalid = history_z_valid.float()
        mask = torch.stack((valid, valid, zvalid), -1).bool()
        h = torch.where(mask, history_xyz, torch.zeros_like(history_xyz)) / self.xyz_scale
        delta = torch.cat((torch.zeros_like(h[:, :1]), h[:, 1:] - h[:, :-1]), 1)
        previous_valid = torch.cat((valid[:, :1], valid[:, :-1]), 1)
        previous_zvalid = torch.cat((zvalid[:, :1], zvalid[:, :-1]), 1)
        delta = delta * torch.stack((valid * previous_valid, valid * previous_valid, zvalid * previous_zvalid), -1)
        features = torch.cat((h, delta, valid.unsqueeze(-1), history_z_valid.float().unsqueeze(-1)), -1)
        context = torch.cat((origin_context, time_features), -1)
        if self.architecture == 'linear':
            raw = self.linear(torch.cat((features.flatten(1), context), -1)).reshape(-1, 1, self.future_steps, 3)
            logits = raw.new_zeros((len(raw), 1))
        else:
            encoded = self.encoder(self.project(features) + self.position[None] + self.context(context)[:, None])
            pooled = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            summary = self.summary(torch.cat((encoded[:, -1], pooled), -1))
            time_query = self.horizon[None, None] + self.mode[None, :, None]
            time_query = time_query.expand(len(summary), -1, -1, -1)
            state = self.decode(torch.cat((summary[:, None, None].expand_as(time_query), time_query), -1))
            raw = self.xyz_head(state)
            logits = self.confidence(state.mean(2)).squeeze(-1)
        xyz = torch.cat((raw[..., :2].float() * self.output_xy_scale[None, None, :, None], raw[..., 2:3].float() * 20.0), -1)
        output = {'trajectories': xyz, 'mode_logits': logits.float()}
        if self.architecture == 'attention':
            output['latent_horizon'] = state
        return output
