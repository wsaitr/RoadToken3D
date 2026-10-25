"""Decoupled coordinate, road-motion and frozen-forecast risk learning.

All geometry is tensor-only. No labels enter forward. A Cartesian output is
explicitly unconstrained; a road output follows the cached candidate polyline.
The legacy stationary slot is retained for file compatibility but invalid.
"""
from torch import nn

class ResidualMLP(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, x):
        return x + self.net(x)
