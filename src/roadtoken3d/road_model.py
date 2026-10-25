from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class AscendCausalSelfAttention(nn.Module):
    """Causal attention expressed only with native Linear/BMM/Softmax ops.

    PyTorch's TransformerEncoder eval fast path dispatches
    ``aten::_transformer_encoder_layer_fwd``.  CANN 8.0 RC3 sends that op to
    the host CPU.  This implementation keeps both training and validation on
    the NPU and uses BMM dimensions aligned by ``RoadToken3DModel``.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError('d_model must be divisible by nhead')
        self.nhead = int(nhead)
        self.head_dim = int(d_model // nhead)
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        (batch, length, width) = value.shape
        qkv = self.qkv(value).reshape(batch, length, 3, self.nhead, self.head_dim)
        (query, key, content) = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3).reshape(batch * self.nhead, length, self.head_dim)
        key = key.permute(0, 2, 3, 1).reshape(batch * self.nhead, self.head_dim, length)
        content = content.permute(0, 2, 1, 3).reshape(batch * self.nhead, length, self.head_dim)
        score = torch.bmm(query, key) * self.scale
        score = score.masked_fill(causal_mask, -10000.0)
        probability = torch.softmax(score.float(), dim=-1).to(content.dtype)
        probability = self.attention_dropout(probability)
        attended = torch.bmm(probability, content)
        attended = attended.reshape(batch, self.nhead, length, self.head_dim)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, length, width)
        return self.output_dropout(self.projection(attended))

class AscendCausalBlock(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = AscendCausalSelfAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(self.norm1(value), causal_mask)
        hidden = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(value)))))
        return value + self.residual_dropout(hidden)

class AscendCausalEncoder(nn.Module):

    def __init__(self, d_model: int, nhead: int, layers: int, dim_feedforward: int, dropout: float, maximum_length: int, attention_window: int=0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AscendCausalBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(layers)])
        positions = torch.arange(maximum_length)
        query = positions[:, None]
        key = positions[None, :]
        causal_mask = key > query
        if int(attention_window) > 0:
            causal_mask = causal_mask | (key < query - int(attention_window) + 1)
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mask = self.causal_mask[:value.shape[1], :value.shape[1]]
        for layer in self.layers:
            value = layer(value, mask)
        return value
