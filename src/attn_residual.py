"""Attention Residual modules for AttnRes-QST.

Implements the Attention Residuals idea from Kimi/Moonshot AI, adapted as a
drop-in replacement for the "previous side-layer hidden state" that QST's
z-gate carries forward (see qst_arch_notes.md §4.1 for the insertion point).

Given a history of prior side-layer hidden states
    prev = [h_0, h_1, ..., h_{L-1}],   each h_i of shape [B, T, H]
we aggregate them via a learnable pseudo-query q and a per-position Norm:

    alpha_i   = softmax_i( q^T · Norm(h_i) )       for i = 0..L-1
    h_res     = sum_i alpha_i · h_i                [B, T, H]

Two variants:
    * AttentionResidual         — uses the full history (Full AttnRes)
    * BlockAttentionResidual    — sliding window over last `block_size` states

Design notes
------------
- `q` is zero-initialized so that alpha starts uniform. This keeps early
  training stable and makes `Uniform Average` a proper ablation baseline
  (freeze q at zero and norm=none).
- Norm defaults to RMSNorm (following Kimi). LayerNorm / none are supported
  for the §6.4 ablation.
- For Block AttnRes with window larger than current history length, we fall
  back to using the full history — this avoids special-casing early layers.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def _make_norm(kind: str, dim: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "rmsnorm":
        return RMSNorm(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"Unknown norm kind: {kind}")


class AttentionResidual(nn.Module):
    """Full Attention Residual: aggregate all prior hidden states.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality H of each hidden state tensor.
    norm_type : str, default "rmsnorm"
        One of {"rmsnorm", "layernorm", "none"}.
    freeze_query : bool, default False
        If True, the pseudo-query `q` is permanently zero (requires_grad=False).
        With q == 0 the softmax produces a uniform distribution, so the output
        degenerates to a simple unweighted mean of the hidden states ("Uniform
        Average Residual" ablation). The norm module still exists but has no
        effect on the output, since it only enters through `q^T · Norm(h_i)`
        which is zero regardless.
    """

    def __init__(self, hidden_dim: int, norm_type: str = "rmsnorm",
                 freeze_query: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_type = norm_type
        self.freeze_query = freeze_query
        self.norm = _make_norm(norm_type, hidden_dim)
        self.q = nn.Parameter(torch.zeros(hidden_dim), requires_grad=not freeze_query)

    def _aggregate(self, stacked: torch.Tensor) -> torch.Tensor:
        # stacked: [B, T, L, H]
        normed = self.norm(stacked)                                # [B, T, L, H]
        scores = torch.einsum("btlh,h->btl", normed, self.q)       # [B, T, L]
        alpha = F.softmax(scores, dim=-1)                          # [B, T, L]
        h_res = torch.einsum("btl,btlh->bth", alpha, stacked)      # [B, T, H]
        return h_res

    def forward(self, prev_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        if len(prev_hidden_states) == 0:
            raise ValueError("prev_hidden_states must contain at least one tensor")
        stacked = torch.stack(prev_hidden_states, dim=2)           # [B, T, L, H]
        return self._aggregate(stacked)


class BlockAttentionResidual(AttentionResidual):
    """Block Attention Residual: aggregate only the last `block_size` states.

    When the provided history is shorter than `block_size`, falls back to using
    the full available history (no padding / no zeroing).
    """

    def __init__(self, hidden_dim: int, block_size: int = 2,
                 norm_type: str = "rmsnorm", freeze_query: bool = False):
        super().__init__(hidden_dim=hidden_dim, norm_type=norm_type,
                         freeze_query=freeze_query)
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = block_size

    def forward(self, prev_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        window = prev_hidden_states[-self.block_size:]
        return super().forward(window)
