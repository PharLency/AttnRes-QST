"""Unit tests for src/attn_residual.py.

Run directly:
    python src/test_attn_residual.py

Covers the checklist in .skills/plan.md §3.3:
    * shape correctness
    * alpha sums to 1 (softmax invariant)
    * single-layer passthrough (softmax over 1 element = 1)
    * gradient flows to both q and prev hidden states
    * block sliding window behavior
    * norm options
    * uniform-at-init property (q=0 => alpha uniform)
"""

import torch
import torch.nn.functional as F

from attn_residual import (
    AttentionResidual,
    BlockAttentionResidual,
    RMSNorm,
    _make_norm,
)


def test_shape():
    m = AttentionResidual(hidden_dim=64)
    prev = [torch.randn(2, 5, 64) for _ in range(4)]
    out = m(prev)
    assert out.shape == (2, 5, 64), out.shape


def test_alpha_sums_to_one():
    m = AttentionResidual(hidden_dim=32)
    prev = [torch.randn(1, 3, 32) for _ in range(5)]
    stacked = torch.stack(prev, dim=2)
    normed = m.norm(stacked)
    scores = torch.einsum("btlh,h->btl", normed, m.q)
    alpha = F.softmax(scores, dim=-1)
    sums = alpha.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), sums.tolist()


def test_single_layer_passthrough():
    """When there is only one prior hidden, softmax is trivially 1 and output == input."""
    m = AttentionResidual(hidden_dim=8)
    h0 = torch.randn(1, 2, 8)
    out = m([h0])
    assert torch.allclose(out, h0, atol=1e-5), (out - h0).abs().max().item()


def test_uniform_at_init():
    """q init=0 and norm=none => alpha uniform => output = mean of inputs."""
    m = AttentionResidual(hidden_dim=32, norm_type="none")
    prev = [torch.randn(1, 1, 32) for _ in range(4)]
    stacked = torch.stack(prev, dim=2)
    expected = stacked.mean(dim=2)
    out = m(prev)
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max().item()


def test_gradient_to_query():
    m = AttentionResidual(hidden_dim=16, norm_type="none")
    prev = [torch.randn(2, 3, 16) for _ in range(3)]
    out = m(prev)
    out.sum().backward()
    assert m.q.grad is not None, "q has no gradient"
    assert m.q.grad.abs().sum().item() > 0, "q gradient is zero for random inputs"


def test_gradient_to_prev():
    m = AttentionResidual(hidden_dim=16)
    prev = [torch.randn(2, 3, 16, requires_grad=True) for _ in range(3)]
    out = m(prev)
    out.sum().backward()
    for i, h in enumerate(prev):
        assert h.grad is not None, f"prev[{i}] has no gradient"
        assert h.grad.abs().sum().item() > 0, f"prev[{i}] gradient is zero"


def test_block_window_uniform():
    """Block=2, q=0, norm=none => output = mean of last 2 inputs."""
    m = BlockAttentionResidual(hidden_dim=8, block_size=2, norm_type="none")
    # h_i is a constant-i vector, so mean of last 2 (i=3, i=4) is 3.5
    prev = [torch.full((1, 1, 8), float(i)) for i in range(5)]
    out = m(prev)
    assert torch.allclose(out, torch.full_like(out, 3.5), atol=1e-5), out


def test_block_shorter_than_block_size():
    """Falls back to full history when history shorter than block_size."""
    m = BlockAttentionResidual(hidden_dim=4, block_size=5)
    prev = [torch.randn(1, 1, 4) for _ in range(2)]
    out = m(prev)
    assert out.shape == (1, 1, 4)


def test_rmsnorm_preserves_shape():
    n = RMSNorm(16)
    x = torch.randn(2, 3, 16)
    y = n(x)
    assert y.shape == x.shape


def test_norm_options():
    assert isinstance(_make_norm("rmsnorm", 8), RMSNorm)
    assert isinstance(_make_norm("layernorm", 8), torch.nn.LayerNorm)
    assert isinstance(_make_norm("none", 8), torch.nn.Identity)
    for good in ["RMSNorm", "LayerNorm", "None", "", "identity"]:
        _make_norm(good, 8)  # should not raise
    try:
        _make_norm("bogus", 8)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown norm kind")


def test_parameter_count_small():
    """AttnRes should be tiny: just q (+ norm scale if present)."""
    m = AttentionResidual(hidden_dim=64, norm_type="rmsnorm")
    n = sum(p.numel() for p in m.parameters())
    # q (64) + rmsnorm.weight (64) = 128
    assert n == 128, n
    m2 = AttentionResidual(hidden_dim=64, norm_type="layernorm")
    n2 = sum(p.numel() for p in m2.parameters())
    # q (64) + layernorm.weight (64) + layernorm.bias (64) = 192
    assert n2 == 192, n2
    m3 = AttentionResidual(hidden_dim=64, norm_type="none")
    n3 = sum(p.numel() for p in m3.parameters())
    assert n3 == 64, n3


def test_dtype_preserved():
    m = AttentionResidual(hidden_dim=16).to(torch.float32)
    prev = [torch.randn(1, 1, 16, dtype=torch.float32) for _ in range(3)]
    out = m(prev)
    assert out.dtype == torch.float32


def test_freeze_query_equals_uniform_mean():
    """With freeze_query=True, q is frozen at 0 → alpha is uniform → output = mean."""
    m = AttentionResidual(hidden_dim=16, norm_type="rmsnorm", freeze_query=True)
    assert not m.q.requires_grad
    prev = [torch.randn(2, 3, 16) for _ in range(5)]
    expected = torch.stack(prev, dim=2).mean(dim=2)
    out = m(prev)
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max().item()


def test_freeze_query_no_grad_accumulates():
    """Frozen q should never accumulate gradient, even when loss depends on output."""
    m = AttentionResidual(hidden_dim=8, norm_type="rmsnorm", freeze_query=True)
    prev = [torch.randn(1, 2, 8, requires_grad=True) for _ in range(3)]
    out = m(prev)
    out.sum().backward()
    assert m.q.grad is None, "q should have no .grad since requires_grad=False"
    # Input gradient should still flow
    for h in prev:
        assert h.grad is not None


def test_block_freeze_query():
    m = BlockAttentionResidual(hidden_dim=8, block_size=2,
                               norm_type="none", freeze_query=True)
    prev = [torch.full((1, 1, 8), float(i)) for i in range(5)]
    out = m(prev)
    # mean of last 2 = (3.0 + 4.0) / 2 = 3.5
    assert torch.allclose(out, torch.full_like(out, 3.5), atol=1e-5)


if __name__ == "__main__":
    tests = [
        test_shape,
        test_alpha_sums_to_one,
        test_single_layer_passthrough,
        test_uniform_at_init,
        test_gradient_to_query,
        test_gradient_to_prev,
        test_block_window_uniform,
        test_block_shorter_than_block_size,
        test_rmsnorm_preserves_shape,
        test_norm_options,
        test_parameter_count_small,
        test_dtype_preserved,
        test_freeze_query_equals_uniform_mean,
        test_freeze_query_no_grad_accumulates,
        test_block_freeze_query,
    ]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
