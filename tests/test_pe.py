"""The two spatial encodings, and what keeps them disjoint.

Two mechanisms, disjoint by design:

  L1RoPE          — within-sensor JOINT spatiotemporal rotary. EQUAL split of
                    head_dim across (contact-index, time), standard freq schedule
                    per axis. Its load-bearing property is RELATIVITY: the q·k
                    score depends only on (Δindex, Δtime), never absolute position
                    — which is exactly why a global per-array depth offset
                    needs no correction.
  RegionIdentity  — absolute region identity: a LEARNED embedding indexed by the
                    DKT hard tag (cross-subject-meaningful), added to tokens. NO
                    geometric RoPE across sensors (no shared metric).
"""

from __future__ import annotations

import pytest
import torch

from mapa.models.pe import (
    L1RoPE,
    RegionIdentityEmbed,
)


def _qk(heads: int, seq: int, hd: int):
    torch.manual_seed(0)
    q = torch.randn(1, heads, seq, hd)
    k = torch.randn(1, heads, seq, hd)
    return q, k


def _score(rope: L1RoPE, q, k, idx, t):
    qr, kr = rope(q, k, idx, t)
    return torch.einsum("bhid,bhjd->bhij", qr, kr)


def test_head_dim_must_be_divisible_by_four() -> None:
    L1RoPE(head_dim=64)  # ok
    with pytest.raises(ValueError):
        L1RoPE(head_dim=62)  # 62 % 4 != 0


def test_rope_preserves_shape() -> None:
    rope = L1RoPE(head_dim=64)
    q, k = _qk(4, 6, 64)
    idx = torch.arange(6).unsqueeze(0)
    t = torch.arange(6).unsqueeze(0)
    qr, kr = rope(q, k, idx, t)
    assert qr.shape == q.shape and kr.shape == k.shape


def test_space_false_kills_the_index_axis_and_leaves_time_bit_identical() -> None:
    """A2's premise, as three assertions rather than one.

    ``space=False`` must (1) make the rotary table INDEPENDENT of contact index,
    (2) leave the index half at the IDENTITY rotation (cos 1 / sin 0) rather than at
    some other fixed angle, and (3) leave the TIME half BIT-identical to the locked
    ``base_index=8`` model — the ablation has to be one-variable, and the time axis is
    the conventional half we are not testing.
    """
    on, off = L1RoPE(head_dim=64), L1RoPE(head_dim=64, space=False)
    hd_half = 32  # head_dim/2 rotary entries; first 16 = index pairs, last 16 = time
    t = torch.tensor([[0, 1, 2, 3, 4]])
    idx_a = torch.tensor([[0, 1, 2, 4, 5]])
    idx_b = torch.tensor([[7, 3, 0, 11, 2]])  # a DIFFERENT, non-shifted index set

    cos_a, sin_a = off.cos_sin(idx_a, t)
    cos_b, sin_b = off.cos_sin(idx_b, t)
    assert torch.equal(cos_a, cos_b) and torch.equal(sin_a, sin_b)

    n_idx = hd_half  # index pairs occupy the first head_dim/2 dims after interleave
    assert torch.allclose(cos_a[..., :n_idx], torch.ones_like(cos_a[..., :n_idx]))
    assert torch.allclose(sin_a[..., :n_idx], torch.zeros_like(sin_a[..., :n_idx]))

    cos_on, sin_on = on.cos_sin(idx_a, t)
    assert torch.equal(cos_on[..., n_idx:], cos_a[..., n_idx:])
    assert torch.equal(sin_on[..., n_idx:], sin_a[..., n_idx:])

    # and the consequence that matters: the q.k score no longer sees index at all.
    q, k = _qk(2, 5, 64)
    assert torch.equal(_score(off, q, k, idx_a, t), _score(off, q, k, idx_b, t))
    assert not torch.allclose(_score(on, q, k, idx_a, t), _score(on, q, k, idx_b, t))
    print(
        "[check] OK space=False: table index-INDEPENDENT, index half = identity "
        "(cos 1/sin 0), time half BIT-identical to base_index=8, score index-blind"
    )


def test_score_is_translation_invariant_on_both_axes() -> None:
    # The RoPE relativity property: shifting index and/or time by a constant
    # leaves every pairwise q·k score unchanged.
    rope = L1RoPE(head_dim=64)
    q, k = _qk(2, 5, 64)
    idx = torch.tensor([[0, 1, 2, 4, 5]])  # note the gap (dropped contact 3)
    t = torch.tensor([[0, 1, 2, 3, 4]])
    base = _score(rope, q, k, idx, t)
    for da, db in [(3, 0), (0, 7), (3, 7), (-2, 4)]:
        shifted = _score(rope, q, k, idx + da, t + db)
        assert torch.allclose(base, shifted, atol=1e-4), f"shift {(da, db)} changed score"


def test_score_is_not_position_degenerate() -> None:
    # Different relative offsets must give different scores (RoPE actually encodes
    # position, not a no-op).
    rope = L1RoPE(head_dim=64)
    q, k = _qk(1, 3, 64)
    t = torch.zeros(1, 3, dtype=torch.long)
    s_close = _score(rope, q, k, torch.tensor([[0, 1, 2]]), t)
    s_far = _score(rope, q, k, torch.tensor([[0, 10, 20]]), t)
    assert not torch.allclose(s_close, s_far, atol=1e-3)


def test_spatial_and_temporal_axes_are_independent() -> None:
    # Varying only time (index fixed) must leave the score identical to a pure
    # temporal-only rotation, and vice-versa — the two axes occupy disjoint
    # head_dim halves.
    rope = L1RoPE(head_dim=64)
    q, k = _qk(1, 4, 64)
    idx0 = torch.zeros(1, 4, dtype=torch.long)
    t = torch.tensor([[0, 1, 2, 3]])
    # Shift index alone while time fixed: with identical index, the spatial half
    # contributes a constant (Δindex=0) so only the temporal half varies.
    s_a = _score(rope, q, k, idx0, t)
    s_b = _score(rope, q, k, idx0 + 5, t)  # still all-equal index → Δindex=0
    assert torch.allclose(s_a, s_b, atol=1e-4)


def test_default_base_makes_every_pair_position_live() -> None:
    # Audit L12/#8: base must fit OUR ranges, not the LLM-inherited 10000. A pair
    # with frequency f turns R·f radians across a range R; a pair turning <<1 rad
    # end-to-end is position-DEAD. The default base_index/base_time must leave EVERY
    # rotary pair turning a meaningful angle across index (~20 contacts) and time
    # (128 slots) — and the old 10000 must NOT (regression witness).
    R_index, R_time = 20.0, 128.0
    rope = L1RoPE(head_dim=64)  # encoder width; pairs = 16 per axis
    idx_turn = R_index * rope.idx_freq  # (pairs,) radians across the index range
    t_turn = R_time * rope.t_freq
    assert (idx_turn >= 1.0).all(), f"dead index pairs: {idx_turn.tolist()}"
    assert (t_turn >= 1.0).all(), f"dead time pairs: {t_turn.tolist()}"
    # lowest-freq pair stays under a full turn (≤2π) ⇒ no wrap-around ambiguity
    assert idx_turn.min() <= 2 * torch.pi and t_turn.min() <= 2 * torch.pi

    dead = L1RoPE(head_dim=64, base_index=10_000.0, base_time=10_000.0)
    assert (R_index * dead.idx_freq < 1.0).sum() >= 10  # most index pairs were dead
    assert (R_time * dead.t_freq < 1.0).sum() >= 6


def test_cos_sin_and_rotate_reconstruct_forward_exactly() -> None:
    # B5 hoist mechanism: forward(q,k,idx,t) must be BIT-IDENTICAL to
    # rotate(q,k,*cos_sin(idx,t)). This is what lets the tower build the table ONCE
    # and feed every L1 block instead of each block recomputing it.
    rope = L1RoPE(head_dim=64)
    q, k = _qk(4, 6, 64)
    idx = torch.tensor([[0, 1, 2, 4, 5, 6]])
    t = torch.tensor([[0, 1, 2, 3, 4, 5]])
    q_ref, k_ref = rope(q, k, idx, t)
    cos, sin = rope.cos_sin(idx, t)
    q_hoist, k_hoist = rope.rotate(q, k, cos, sin)
    assert torch.equal(q_ref, q_hoist)  # exact, not allclose
    assert torch.equal(k_ref, k_hoist)


def test_cos_sin_is_fp32_regardless_of_input_dtype() -> None:
    # The table must stay fp32 (the rope math runs in fp32, result down-cast AFTER)
    # — a bf16 table would change the rounding and break equivalence.
    rope = L1RoPE(head_dim=64)
    idx = torch.tensor([[0, 1, 2]])
    t = torch.tensor([[0, 1, 2]])
    cos, sin = rope.cos_sin(idx, t)
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32


def test_shared_table_matches_per_instance_recompute() -> None:
    # Two L1RoPE instances with the same head_dim/bases produce the SAME table, so a
    # table built by block[0].rope applied via block[1].rope.rotate is exact — the
    # invariant the tower relies on to share one table across all L1 blocks.
    a, b = L1RoPE(head_dim=64), L1RoPE(head_dim=64)
    q, k = _qk(2, 5, 64)
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    t = torch.tensor([[0, 1, 2, 3, 4]])
    cos, sin = a.cos_sin(idx, t)  # built by instance a
    q_shared, k_shared = b.rotate(q, k, cos, sin)  # applied via instance b
    q_ref, k_ref = b(q, k, idx, t)  # b recomputes its own
    assert torch.equal(q_ref, q_shared)
    assert torch.equal(k_ref, k_shared)


def test_region_identity_shape_and_indexing() -> None:
    emb = RegionIdentityEmbed(n_regions=74, d_model=256)
    region_id = torch.tensor([[3, 3, 7, 0]])
    out = emb(region_id)
    assert out.shape == (1, 4, 256)
    # same region id → identical vector; different id → different.
    assert torch.allclose(out[0, 0], out[0, 1])
    assert not torch.allclose(out[0, 0], out[0, 2])


def test_region_identity_is_single_learned_embedding() -> None:
    emb = RegionIdentityEmbed(n_regions=74, d_model=256)
    embeds = [m for m in emb.modules() if isinstance(m, torch.nn.Embedding)]
    assert len(embeds) == 1
    assert embeds[0].weight.shape == (74, 256)
    assert embeds[0].weight.requires_grad  # learned
