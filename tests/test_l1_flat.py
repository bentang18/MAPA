"""The one attention block the released encoder runs, checked against a dense reference.

Tokens are ragged per contact and band, so there is no rectangle to reshape through. The
pack plan lays them out flat and array-contiguous with a per-token index and time
coordinate, and ``cu_seqlens`` delimits each array. The three properties that carry the
architecture are asserted here: the packed kernel equals dense per-array attention, a
contact attends only inside its own array, and it attends jointly over contact and time.
"""

from __future__ import annotations

import math

import torch

from mapa.models.attention import L1Block
from mapa.models.geometry import build_l1_geometry
from mapa.models.sidecar import build_sidecar

D, H, T = 32, 4, 6


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        region_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def _flat_layout(geom):
    """The rectangle unrolled: array-contiguous, contact-major, one row per (contact, time)."""
    canon = geom.gather_idx[geom.valid]                 # (N,) contact index, canonical order
    depth = geom.depth[geom.valid].repeat_interleave(T)  # (N*T,)
    time_pos = torch.arange(T).repeat(int(canon.shape[0]))
    n_per = geom.valid.sum(1)                            # (S,) contacts per array
    cu = torch.zeros(geom.n_arrays + 1, dtype=torch.int32)
    cu[1:] = (n_per * T).cumsum(0).to(torch.int32)
    array = torch.arange(geom.n_arrays).repeat_interleave(n_per * T)
    return canon, depth, time_pos, cu, int((n_per * T).max()), array


def _dense_reference(blk, x_flat, depth, time_pos, cu):
    """Dense softmax attention inside each array block, written out by hand.

    Same weights and same rotary as the block, but no packed kernel and no masking: every
    array is sliced out and attended over on its own. Any disagreement is a packing bug.
    """
    h = blk.norm1(x_flat)
    m = h.shape[0]
    qkv = blk.qkv(h).reshape(m, 3, blk.n_heads, blk.head_dim)
    q, k, v = qkv.unbind(dim=1)
    qh, kh = blk.rope(q.transpose(0, 1), k.transpose(0, 1), depth, time_pos)
    q, k = qh.transpose(0, 1), kh.transpose(0, 1)
    ctx = torch.zeros_like(v)
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        qs, ks, vs = q[a:b].transpose(0, 1), k[a:b].transpose(0, 1), v[a:b].transpose(0, 1)
        att = (qs @ ks.transpose(-2, -1)) / math.sqrt(blk.head_dim)
        ctx[a:b] = (att.softmax(-1) @ vs).transpose(0, 1)
    x = x_flat + blk.out(ctx.reshape(m, -1))
    return x + blk.mlp(blk.norm2(x))


def test_packed_attention_matches_dense_per_array_attention() -> None:
    torch.manual_seed(0)
    _, geom = _session()
    blk = L1Block(D, H).eval()
    canon, depth, time_pos, cu, max_seqlen, _ = _flat_layout(geom)
    x = torch.randn(int(canon.shape[0]) * T, D)

    got = blk.forward_flat(x, depth, time_pos, cu, cu.to(torch.int64), max_seqlen)
    ref = _dense_reference(blk, x, depth, time_pos, cu)
    assert torch.allclose(got, ref, atol=1e-5)


def test_a_contact_attends_only_inside_its_own_array() -> None:
    torch.manual_seed(1)
    _, geom = _session()
    blk = L1Block(D, H).eval()
    canon, depth, time_pos, cu, max_seqlen, array = _flat_layout(geom)

    x = torch.randn(int(canon.shape[0]) * T, D)
    a = blk.forward_flat(x, depth, time_pos, cu, cu.to(torch.int64), max_seqlen)
    x2 = x.clone()
    x2[array == 1] += torch.randn_like(x2[array == 1]) * 3.0
    b = blk.forward_flat(x2, depth, time_pos, cu, cu.to(torch.int64), max_seqlen)

    assert torch.allclose(a[array == 0], b[array == 0], atol=1e-5)
    assert not torch.allclose(a[array == 1], b[array == 1], atol=1e-4)


def test_attention_is_joint_over_contact_and_time() -> None:
    """Perturbing one contact at a late time moves another contact at time zero.

    Time and contact are one attention, not two factorized ones. A factorized block would
    leave the earlier time untouched.
    """
    torch.manual_seed(2)
    _, geom = _session()
    blk = L1Block(D, H).eval()
    canon, depth, time_pos, cu, max_seqlen, array = _flat_layout(geom)

    x = torch.randn(int(canon.shape[0]) * T, D)
    a = blk.forward_flat(x, depth, time_pos, cu, cu.to(torch.int64), max_seqlen)
    x2 = x.clone()
    x2[1 * T + 2] += torch.randn(D) * 3.0  # array LA, second contact, time 2
    b = blk.forward_flat(x2, depth, time_pos, cu, cu.to(torch.int64), max_seqlen)
    assert not torch.allclose(a[0], b[0], atol=1e-4)  # first contact, time 0, moved
