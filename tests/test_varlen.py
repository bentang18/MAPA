"""Block-diagonal attention over the flat token set.

The CPU backend is what the other tests compare against, so it is pinned here to an
independent per-block softmax written out by hand. Two structural guarantees are checked
alongside it: no token reads a key outside its own block, and an empty block is inert.
The GPU backend has to agree with the CPU one, so that is checked wherever CUDA is present.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from mapa.models.varlen import _reference_block_diag, _segment_ids, gpu_block_diag


def _per_block_truth(q, k, v, cu):
    """Dense softmax attention inside each block, one block at a time."""
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        if b == a:
            continue
        qs, ks, vs = (t[a:b].transpose(0, 1) for t in (q, k, v))
        att = torch.softmax((qs @ ks.transpose(-1, -2)) * scale, dim=-1)
        out[a:b] = (att @ vs).transpose(0, 1)
    return out


def test_segment_ids_from_cu_seqlens() -> None:
    cu = torch.tensor([0, 3, 3, 5], dtype=torch.int32)  # block lengths 3, 0, 2
    assert _segment_ids(cu, total=5).tolist() == [0, 0, 0, 2, 2]


def test_reference_matches_per_block_softmax() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(7, 2, 8, dtype=torch.float64) for _ in range(3))
    cu = torch.tensor([0, 4, 7], dtype=torch.int32)  # two arrays, 4 and 3 tokens
    assert torch.allclose(_reference_block_diag(q, k, v, cu), _per_block_truth(q, k, v, cu),
                          atol=1e-10)


def test_no_token_reads_outside_its_own_block() -> None:
    torch.manual_seed(1)
    q, k, v = (torch.randn(6, 1, 4, dtype=torch.float64) for _ in range(3))
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    out1 = _reference_block_diag(q, k, v, cu)
    v2 = v.clone()
    v2[3:] += 100.0
    out2 = _reference_block_diag(q, k, v2, cu)
    assert torch.allclose(out1[:3], out2[:3], atol=1e-10)
    assert not torch.allclose(out1[3:], out2[3:])


def test_an_empty_block_is_inert() -> None:
    """An array with no surviving contacts adds a bound and no tokens, so the output
    must not move."""
    torch.manual_seed(2)
    q, k, v = (torch.randn(5, 2, 4, dtype=torch.float64) for _ in range(3))
    dense = torch.tensor([0, 3, 5], dtype=torch.int32)
    gappy = torch.tensor([0, 3, 3, 3, 5], dtype=torch.int32)
    assert torch.allclose(_reference_block_diag(q, k, v, dense),
                          _reference_block_diag(q, k, v, gappy), atol=1e-12)


def test_a_single_block_is_plain_attention() -> None:
    torch.manual_seed(3)
    q, k, v = (torch.randn(5, 2, 8, dtype=torch.float64) for _ in range(3))
    cu = torch.tensor([0, 5], dtype=torch.int32)
    qh, kh, vh = (t.transpose(0, 1)[None] for t in (q, k, v))
    want = F.scaled_dot_product_attention(qh, kh, vh, scale=1.0 / math.sqrt(8))[0].transpose(0, 1)
    assert torch.allclose(_reference_block_diag(q, k, v, cu), want, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the jagged path is CUDA only")
def test_the_gpu_backend_agrees_with_the_reference() -> None:
    dev = torch.device("cuda")
    torch.manual_seed(7)
    q, k, v = (torch.randn(9, 4, 32, device=dev, requires_grad=True) for _ in range(3))
    cu = torch.tensor([0, 4, 9], dtype=torch.int64, device=dev)
    got = gpu_block_diag(q, k, v, cu, max_seqlen=5)
    want = _reference_block_diag(q, k, v, cu.to(torch.int32))
    assert torch.allclose(got, want, atol=1e-4)
    got.sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in (q, k, v))
