"""The encoder stack: sizing, initialisation, and what a checkpoint reveals about width.

Depth is fixed at twelve blocks and head dimension at 64, so one integer sizes the model.
The released width is 384, which is ViT-Small. The ablation ladder was tuned at 256, and
that default has to stay exact, or runs from before the width arm stop being comparable.
"""

from __future__ import annotations

import math

import pytest

from mapa.models.towers import (
    DEPTH,
    ENC_D_MODEL,
    HEAD_DIM,
    N_LEVELS,
    VIT_SMALL,
    build_encoder,
    n_heads_for,
)


def test_the_default_width_is_unchanged() -> None:
    assert (ENC_D_MODEL, n_heads_for(ENC_D_MODEL)) == (256, 4)


def test_vit_small_is_384_over_6_heads() -> None:
    assert (VIT_SMALL, n_heads_for(VIT_SMALL)) == (384, 6)


@pytest.mark.parametrize("d", [128, 192, 256, 384, 512, 768])
def test_head_dimension_is_held_across_the_ladder(d: int) -> None:
    assert d // n_heads_for(d) == HEAD_DIM


@pytest.mark.parametrize("bad", [300, 100, 0, -64])
def test_an_off_ladder_width_fails_at_construction(bad: int) -> None:
    with pytest.raises(ValueError, match="multiple of 64"):
        n_heads_for(bad)


def test_the_stack_is_twelve_blocks_with_four_supervised_taps() -> None:
    enc = build_encoder(n_regions=8, d_model=VIT_SMALL)
    assert len(enc.blocks) == DEPTH
    assert enc.sup_taps == (3, 6, 9, 12) and N_LEVELS == 4
    assert len(enc.norms_block) == N_LEVELS and enc.norm_out is None


def test_width_is_readable_off_a_state_dict() -> None:
    """Loading infers width from the checkpoint rather than taking a flag, so nothing has
    to be remembered at encode time."""
    sd = build_encoder(n_regions=8, d_model=VIT_SMALL).state_dict()
    assert sd["blocks.0.norm1.weight"].shape[0] == VIT_SMALL
    assert build_encoder(n_regions=8).state_dict()["blocks.0.norm1.weight"].shape[0] == 256


def test_a_wrong_width_shell_cannot_silently_load_a_checkpoint() -> None:
    wide = build_encoder(n_regions=8, d_model=VIT_SMALL).state_dict()
    narrow = build_encoder(n_regions=8)
    with pytest.raises(RuntimeError, match="size mismatch"):
        narrow.load_state_dict(wide, strict=False)


def test_initialisation_matches_v_jepa_2() -> None:
    """Linear weights are trunc_normal(0.02) with zero bias, then the attention output
    projection and the second MLP layer are divided by sqrt(2*layer), one-indexed."""
    enc = build_encoder(n_regions=8)
    b0 = enc.blocks[0]
    assert b0.qkv.bias is not None and b0.out.bias is not None
    assert b0.qkv.bias.abs().max().item() == 0.0
    assert abs(b0.mlp.fc1.weight.std().item() - 0.02) < 0.004
    assert b0.mlp.fc1.bias.abs().max().item() == 0.0

    s0 = enc.blocks[0].out.weight.std().item()
    s5 = enc.blocks[5].out.weight.std().item()
    assert s0 > s5
    assert abs(s0 / s5 - math.sqrt(6.0)) < 0.4
