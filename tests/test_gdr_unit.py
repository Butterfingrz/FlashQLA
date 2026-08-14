# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import pytest
import torch

from gdr_common import (
    RTOL,
    HEAD_DIM_K,
    HEAD_DIM_V,
    CHUNK_SIZE,
    REF_DTYPE,
    DATA_DTYPE,
    DEVICE,
    chunk_gated_delta_rule_fwd_qla,
    chunk_gated_delta_rule_bwd_qla,
    chunk_gated_delta_rule_fwd_ref,
    chunk_gated_delta_rule_bwd_ref,
    _make_inputs,
    _assert_relative,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DETERMINISM_ITERS = 1000

# ---------------------------------------------------------------------------
# Shape configurations (inlined from settings/*.csv)
# ---------------------------------------------------------------------------

# (B, T, Hk, Hv, varlen, cu_seqlens | None)
CORE_CONFIGS = [
    pytest.param(1, 4096, 4, 4, False, None, id="B1-T4096-H4"),
    pytest.param(1, 4096, 2, 8, False, None, id="B1-T4096-H8G2"),
    pytest.param(1, 4096, 4, 16, False, None, id="B1-T4096-H16G4"),
    pytest.param(3, 4096, 4, 4, True, None, id="B3-T4096-H4-varlen"),
    pytest.param(1, 4096, 16, 32, True,
                 [0, 410, 841, 1135, 2126, 2512, 4096],
                 id="B1-T4096-H32G16-varlen"),
    pytest.param(1, 1000, 16, 64, True,
                 [0, 211, 985],
                 id="B1-T4096-H64G16-padding"),
]

DEVELOP_CONFIGS = [
    pytest.param(1, 32768, 4, 4, False, None, id="dev-H4"),
    pytest.param(1, 32768, 8, 8, False, None, id="dev-H8"),
    pytest.param(1, 32768, 16, 16, False, None, id="dev-H16"),
]

VARLEN_CONFIGS = [
    pytest.param(11, 33, 4, 4, True, None, id="varlen-B11-T33"),
    pytest.param(7, 4321, 4, 4, True, None, id="varlen-B7-T4321"),
    pytest.param(3, 16789, 4, 4, True, None, id="varlen-B3-T16789-vl"),
    pytest.param(5, 8192, 4, 4, True, None, id="varlen-B5-T8192-vl"),
    pytest.param(10, 1024, 4, 4, True, None, id="varlen-B10-T1024-vl"),
    pytest.param(20, 512, 4, 4, True, None, id="varlen-B20-T512-vl"),
]

PRODUCT_CONFIGS = [
    pytest.param(1, 128, 4, 4, True,
                 [0, 47, 128],
                 id="prod-mixed-small"),
    pytest.param(1, 4096, 4, 4, True,
                 [0, 517, 883, 1010, 3767, 4096],
                 id="prod-mixed-segs"),
    pytest.param(1, 16384, 4, 4, True,
                 [0, 4096, 6893, 7665, 8192, 12288, 16384],
                 id="prod-mixed-large"),
    pytest.param(1, 256, 4, 4, True,
                 [0, 73, 115, 209],
                 id="prod-mixed-pad"),
]

ALL_CONFIGS = CORE_CONFIGS + DEVELOP_CONFIGS + VARLEN_CONFIGS + PRODUCT_CONFIGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
             state_v_first, auto_cp):
    # Reference forward (float64)
    g_ref, o_ref, A_ref, h_ref, s_ref = chunk_gated_delta_rule_fwd_ref(
        q=q.to(REF_DTYPE, copy=True),
        k=k.to(REF_DTYPE, copy=True),
        v=v.to(REF_DTYPE, copy=True),
        g=g.to(REF_DTYPE, copy=True),
        beta=beta.to(REF_DTYPE, copy=True),
        scale=scale,
        initial_state=h0_ref,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )

    # QLA forward
    g_qla, A_qla, o_qla, h_qla, s_qla, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=h0_qla,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        output_h=True,
        auto_cp=auto_cp,
        state_v_first=state_v_first,
    )

    return (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    )


def _run_bwd(q, k, v, g_ref, g_qla, beta, A_ref, A_qla, do,
             h0_ref, h0_qla, dht_ref, dht_qla, cu_seqlens, scale,
             state_v_first, auto_cp):
    # Reference backward (float64)
    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd_ref(
        q.to(REF_DTYPE, copy=True),
        k.to(REF_DTYPE, copy=True),
        v.to(REF_DTYPE, copy=True),
        g_ref,
        beta.to(REF_DTYPE, copy=True),
        A_ref.to(REF_DTYPE, copy=True),
        scale,
        h0_ref,
        do.to(REF_DTYPE, copy=True),
        dht_ref,
        cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )

    # QLA backward
    dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla, beta, A_qla, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, auto_cp,
    )

    return (
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla,
    )


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    CORE_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_fwd(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first, use_h0,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first, auto_cp=True)

    h_qla_cmp = h_qla.transpose(-1, -2) if state_v_first else h_qla
    s_qla_cmp = s_qla.transpose(-1, -2) if state_v_first else s_qla

    _assert_relative(o_qla, o_ref, "o_qla")
    _assert_relative(h_qla_cmp, h_ref, "h_qla")
    if h0_ref is not None:
        _assert_relative(s_qla_cmp, s_ref, "s_qla")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    DEVELOP_CONFIGS + VARLEN_CONFIGS + PRODUCT_CONFIGS,
)
def test_fwd_extended(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=False,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first=False, auto_cp=True)

    _assert_relative(o_qla, o_ref, "o_qla")
    _assert_relative(s_qla, s_ref, "s_qla")


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    CORE_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_bwd(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first, use_h0,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first, auto_cp=True)

    (
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla,
    ) = _run_bwd(q, k, v, g_ref, g_qla, beta, A_ref, A_qla, do,
                 h0_ref, h0_qla, dht_ref, dht_qla, cu_seqlens, scale,
                 state_v_first, auto_cp=True)

    _assert_relative(dq_qla, dq_ref, "dq_qla")
    _assert_relative(dk_qla, dk_ref, "dk_qla")
    _assert_relative(dv_qla, dv_ref, "dv_qla")
    _assert_relative(db_qla, db_ref, "db_qla")
    _assert_relative(dg_qla, dg_ref, "dg_qla")
    if dht_ref is not None:
        dh0_qla_cmp = (
            dh0_qla.transpose(-1, -2)
            if state_v_first else dh0_qla
        )
        _assert_relative(dh0_qla_cmp, dh0_ref, "dh0_qla")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    DEVELOP_CONFIGS + VARLEN_CONFIGS + PRODUCT_CONFIGS,
)
def test_bwd_extended(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=False,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first=False, auto_cp=True)

    (
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla,
    ) = _run_bwd(q, k, v, g_ref, g_qla, beta, A_ref, A_qla, do,
                 h0_ref, h0_qla, dht_ref, dht_qla, cu_seqlens, scale,
                 state_v_first=False, auto_cp=True)

    _assert_relative(dq_qla, dq_ref, "dq_qla")
    _assert_relative(dk_qla, dk_ref, "dk_qla")
    _assert_relative(dv_qla, dv_ref, "dv_qla")
    _assert_relative(db_qla, db_ref, "db_qla")
    _assert_relative(dg_qla, dg_ref, "dg_qla")
    if dht_ref is not None:
        _assert_relative(dh0_qla, dh0_ref, "dh0_qla")


# ---------------------------------------------------------------------------
# Deterministic tests (run kernel many times, ensure no flaky results)
# ---------------------------------------------------------------------------

DETERMINISM_CONFIGS = [
    pytest.param(1, 4096, 2, 8, False, None, id="fixed-H8G2"),
    pytest.param(3, 4096, 4, 4, True, None, id="varlen-H4"),
    pytest.param(32, 1024, 2, 4, False, None, id="varlen-H4G2"),
]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    DETERMINISM_CONFIGS,
)
def test_fwd_deterministic(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=False,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first=False, auto_cp=True)

    for i in range(DETERMINISM_ITERS):
        _, _, o_qla_i, _, s_qla_i, _ = chunk_gated_delta_rule_fwd_qla(
            q, k, v, g, beta, scale, h0_qla, cu_seqlens,
            True, False, True, False,
        )
        _assert_relative(o_qla_i, o_ref, f"o_qla iter {i}")
        _assert_relative(s_qla_i, s_ref, f"s_qla iter {i}")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    DETERMINISM_CONFIGS,
)
def test_bwd_deterministic(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list,
):
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=False,
    )

    (
        g_ref, o_ref, A_ref, h_ref, s_ref,
        g_qla, A_qla, o_qla, h_qla, s_qla,
    ) = _run_fwd(q, k, v, g, beta, scale, h0_ref, h0_qla, cu_seqlens,
                 state_v_first=False, auto_cp=True)

    (
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref,
        _, _, _, _, _, _,
    ) = _run_bwd(q, k, v, g_ref, g_qla, beta, A_ref, A_qla, do,
                 h0_ref, h0_qla, dht_ref, dht_qla, cu_seqlens, scale,
                 state_v_first=False, auto_cp=True)

    for i in range(DETERMINISM_ITERS):
        dq_i, dk_i, dv_i, db_i, dg_i, dh0_i = chunk_gated_delta_rule_bwd_qla(
            q, k, v, g_qla, beta, A_qla, do, dht_qla,
            scale, h0_qla, cu_seqlens, False, True,
        )
        _assert_relative(dq_i, dq_ref, f"dq iter {i}")
        _assert_relative(dk_i, dk_ref, f"dk iter {i}")
        _assert_relative(dv_i, dv_ref, f"dv iter {i}")
        _assert_relative(dg_i, dg_ref, f"dg iter {i}")
        _assert_relative(db_i, db_ref, f"db iter {i}")
        if dht_ref is not None:
            _assert_relative(dh0_i, dh0_ref, f"dh0 iter {i}")
