# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""L3: intra-card CP features that do not depend on the world size.

Three groups, all single-GPU and all checked against the float64 reference
implementation (or against the same run with the feature turned off):

* **auto_cp heuristic** -- forced split vs heuristic-chosen split vs no split.
* **Forward CP cache** -- ``enable_fwd_cp_cache`` must not change any result, and the
  backward must be able to consume the cache instead of recomputing it.
* **Mixed CP control** -- forward and backward may independently enable CP.

These use the direct-call wrappers in ``gdr_common`` rather than the case matrix in
``cp_common`` because they need to drive the forward and backward halves separately.
The end-to-end mode matrix (every mode at every world size, through the public API) is
``test_cp_e2e.py``.

The ``force_intra_cp`` axis matters here: without it the heuristic decides, and on some
configurations it declines -- so a test that only ran ``auto_cp=True`` could silently
stop covering the intra path. Every parametrization below pins which path it expects.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE, _auto_intra_cp_context

from gdr_common import (
    chunk_gated_delta_rule_fwd_qla,
    chunk_gated_delta_rule_bwd_qla,
    chunk_gated_delta_rule_fwd_ref,
    chunk_gated_delta_rule_bwd_ref,
    _make_inputs,
    _assert_relative,
    REF_DTYPE,
)


# ===========================================================================
# auto_cp heuristic vs forced split, against the float64 reference
# ===========================================================================
_AUTO_CP_CONFIGS = [
    pytest.param(1, 16384, 4, 4, False, None, id="long-fixed"),
    pytest.param(1, 16384, 4, 4, True, [0, 4096, 8192, 12288, 16384], id="long-varlen"),
]


def _intra_enabled(k, v, cu_seqlens, *, is_train, force_intra_cp):
    """What the context builder decides for this input -- the same cached context the
    kernel driver will use, so the tests can assert on the path that actually ran.
    """
    ctx = _auto_intra_cp_context(
        k, v, cu_seqlens, CHUNK_SIZE, True, is_train, force_intra_cp=force_intra_cp,
    )
    return ctx.is_intra


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    _AUTO_CP_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("force_intra_cp", [False, True], ids=["heuristic", "forced"])
def test_fwd_auto_cp(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first, force_intra_cp,
):
    """Forward with CP (heuristic or forced) and without must both match the reference."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=state_v_first,
    )

    is_intra = _intra_enabled(k, v, cu_seqlens, is_train=True, force_intra_cp=force_intra_cp)
    if force_intra_cp:
        assert is_intra, "force_intra_cp=True must enable the intra split"

    _, _, o_cp, _, s_cp, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, True, state_v_first, force_intra_cp=force_intra_cp,
    )
    _, _, o_nocp, _, s_nocp, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, False, state_v_first,
    )

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
    s_cp_cmp = s_cp.transpose(-1, -2) if state_v_first else s_cp
    s_nocp_cmp = s_nocp.transpose(-1, -2) if state_v_first else s_nocp

    _assert_relative(o_cp, o_ref, f"o_cp_vs_ref(intra={is_intra})")
    _assert_relative(o_nocp, o_ref, "o_nocp_vs_ref")
    _assert_relative(s_cp_cmp, s_ref, f"s_cp_vs_ref(intra={is_intra})")
    _assert_relative(s_nocp_cmp, s_ref, "s_nocp_vs_ref")


@pytest.mark.gpu
@pytest.mark.needs_bwd
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    _AUTO_CP_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("force_intra_cp", [False, True], ids=["heuristic", "forced"])
def test_bwd_auto_cp(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first, force_intra_cp,
):
    """Backward with CP (heuristic or forced) and without must both match the reference."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=state_v_first,
    )

    is_intra = _intra_enabled(k, v, cu_seqlens, is_train=True, force_intra_cp=force_intra_cp)
    if force_intra_cp:
        assert is_intra, "force_intra_cp=True must enable the intra split"

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

    g_qla_cp, A_qla_cp, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, True, state_v_first, force_intra_cp=force_intra_cp,
    )
    g_qla_nocp, A_qla_nocp, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, False, state_v_first,
    )

    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd_ref(
        q.to(REF_DTYPE, copy=True),
        k.to(REF_DTYPE, copy=True),
        v.to(REF_DTYPE, copy=True),
        g_ref,
        beta.to(REF_DTYPE, copy=True),
        A_ref.to(REF_DTYPE, copy=True),
        scale, h0_ref,
        do.to(REF_DTYPE, copy=True),
        dht_ref, cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )

    dq_cp, dk_cp, dv_cp, db_cp, dg_cp, dh0_cp = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla_cp, beta, A_qla_cp, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, True,
        force_intra_cp=force_intra_cp,
    )
    dq_nocp, dk_nocp, dv_nocp, db_nocp, dg_nocp, dh0_nocp = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla_nocp, beta, A_qla_nocp, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, False,
    )

    for prefix, dq, dk, dv, db, dg, dh0 in [
        (f"cp(intra={is_intra})", dq_cp, dk_cp, dv_cp, db_cp, dg_cp, dh0_cp),
        ("nocp", dq_nocp, dk_nocp, dv_nocp, db_nocp, dg_nocp, dh0_nocp),
    ]:
        _assert_relative(dq, dq_ref, f"dq_{prefix}")
        _assert_relative(dk, dk_ref, f"dk_{prefix}")
        _assert_relative(dv, dv_ref, f"dv_{prefix}")
        _assert_relative(db, db_ref, f"db_{prefix}")
        _assert_relative(dg, dg_ref, f"dg_{prefix}")
        if dht_ref is not None:
            dh0_cmp = dh0.transpose(-1, -2) if state_v_first else dh0
            _assert_relative(dh0_cmp, dh0_ref, f"dh0_{prefix}")


@pytest.mark.gpu
@pytest.mark.parametrize("is_train", [False, True], ids=["infer", "train"])
def test_auto_cp_decision_can_decline(is_train):
    """A configuration the decision dislikes must come back with the split off.

    This is the guard rail for the class of bug where an unconditional override makes the
    threshold logic dead code: if this ever starts returning ``True``, either the
    thresholds moved or something is forcing CP on again.
    """
    dev = "cuda:0"
    T = 4 * CHUNK_SIZE
    k = torch.randn(1, T, 64, 128, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, 64, 128, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
    assert not _intra_enabled(k, v, cu, is_train=is_train, force_intra_cp=False), (
        "the decision enabled intra CP for 64 heads over 4 chunks"
    )
    assert _intra_enabled(k, v, cu, is_train=is_train, force_intra_cp=True)


# ===========================================================================
# CP cache (enable_fwd_cp_cache=True vs False)
# ===========================================================================
CP_CACHE_CONFIGS = [
    pytest.param(1, 32768, 4, 4, False, None, id="cp-cache-H4"),
    pytest.param(1, 32768, 8, 8, False, None, id="cp-cache-H8"),
    pytest.param(1, 32768, 16, 16, False, None, id="cp-cache-H16"),
]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    CP_CACHE_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_fwd_cp_cache(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first, use_h0,
):
    """Forward with enable_fwd_cp_cache should match forward without it."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
    )

    common = dict(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=True, state_v_first=state_v_first,
        force_intra_cp=True,
    )
    _, _, o_base, _, s_base, _ = chunk_gated_delta_rule_fwd_qla(
        **common, enable_fwd_cp_cache=False)
    _, _, o_cache, _, s_cache, _ = chunk_gated_delta_rule_fwd_qla(
        **common, enable_fwd_cp_cache=True)

    _assert_relative(o_cache, o_base, "o_cp_cache_vs_base")
    if use_h0:
        s_base_cmp = s_base.transpose(-1, -2) if state_v_first else s_base
        s_cache_cmp = s_cache.transpose(-1, -2) if state_v_first else s_cache
        _assert_relative(s_cache_cmp, s_base_cmp, "s_cp_cache_vs_base")


@pytest.mark.gpu
@pytest.mark.needs_bwd
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    CP_CACHE_CONFIGS,
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_h0", [False, True], ids=["no_h0", "h0"])
def test_bwd_cp_cache(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first, use_h0,
):
    """Backward reusing the forward cache should match backward recomputing it."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
    )

    common = dict(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=True, state_v_first=state_v_first,
        force_intra_cp=True,
    )
    g_qla, A_qla, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        **common, enable_fwd_cp_cache=False)
    g_qla_c, A_qla_c, _, _, _, cp_cache = chunk_gated_delta_rule_fwd_qla(
        **common, enable_fwd_cp_cache=True)
    assert cp_cache is not None, "enable_fwd_cp_cache=True produced no cache"

    dq_base, dk_base, dv_base, db_base, dg_base, dh0_base = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla, beta, A_qla, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, True, force_intra_cp=True,
    )
    dq_cache, dk_cache, dv_cache, db_cache, dg_cache, dh0_cache = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla_c, beta, A_qla_c, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, True, cp_cache=cp_cache,
        force_intra_cp=True,
    )

    _assert_relative(dq_cache, dq_base, "dq_cp_cache")
    _assert_relative(dk_cache, dk_base, "dk_cp_cache")
    _assert_relative(dv_cache, dv_base, "dv_cp_cache")
    _assert_relative(db_cache, db_base, "db_cp_cache")
    _assert_relative(dg_cache, dg_base, "dg_cp_cache")
    if dht_qla is not None:
        _assert_relative(dh0_cache, dh0_base, "dh0_cp_cache")


# ===========================================================================
# Mixed CP control (forward and backward disagree about CP)
# ===========================================================================
@pytest.mark.gpu
@pytest.mark.needs_bwd
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("fwd_cp", [False, True], ids=["fwd_no_cp", "fwd_cp"])
@pytest.mark.parametrize("bwd_cp", [False, True], ids=["bwd_no_cp", "bwd_cp"])
def test_mixed_cp_control(state_v_first, fwd_cp, bwd_cp):
    """Any forward/backward combination of CP on/off must still match the reference."""
    B, T, Hk, Hv = 1, 32768, 4, 4
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(B, T, Hk, Hv, False, None, True, state_v_first)

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
    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd_ref(
        q.to(REF_DTYPE, copy=True),
        k.to(REF_DTYPE, copy=True),
        v.to(REF_DTYPE, copy=True),
        g_ref,
        beta.to(REF_DTYPE, copy=True),
        A_ref.to(REF_DTYPE, copy=True),
        scale, h0_ref,
        do.to(REF_DTYPE, copy=True),
        dht_ref, cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )

    g_qla, A_qla, o, _, s, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, fwd_cp, state_v_first, force_intra_cp=fwd_cp,
    )
    dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla, beta, A_qla, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, bwd_cp, force_intra_cp=bwd_cp,
    )

    _assert_relative(o, o_ref, "o")
    _assert_relative(dq, dq_ref, "dq")
    _assert_relative(dk, dk_ref, "dk")
    _assert_relative(dv, dv_ref, "dv")
    _assert_relative(db, db_ref, "db")
    _assert_relative(dg, dg_ref, "dg")
    dh0_cmp = dh0.transpose(-1, -2) if state_v_first else dh0
    _assert_relative(dh0_cmp, dh0_ref, "dh0")
