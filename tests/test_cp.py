# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Context-parallel (CP) tests for the gated-delta-rule chunk kernel.

Consolidates every CP mode in one place:

* Multi-GPU distributed CP -- both **inter-card** and **inter+intra** -- share a
  single spawn harness parametrized by ``mode`` (see ``CP_MODES``). Each rank owns
  a slice of the global sequence and is compared against a single-GPU non-CP
  reference (forward output, final state, and gradients).
* Single-GPU **intra-card** CP -- the ``auto_cp`` / ``cp_cache`` path, compared
  against a float64 reference.
* CPU-only unit tests for the ``_calc_inter_cp_seqs`` sequence partitioner.
"""
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context
from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE

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
# Part A -- Multi-GPU distributed CP (inter-card and inter+intra)
# ===========================================================================

RTOL = 1e-2
SEED = 1234
DTYPE = torch.bfloat16

WORLD_SIZE = min(max(torch.cuda.device_count(), 2), 4)

# Per-card token counts for inter+intra mode. LONG makes each card's local
# sequence long enough to trigger the intra split on SM100 (>= 256 chunks);
# SHORT keeps intra disabled (degenerate: pure inter-card within a single card).
LONG = 16384
SHORT = 1024


def make_inter_cases(W: int):
    base = 1024 * W
    # (cu_seqlens, Hk, Hv, g_scale, state_v_first, use_h0, use_dht)
    return [
        ([0, base], 8, 8, 1.0 / 16, False, False, False),
        ([0, base // 4, base * 3 // 4, base], 8, 8, 1.0 / 16, False, False, False),
        ([0, base], 8, 16, 1.0 / 16, False, False, False),
        ([0, base], 8, 8, 1.0, False, False, False),
        ([0, base // 4, base * 3 // 4, base], 8, 8, 1.0, False, False, False),
        ([0, base], 8, 16, 1.0, False, False, False),
        ([0, base * 2, base * 12, base * 16], 8, 8, 1.0 / 16, False, False, False),
        # state_v_first=True
        ([0, base], 8, 8, 1.0 / 16, True, False, False),
        ([0, base], 8, 16, 1.0 / 16, True, False, False),
        # initial_state (h0)
        ([0, base], 8, 8, 1.0 / 16, False, True, False),
        ([0, base // 4, base * 3 // 4, base], 8, 8, 1.0 / 16, False, True, False),
        # dht (output_final_state + loss includes final_state)
        ([0, base], 8, 8, 1.0 / 16, False, False, True),
        ([0, base], 8, 16, 1.0 / 16, False, False, True),
        ([0, base // 4, base * 3 // 4, base], 8, 8, 1.0 / 16, False, False, True),
        # combined: state_v_first + h0 + dht
        ([0, base], 8, 8, 1.0 / 16, True, True, True),
        ([0, base // 2, base], 8, 8, 1.0 / 16, True, True, True),
    ]


def make_inter_intra_cases(W: int):
    L = LONG * W
    S = SHORT * W
    # (cu_seqlens, Hk, Hv, g_scale, state_v_first, use_h0, use_dht)
    return [
        ([0, L], 8, 8, 1.0 / 16, False, False, False),
        ([0, L], 8, 16, 1.0 / 16, False, False, False),
        ([0, L], 8, 8, 1.0 / 16, True, False, False),
        ([0, L], 8, 8, 1.0 / 16, False, True, False),
        ([0, L], 8, 8, 1.0 / 16, False, False, True),
        ([0, L], 8, 8, 1.0, False, False, False),          # heavy decay -> resets
        ([0, L // 2, L], 8, 8, 1.0 / 16, False, False, False),  # multi-seq, boundary spans cards
        ([0, L // 16, L], 8, 8, 1.0 / 16, False, False, False),
        ([0, L], 8, 8, 1.0 / 16, True, True, True),        # combined vf+h0+dht
        ([0, S], 8, 8, 1.0 / 16, False, False, False),     # short -> intra disabled
        ([0, L // 16, L // 8, L], 8, 8, 1.0 / 16, False, False, False),  # 3 seqs on card 0, middle is non-boundary
    ]


def _case_id(case) -> str:
    cu, Hk, Hv, gs, svf, use_h0, use_dht = case
    flags = "".join(f for f, on in [("vk", svf), ("h0", use_h0), ("dht", use_dht)] if on)
    flags = f"-{flags}" if flags else ""
    return f"n{len(cu) - 1}-T{cu[-1]}-Hk{Hk}Hv{Hv}-g{gs:g}{flags}"


# Per-mode registry: the only things that differ between inter-card and
# inter+intra CP are the context factory, the master port, and the case list.
CP_MODES = {
    "inter": dict(
        port="29531",
        make_ctx=lambda cu_g, Hv: build_cp_context(
            cu_g, group=dist.group.WORLD, enable_inter=True),
        cases=make_inter_cases(WORLD_SIZE),
    ),
    "inter_intra": dict(
        port="29533",
        make_ctx=lambda cu_g, Hv: build_cp_context(
            cu_g, group=dist.group.WORLD, num_v_heads=Hv, chunk_size=CHUNK_SIZE,
            enable_inter=True, enable_intra=True),
        cases=make_inter_intra_cases(WORLD_SIZE),
    ),
}


def run_case(mode, cu_global, Hk, Hv, g_scale, state_v_first, use_h0, use_dht,
             dev, rank, W, seed=SEED):
    T = cu_global[-1]
    K = V = 128
    N_seqs = len(cu_global) - 1
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=DTYPE), p=2, dim=-1)
    k = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=DTYPE), p=2, dim=-1)
    v = torch.randn(1, T, Hv, V, device=dev, dtype=DTYPE)
    beta = torch.randn(1, T, Hv, device=dev, dtype=torch.float32).sigmoid()
    g = F.logsigmoid(torch.randn(1, T, Hv, device=dev, dtype=torch.float32)) * g_scale
    scale = K ** -0.5
    cu_g = torch.tensor(cu_global, device=dev, dtype=torch.int32)

    if use_h0:
        if state_v_first:
            h0 = torch.randn(N_seqs, Hv, V, K, device=dev, dtype=torch.float32) * 0.01
        else:
            h0 = torch.randn(N_seqs, Hv, K, V, device=dev, dtype=torch.float32) * 0.01
    else:
        h0 = None

    o_ref, final_state_ref = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, cu_seqlens=cu_g,
        output_final_state=True, auto_cp=False, initial_state=h0,
        state_v_first=state_v_first)

    part = T // W
    lo, hi = rank * part, (rank + 1) * part
    ctx = CP_MODES[mode]["make_ctx"](cu_g, Hv)
    N_local = ctx.num_seqs
    if h0 is not None:
        start_seq = torch.searchsorted(cu_g[1:], lo, side="right").item()
        local_h0 = h0[start_seq: start_seq + N_local]
    else:
        local_h0 = None
    o_loc, final_state_loc = chunk_gated_delta_rule(
        q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi],
        scale=scale, cp_context=ctx, output_final_state=True, initial_state=local_h0,
        state_v_first=state_v_first)

    ref_slice = o_ref[:, lo:hi]
    err = (o_loc.float() - ref_slice.float()).abs().max().item()
    ref_norm = ref_slice.float().abs().max().item()
    o_ratio = err / (ref_norm + 1e-30)

    s_ratio = 0.0
    if rank == W - 1 and final_state_loc is not None and final_state_ref is not None:
        fs_loc = final_state_loc[-1].float()
        fs_ref = final_state_ref[-1].float()
        s_err = (fs_loc - fs_ref).abs().max().item()
        s_ref_norm = fs_ref.abs().max().item()
        s_ratio = s_err / (s_ref_norm + 1e-30)

    t = torch.tensor([o_ratio, s_ratio], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t[0].item(), t[1].item()


def _generate_inputs(T, Hk, Hv, K, V, g_scale, dev, seed):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=DTYPE), p=2, dim=-1)
    k = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=DTYPE), p=2, dim=-1)
    v = torch.randn(1, T, Hv, V, device=dev, dtype=DTYPE)
    beta = torch.randn(1, T, Hv, device=dev, dtype=torch.float32).sigmoid()
    g = F.logsigmoid(torch.randn(1, T, Hv, device=dev, dtype=torch.float32)) * g_scale
    return q, k, v, beta, g


def run_case_backward(mode, cu_global, Hk, Hv, g_scale, state_v_first, use_h0, use_dht,
                      dev, rank, W, seed=SEED):
    T = cu_global[-1]
    K = V = 128
    N_seqs = len(cu_global) - 1
    scale = K ** -0.5
    cu_g = torch.tensor(cu_global, device=dev, dtype=torch.int32)
    part = T // W
    lo, hi = rank * part, (rank + 1) * part

    if use_h0:
        torch.manual_seed(seed + 9999)
        if state_v_first:
            h0 = torch.randn(N_seqs, Hv, V, K, device=dev, dtype=torch.float32) * 0.01
        else:
            h0 = torch.randn(N_seqs, Hv, K, V, device=dev, dtype=torch.float32) * 0.01
    else:
        h0 = None

    # ---------- golden: non-cp backward ----------
    q_ref, k_ref, v_ref, beta_ref, g_ref = _generate_inputs(T, Hk, Hv, K, V, g_scale, dev, seed)
    q_ref.requires_grad_(True)
    k_ref.requires_grad_(True)
    v_ref.requires_grad_(True)
    beta_ref.requires_grad_(True)
    g_ref.requires_grad_(True)
    h0_ref = h0.clone().requires_grad_(True) if h0 is not None else None

    need_fs = use_dht or use_h0
    o_ref, fs_ref = chunk_gated_delta_rule(
        q_ref, k_ref, v_ref, g_ref, beta_ref, scale=scale,
        cu_seqlens=cu_g, output_final_state=need_fs, auto_cp=False,
        initial_state=h0_ref, state_v_first=state_v_first)
    loss_ref = o_ref.sum()
    if use_dht and fs_ref is not None:
        loss_ref = loss_ref + fs_ref.float().sum()
    loss_ref.backward()

    # ---------- CP backward ----------
    q_cp, k_cp, v_cp, beta_cp, g_cp = _generate_inputs(T, Hk, Hv, K, V, g_scale, dev, seed)
    q_cp = q_cp[:, lo:hi].clone().requires_grad_(True)
    k_cp = k_cp[:, lo:hi].clone().requires_grad_(True)
    v_cp = v_cp[:, lo:hi].clone().requires_grad_(True)
    beta_cp = beta_cp[:, lo:hi].clone().requires_grad_(True)
    g_cp = g_cp[:, lo:hi].clone().requires_grad_(True)

    ctx = CP_MODES[mode]["make_ctx"](cu_g, Hv)
    N_local = ctx.num_seqs
    if h0 is not None:
        start_seq = torch.searchsorted(cu_g[1:], lo, side="right").item()
        local_h0_cp = h0[start_seq: start_seq + N_local].clone().requires_grad_(True)
    else:
        local_h0_cp = None

    o_cp, fs_cp = chunk_gated_delta_rule(
        q_cp, k_cp, v_cp, g_cp, beta_cp,
        scale=scale, cp_context=ctx, output_final_state=need_fs,
        initial_state=local_h0_cp, state_v_first=state_v_first)
    loss_cp = o_cp.sum()
    if use_dht and fs_cp is not None:
        loss_cp = loss_cp + fs_cp.float().sum()
    loss_cp.backward()

    max_ratio = 0.0
    for name, ref_param, cp_param in [
        ("dq", q_ref, q_cp),
        ("dk", k_ref, k_cp),
        ("dv", v_ref, v_cp),
        ("dg", g_ref, g_cp),
        ("db", beta_ref, beta_cp),
    ]:
        ref_grad = ref_param.grad[:, lo:hi].float()
        cp_grad = cp_param.grad.float()
        err = (cp_grad - ref_grad).abs().max().item()
        ref_norm = ref_grad.abs().max().item()
        ratio = err / (ref_norm + 1e-30)
        max_ratio = max(max_ratio, ratio)

    if use_h0 and h0_ref is not None and local_h0_cp is not None:
        ref_grad = h0_ref.grad
        cp_grad = local_h0_cp.grad
        if ref_grad is not None and cp_grad is not None:
            start_seq = torch.searchsorted(cu_g[1:], lo, side="right").item()
            ref_slice = ref_grad[start_seq: start_seq + N_local].float()
            cp_slice = cp_grad.float()
            for si in range(N_local):
                seq_global = start_seq + si
                seq_start = cu_global[seq_global]
                if seq_start < lo:
                    continue
                err = (cp_slice[si] - ref_slice[si]).abs().max().item()
                ref_norm = ref_slice[si].abs().max().item()
                max_ratio = max(max_ratio, err / (ref_norm + 1e-30))

    t = torch.tensor([max_ratio], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t[0].item()


def _init_distributed(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _cp_worker(rank, world_size, mode, case):
    try:
        _init_distributed(rank, world_size, CP_MODES[mode]["port"])
        dev = f"cuda:{rank}"
        o_ratio, s_ratio = run_case(mode, *case, dev, rank, world_size)
        grad_ratio = run_case_backward(mode, *case, dev, rank, world_size)
    finally:
        _cleanup_distributed()

    assert o_ratio <= RTOL and s_ratio <= RTOL and grad_ratio <= RTOL, (
        f"{mode}-CP mismatch (rank {rank}): case={case} "
        f"o_ratio={o_ratio:.2e} s_ratio={s_ratio:.2e} grad_ratio={grad_ratio:.2e} (rtol={RTOL})"
    )


def _run_cp_case(world_size, mode, case):
    mp.start_processes(
        _cp_worker,
        args=(world_size, mode, case),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )


_PARAMS = [(mode, c) for mode, cfg in CP_MODES.items() for c in cfg["cases"]]
_IDS = [f"{mode}-{_case_id(c)}" for mode, c in _PARAMS]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("mode,case", _PARAMS, ids=_IDS)
def test_distributed_cp(mode, case):
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"distributed CP test requires >= {WORLD_SIZE} GPUs")
    _run_cp_case(WORLD_SIZE, mode, case)


# ===========================================================================
# Part B -- Single-GPU intra-card CP (auto_cp / cp_cache)
# ===========================================================================

@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    [
        pytest.param(1, 16384, 4, 4, False, None, id="long-fixed"),
        pytest.param(1, 16384, 4, 4, True,
                     [0, 4096, 8192, 12288, 16384],
                     id="long-varlen"),
    ],
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_fwd_auto_cp(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first,
):
    """auto_cp=True and auto_cp=False should produce equivalent results."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=state_v_first,
    )

    _, _, o_cp, _, s_cp, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, True, state_v_first,
    )
    _, _, o_nocp, _, s_nocp, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, False, state_v_first,
    )

    # Also check both against reference
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
    s_ref_cmp = s_ref
    s_cp_cmp = s_cp.transpose(-1, -2) if state_v_first else s_cp
    s_nocp_cmp = s_nocp.transpose(-1, -2) if state_v_first else s_nocp

    _assert_relative(o_cp, o_ref, "o_cp_vs_ref")
    _assert_relative(o_nocp, o_ref, "o_nocp_vs_ref")
    _assert_relative(s_cp_cmp, s_ref_cmp, "s_cp_vs_ref")
    _assert_relative(s_nocp_cmp, s_ref_cmp, "s_nocp_vs_ref")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch_size, num_tokens, num_k_heads, num_v_heads, varlen, cu_seqlens_list",
    [
        pytest.param(1, 16384, 4, 4, False, None, id="long-fixed"),
        pytest.param(1, 16384, 4, 4, True,
                     [0, 4096, 8192, 12288, 16384],
                     id="long-varlen"),
    ],
)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_bwd_auto_cp(
    batch_size, num_tokens, num_k_heads, num_v_heads,
    varlen, cu_seqlens_list, state_v_first,
):
    """Backward: auto_cp=True and auto_cp=False should match reference."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0=True, state_v_first=state_v_first,
    )

    # Run fwd for both cp modes
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
        True, False, True, state_v_first,
    )
    g_qla_nocp, A_qla_nocp, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, False, state_v_first,
    )

    # Ref bwd
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

    # QLA bwd with cp
    dq_cp, dk_cp, dv_cp, db_cp, dg_cp, dh0_cp = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla_cp, beta, A_qla_cp, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, True,
    )
    # QLA bwd without cp
    dq_nocp, dk_nocp, dv_nocp, db_nocp, dg_nocp, dh0_nocp = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla_nocp, beta, A_qla_nocp, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, False,
    )

    for prefix, dq, dk, dv, db, dg, dh0 in [
        ("cp", dq_cp, dk_cp, dv_cp, db_cp, dg_cp, dh0_cp),
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


# ---------------------------------------------------------------------------
# CP cache tests (enable_fwd_cp_cache=True vs False)
# ---------------------------------------------------------------------------

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

    _, _, o_base, _, s_base, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=True, state_v_first=state_v_first,
        enable_fwd_cp_cache=False,
    )
    _, _, o_cache, _, s_cache, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=True, state_v_first=state_v_first,
        enable_fwd_cp_cache=True,
    )

    _assert_relative(o_cache, o_base, "o_cp_cache_vs_base")
    if use_h0:
        s_base_cmp = s_base.transpose(-1, -2) if state_v_first else s_base
        s_cache_cmp = s_cache.transpose(-1, -2) if state_v_first else s_cache
        _assert_relative(s_cache_cmp, s_base_cmp, "s_cp_cache_vs_base")


@pytest.mark.gpu
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
    """Backward with cp_cache should match backward without it."""
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(
        batch_size, num_tokens, num_k_heads, num_v_heads,
        varlen, cu_seqlens_list, use_h0, state_v_first,
    )

    # Forward without cache (baseline)
    g_qla, A_qla, _, _, _, _ = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=True, state_v_first=state_v_first,
        enable_fwd_cp_cache=False,
    )
    # Forward with cache
    g_qla_c, A_qla_c, _, _, _, cp_cache = chunk_gated_delta_rule_fwd_qla(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0_qla, cu_seqlens=cu_seqlens,
        output_final_state=True, output_h=False,
        auto_cp=True, state_v_first=state_v_first,
        enable_fwd_cp_cache=True,
    )

    # Backward without cache
    dq_base, dk_base, dv_base, db_base, dg_base, dh0_base = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla, beta, A_qla, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, True,
    )
    # Backward with cache
    dq_cache, dk_cache, dv_cache, db_cache, dg_cache, dh0_cache = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla_c, beta, A_qla_c, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, True, cp_cache=cp_cache,
    )

    _assert_relative(dq_cache, dq_base, "dq_cp_cache")
    _assert_relative(dk_cache, dk_base, "dk_cp_cache")
    _assert_relative(dv_cache, dv_base, "dv_cp_cache")
    _assert_relative(db_cache, db_base, "db_cp_cache")
    _assert_relative(dg_cache, dg_base, "dg_cp_cache")
    if dht_qla is not None:
        _assert_relative(dh0_cache, dh0_base, "dh0_cp_cache")


# ---------------------------------------------------------------------------
# Mixed CP control tests (fwd and bwd use different auto_cp settings)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("fwd_cp", [False, True], ids=["fwd_no_cp", "fwd_cp"])
@pytest.mark.parametrize("bwd_cp", [False, True], ids=["bwd_no_cp", "bwd_cp"])
def test_mixed_cp_control(state_v_first, fwd_cp, bwd_cp):
    """Mixing auto_cp=True/False between fwd and bwd should still match reference."""
    B, T, Hk, Hv = 1, 32768, 4, 4
    (
        q, k, v, g, beta, do,
        h0_ref, dht_ref, h0_qla, dht_qla,
        cu_seqlens, scale,
    ) = _make_inputs(B, T, Hk, Hv, False, None, True, state_v_first)

    # Reference
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

    # Case 1: fwd auto_cp=True, bwd auto_cp=False
    g_qla, A_qla, o, _, s, _ = chunk_gated_delta_rule_fwd_qla(
        q, k, v, g, beta, scale, h0_qla, cu_seqlens,
        True, False, fwd_cp, state_v_first,
    )
    dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd_qla(
        q, k, v, g_qla, beta, A_qla, do, dht_qla,
        scale, h0_qla, cu_seqlens, state_v_first, bwd_cp,
    )

    _assert_relative(o, o_ref, "case1_o")
    _assert_relative(dq, dq_ref, "case1_dq")
    _assert_relative(dk, dk_ref, "case1_dk")
    _assert_relative(dv, dv_ref, "case1_dv")
    _assert_relative(db, db_ref, "case1_db")
    _assert_relative(dg, dg_ref, "case1_dg")
    dh0_cmp = dh0.transpose(-1, -2) if state_v_first else dh0
    _assert_relative(dh0_cmp, dh0_ref, "case1_dh0")


# ===========================================================================
# Part C -- CPU-only unit tests for the inter-card sequence partitioner
# ===========================================================================

@pytest.mark.parametrize("total, world_size", [(1000, 3), (1000, 7), (100, 3)])
def test_inter_cp_indivisible_raises(total, world_size):
    from flash_qla.ops.gated_delta_rule.chunk.cp import _calc_inter_cp_seqs

    assert total % world_size != 0, "test setup: total must be indivisible"
    cu = torch.tensor([0, total], dtype=torch.int32)
    with pytest.raises(AssertionError, match="divisible by"):
        _calc_inter_cp_seqs(cu, world_size=world_size, rank=0, group=None)


def test_inter_cp_divisible_ok():
    from flash_qla.ops.gated_delta_rule.chunk.cp import _calc_inter_cp_seqs

    cu = torch.tensor([0, 1000], dtype=torch.int32)  # 1000 % 4 == 0
    ctx = _calc_inter_cp_seqs(cu, world_size=4, rank=0, group=None)
    assert ctx.is_inter and not ctx.is_intra
    assert ctx.num_seqs >= 1
