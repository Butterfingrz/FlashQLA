# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context

RTOL = 1e-2
SEED = 1234
DTYPE = torch.bfloat16
MASTER_PORT = "29531"

WORLD_SIZE = min(max(torch.cuda.device_count(), 2), 4)


def make_cases(W: int):
    base = 1024 * W
    # (cu_seqlens, Hk, Hv, g_scale, state_v_first, use_h0, use_dht)
    cases = [
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
    return cases


def _case_id(case) -> str:
    cu, Hk, Hv, gs, svf, use_h0, use_dht = case
    flags = "".join(f for f, on in [("vk", svf), ("h0", use_h0), ("dht", use_dht)] if on)
    flags = f"-{flags}" if flags else ""
    return f"n{len(cu) - 1}-Hk{Hk}Hv{Hv}-g{gs:g}{flags}"


CASES = make_cases(WORLD_SIZE)
_CASE_IDS = [_case_id(c) for c in CASES]


def run_case(cu_global, Hk, Hv, g_scale, state_v_first, use_h0, use_dht, dev, rank, W, seed=SEED):
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
    ctx = build_cp_context(cu_g, group=dist.group.WORLD)
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


def run_case_backward(cu_global, Hk, Hv, g_scale, state_v_first, use_h0, use_dht, dev, rank, W, seed=SEED):
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

    ctx = build_cp_context(cu_g, group=dist.group.WORLD)
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


def _init_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = MASTER_PORT
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _cp_worker(rank, world_size, case):
    try:
        _init_distributed(rank, world_size)
        dev = f"cuda:{rank}"
        o_ratio, s_ratio = run_case(*case, dev, rank, world_size)
        grad_ratio = run_case_backward(*case, dev, rank, world_size)
    finally:
        _cleanup_distributed()

    assert o_ratio <= RTOL and s_ratio <= RTOL and grad_ratio <= RTOL, (
        f"inter-CP mismatch (rank {rank}): case={case} "
        f"o_ratio={o_ratio:.2e} s_ratio={s_ratio:.2e} grad_ratio={grad_ratio:.2e} (rtol={RTOL})"
    )


def _run_cp_case(world_size, case):
    mp.start_processes(
        _cp_worker,
        args=(world_size, case),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_inter_cp(case):
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"inter-card CP test requires >= {WORLD_SIZE} GPUs")
    _run_cp_case(WORLD_SIZE, case)


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
    assert ctx.type == "inter"
    assert ctx.num_seqs >= 1
