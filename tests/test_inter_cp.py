# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""单层 inter-card CP 真多卡验证（独立脚本，不污染 tests）。

  torchrun --nproc_per_node=<N> test_gdr_inter_cp.py

支持任意 GPU 数（要求全局 token 总数能被 GPU 数整除）。
各 rank 用**同一 seed** 生成同一条全局序列，切自己的 token 片，用 build_cp_context 建上下文，
走 chunk_gated_delta_rule(cp_context=...)；输出与**单卡** auto_cp=False golden 的对应片对齐。
同时比较 output_final_state：末 rank 最后一个 local 序列的 final state 与 golden 的最后一个
全局序列的 final state 对齐。

后向验证：各梯度（dq, dk, dv, dg, db）的对应片与单卡 golden 的梯度对齐。
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context

# ===== 可配置（直接改这里做轻量调试）=====
RTOL = 1e-2
SEED = 1234
DTYPE = torch.bfloat16


def make_cases(W: int):
    """根据 world_size W 生成测试用例，保证全局 token 总数能被 W 整除。"""
    base = 1024 * W
    # (全局 cu_seqlens, Hk, Hv, g_scale, state_v_first, use_h0, use_dht)
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
        ([0, base // 2, base], 8, 8, 1.0 / 16, True, True, True)
    ]
    return cases


# =========================================


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

    # 单卡 golden
    o_ref, final_state_ref = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, cu_seqlens=cu_g,
        output_final_state=True, auto_cp=False, initial_state=h0,
        state_v_first=state_v_first)

    # CP：本 rank token 片 [lo,hi)
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

    # --- output 比较 ---
    ref_slice = o_ref[:, lo:hi]
    err = (o_loc.float() - ref_slice.float()).abs().max().item()
    ref_norm = ref_slice.float().abs().max().item()
    o_ratio = err / (ref_norm + 1e-30)

    # --- final state 比较（仅末 rank 最后一个 local 序列 vs golden 最后一个全局序列）---
    s_ratio = 0.0
    if rank == W - 1 and final_state_loc is not None and final_state_ref is not None:
        fs_loc = final_state_loc[-1].float()
        fs_ref = final_state_ref[-1].float()
        s_err = (fs_loc - fs_ref).abs().max().item()
        s_ref_norm = fs_ref.abs().max().item()
        s_ratio = s_err / (s_ref_norm + 1e-30)

    t = torch.tensor([o_ratio, s_ratio], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    max_o_ratio, max_s_ratio = t[0].item(), t[1].item()
    ok = max_o_ratio <= RTOL and max_s_ratio <= RTOL
    if rank == 0:
        flags = []
        if state_v_first: flags.append("vk")
        if use_h0: flags.append("h0")
        if use_dht: flags.append("dht")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        print(f"[{'PASS' if ok else 'FAIL'}] cu={cu_global} Hk/Hv={Hk}/{Hv} "
              f"g×{g_scale:<7} o_ratio={max_o_ratio:.2e} s_ratio={max_s_ratio:.2e}{flag_str}")
    return ok


def _generate_inputs(T, Hk, Hv, K, V, g_scale, dev, seed):
    """生成一组输入张量（detached，不 requires_grad）。"""
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

    # ---------- golden: 单卡 backward ----------
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

    # ---------- 比较各梯度 ----------
    max_ratio = 0.0
    worst_name = ""
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
        if ratio > max_ratio:
            max_ratio = ratio
            worst_name = name

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
                ratio = err / (ref_norm + 1e-30)
                if ratio > max_ratio:
                    max_ratio = ratio
                    worst_name = "dh0"

    t = torch.tensor([max_ratio], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    max_ratio_all = t[0].item()
    ok = max_ratio_all <= RTOL
    if rank == 0:
        flags = []
        if state_v_first: flags.append("vk")
        if use_h0: flags.append("h0")
        if use_dht: flags.append("dht")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        print(f"[{'PASS' if ok else 'FAIL'}] cu={cu_global} Hk/Hv={Hk}/{Hv} "
              f"g×{g_scale:<7} grad_ratio={max_ratio_all:.2e} worst={worst_name}{flag_str}")
    return ok


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    W = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"

    cases = make_cases(W)

    # Forward 测试
    if rank == 0:
        print("=" * 80)
        print(f"单层 inter-card CP 真 {W} 卡 前向 验证（vs 单卡 auto_cp=False golden，rtol={RTOL}）")
        print("=" * 80)

    fwd_results = []
    for cu_global, Hk, Hv, gs, svf, h0, dht in cases:
        fwd_results.append(run_case(cu_global, Hk, Hv, gs, svf, h0, dht, dev, rank, W))

    if rank == 0:
        n = sum(fwd_results)
        print(f"前向：{n}/{len(fwd_results)} 通过")

    # Backward 测试
    if rank == 0:
        print("=" * 80)
        print(f"单层 inter-card CP 真 {W} 卡 后向 验证（vs 单卡 auto_cp=False golden，rtol={RTOL}）")
        print("=" * 80)

    bwd_results = []
    for cu_global, Hk, Hv, gs, svf, h0, dht in cases:
        torch.cuda.empty_cache()
        bwd_results.append(run_case_backward(cu_global, Hk, Hv, gs, svf, h0, dht, dev, rank, W))

    if rank == 0:
        n_bwd = sum(bwd_results)
        print(f"后向：{n_bwd}/{len(bwd_results)} 通过")

    # 总结
    if rank == 0:
        print("-" * 80)
        total = sum(fwd_results) + sum(bwd_results)
        total_cases = len(fwd_results) + len(bwd_results)
        print(f"总计 {total}/{total_cases} 通过")

    dist.destroy_process_group()
    all_pass = all(fwd_results) and all(bwd_results)
    if rank == 0 and not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
