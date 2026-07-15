# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""单层 inter-card CP 真多卡验证（独立脚本，不污染 tests）。

  torchrun --nproc_per_node=<N> test_gdr_inter_cp.py

支持任意 GPU 数（要求全局 token 总数能被 GPU 数整除）。
各 rank 用**同一 seed** 生成同一条全局序列，切自己的 token 片，用 build_cp_context 建上下文，
走 chunk_gated_delta_rule(cp_context=...)；输出与**单卡** auto_cp=False golden 的对应片对齐。
同时比较 output_final_state：末 rank 最后一个 local 序列的 final state 与 golden 的最后一个
全局序列的 final state 对齐。
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
    cases = [
        # (全局 cu_seqlens, Hk, Hv, g_scale)
        ([0, base], 8, 8, 1.0 / 16),
        ([0, base // 4, base * 3 // 4, base], 8, 8, 1.0 / 16),
        ([0, base], 8, 16, 1.0 / 16),
        ([0, base], 8, 8, 1.0),
        ([0, base // 4, base * 3 // 4, base], 8, 8, 1.0),
        ([0, base], 8, 16, 1.0),
        ([0, base * 2, base * 12, base * 16], 8, 8, 1.0 / 16),
    ]
    return cases


# =========================================


def run_case(cu_global, Hk, Hv, g_scale, dev, rank, W, seed=SEED):
    T = cu_global[-1]
    K = V = 128
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=DTYPE), p=2, dim=-1)
    k = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=DTYPE), p=2, dim=-1)
    v = torch.randn(1, T, Hv, V, device=dev, dtype=DTYPE)
    beta = torch.randn(1, T, Hv, device=dev, dtype=torch.float32).sigmoid()
    g = F.logsigmoid(torch.randn(1, T, Hv, device=dev, dtype=torch.float32)) * g_scale
    scale = K ** -0.5
    cu_g = torch.tensor(cu_global, device=dev, dtype=torch.int32)

    # 单卡 golden
    o_ref, final_state_ref = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale, cu_seqlens=cu_g,
        output_final_state=True, auto_cp=False)

    # CP：本 rank token 片 [lo,hi)
    part = T // W
    lo, hi = rank * part, (rank + 1) * part
    ctx = build_cp_context(cu_g, group=dist.group.WORLD)
    o_loc, final_state_loc = chunk_gated_delta_rule(
        q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi],
        scale=scale, cp_context=ctx, output_final_state=True)

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
        print(f"[{'PASS' if ok else 'FAIL'}] cu={cu_global} Hk/Hv={Hk}/{Hv} "
              f"g×{g_scale:<7} o_ratio={max_o_ratio:.2e} s_ratio={max_s_ratio:.2e}")
    return ok


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    W = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"

    if rank == 0:
        print("=" * 80)
        print(f"单层 inter-card CP 真 {W} 卡验证（vs 单卡 auto_cp=False golden，rtol={RTOL}）")
        print("=" * 80)

    cases = make_cases(W)
    results = []
    for cu_global, Hk, Hv, gs in cases:
        results.append(run_case(cu_global, Hk, Hv, gs, dev, rank, W))
    if rank == 0:
        n = sum(results)
        print("-" * 80)
        print(f"总计 {n}/{len(results)} 通过")
    dist.destroy_process_group()
    if rank == 0 and sum(results) != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
