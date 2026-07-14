# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""极简单卡 micro-profile：隔离 prepare_h 里 M（calc_mt）计算的开销。

同一组输入分别喂：
  1. prepare_h(is_cp=True,  num_warmup=full) —— 跑 h 递推 + 算 M（calc_mt）
  2. prepare_h(is_cp=False, num_warmup=None) —— 只跑 h 递推，不算 M（关掉 is_cp 即可）
  3. fused_gdr_fwd                           —— 主前向（h 递推 + 输出 o）

M 的开销 ≈ (1) − (2)。单 GPU、无 dist、无 torchrun。

用法::
    python profile/profile_prepare_h.py --seqlen 8192 --nvh 16
    python profile/profile_prepare_h.py --seqlen 8192 --nvh 16 --nkh 2
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
import tilelang
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flash_qla.utils import l2norm
from flash_qla.ops.utils import chunk_local_cumsum
from flash_qla.ops.gated_delta_rule.chunk import kkt_solve, fused_gdr_h, fused_gdr_fwd

CHUNK = 64


def bench(fn, warmup=50, rep=200):
    return tilelang.profiler.do_bench(fn, warmup=warmup, rep=rep)


def main():
    p = argparse.ArgumentParser(description="micro-profile: M(calc_mt) 开销")
    p.add_argument("--seqlen", "--num-tokens", type=int, default=8192)
    p.add_argument("--nvh", "--num-v-heads", type=int, default=16)
    p.add_argument("--nkh", "--num-k-heads", type=int, default=0, help="0=同 nvh")
    p.add_argument("--data-dtype", type=str, default="bfloat16")
    p.add_argument("--swa-ratio", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    Hk = args.nkh if args.nkh > 0 else args.nvh
    Hv = args.nvh
    T = args.seqlen
    K = V = 128
    dtype = getattr(torch, args.data_dtype)
    dev = "cuda"
    scale = K ** -0.5

    torch.manual_seed(args.seed)
    q = l2norm(torch.randn(1, T, Hk, K, device=dev, dtype=dtype))
    k = l2norm(torch.randn(1, T, Hk, K, device=dev, dtype=dtype))
    v = torch.randn(1, T, Hv, V, device=dev, dtype=dtype)
    g = F.logsigmoid(torch.randn(1, T, Hv, device=dev, dtype=torch.float32)) / 16
    beta = torch.randn(1, T, Hv, device=dev, dtype=torch.float32).sigmoid()
    swa = torch.zeros(Hv, dtype=torch.bool, device=dev)
    swa[:math.ceil(args.swa_ratio * Hv)] = 1
    swa = swa[torch.randperm(Hv, device=dev)]
    g[:, :, ~swa] = 0.0

    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
    g_c = chunk_local_cumsum(g, cu_seqlens=cu, chunk_size=CHUNK)
    A = kkt_solve(k, beta, cu_seqlens=cu, chunk_size=CHUNK)
    nchunks = (T + CHUNK - 1) // CHUNK
    nw_full = torch.full((1, Hv), nchunks, device=dev, dtype=cu.dtype)

    def prepare_h_with_M():   # is_cp=True → 算 M
        return fused_gdr_h(k=k, v=v, a=A, g=g_c, b=beta, initial_state=None,
                           output_final_state=True, output_h=False,
                           cu_seqlens=cu, num_warmup_chunks=nw_full)

    def prepare_h_no_M():     # is_cp=False（num_warmup=None）→ 不算 M
        return fused_gdr_h(k=k, v=v, a=A, g=g_c, b=beta, initial_state=None,
                           output_final_state=True, output_h=False,
                           cu_seqlens=cu, num_warmup_chunks=None)

    def main_fwd():
        return fused_gdr_fwd(q=q, k=k, v=v, a=A, g=g_c, b=beta, scale=scale,
                             initial_state=None, output_final_state=False,
                             output_h=False, output_o=True, cu_seqlens=cu,
                             cp_seq_map=None, raw_cu_seqlens=None)

    t_withM = bench(prepare_h_with_M)
    t_noM = bench(prepare_h_no_M)
    t_main = bench(main_fwd)

    print("=" * 68)
    print(f"micro-profile  T={T} Hk={Hk} Hv={Hv} K=V={K} dtype={args.data_dtype}")
    print("=" * 68)
    rows = {
        "prepare_h (is_cp,  +M / calc_mt)": t_withM,
        "prepare_h (no is_cp, -M)": t_noM,
        "  └ M(calc_mt) 开销 = (+M) - (-M)": t_withM - t_noM,
        "fused_gdr_fwd (h + output o)": t_main,
    }
    print(pd.DataFrame({"ms": rows}).round(4))
    print(f"M 占 prepare_h(+M) 的 {100 * (t_withM - t_noM) / t_withM:.0f}%；"
          f"prepare_h(+M) / fused_gdr_fwd = {t_withM / t_main:.2f}x；"
          f"prepare_h(-M) / fused_gdr_fwd = {t_noM / t_main:.2f}x")


if __name__ == "__main__":
    main()
