# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""单层 inter-card CP 前向的 kernel profiling（对标 profile_gdr.py）。

分解 CP 前向各阶段耗时（pre_process 一趟 / all_gather / inter_scan / 主前向），并与
「同一本地切片、非 CP」基线对比出 CP 开销。无正确性检查（用 cp_2gpu_test.py 或 tests）。

计时用 barrier 同步的墙钟均值（`bench_dist`）——`torch.profiler` 包 NCCL 集合通信会
SIGABRT，且 `do_bench` 逐 iter 无 barrier 会把集合通信测出巨大方差。各 rank lockstep，
循环内 all_gather 自然同步，结果稳定可复现（含 CPU 启动开销）。

用法（需多卡）::

    torchrun --nproc_per_node=2 profile/profile_gdr_cp.py --seqlen 16384 --nvh 16
    torchrun --nproc_per_node=4 profile/profile_gdr_cp.py --seqlen 32768 --nvh 8 --nkh 2
    torchrun --nproc_per_node=2 profile/profile_gdr_cp.py --cu-seqlens 0-8192-16384
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flash_qla import chunk_gated_delta_rule as chunk_gated_delta_rule_qla
from flash_qla.utils import l2norm, profile as profile_torch
from flash_qla.ops.utils import chunk_local_cumsum
from flash_qla.ops.gated_delta_rule.chunk import kkt_solve, fused_gdr_h, fused_gdr_fwd
from flash_qla.ops.gated_delta_rule.chunk.cp import (
    build_cp_context, all_gather_into_tensor, pack_hm, unpack_hm, inter_scan,
    inter_card_cp_preprocess_fwd,
)

CHUNK = 64
# fla 自己的 CP 在 fla-latest（未安装；环境里安装的旧版 fla 无 fla.ops.cp）。
# --fla 时把它插到 sys.path 前面导入。可用环境变量 FLA_LATEST 覆盖路径。
FLA_LATEST_PATH = os.environ.get("FLA_LATEST", "/cpfs02/user/cenxi.lx/code/fla-latest")


def bench_dist(fn, warmup: int = 25, rep: int = 100) -> float:
    """barrier 同步窗口内测墙钟均值（本地/含集合通信皆稳）。

    do_bench 逐 iter 无 barrier，两 rank 漂移会把集合通信测出巨大方差；此处用 barrier
    圈定窗口、循环内各 all_gather 自然同步，墙钟/rep 稳定可复现。含 CPU 启动开销。
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) * 1e3 / rep


def profile_cp_forward(
    num_tokens: int, num_k_heads: int, num_v_heads: int,
    cu_seqlens: list[int] | None,
    head_dim_k: int = 128, head_dim_v: int = 128,
    data_dtype: str = "bfloat16", random_seed: int = 42, swa_ratio: float = 0.75,
    use_fla: bool = False, kernels: bool = False,
):
    rank = dist.get_rank()
    W = dist.get_world_size()
    dev = f"cuda:{torch.cuda.current_device()}"
    dtype = getattr(torch, data_dtype)
    scale = head_dim_k ** -0.5
    assert num_tokens % W == 0, f"T={num_tokens} 必须能被 world_size={W} 整除"

    torch.manual_seed(random_seed)  # 各 rank 同 seed → 同一全局序列
    q = l2norm(torch.randn(1, num_tokens, num_k_heads, head_dim_k, device=dev, dtype=dtype))
    k = l2norm(torch.randn(1, num_tokens, num_k_heads, head_dim_k, device=dev, dtype=dtype))
    v = torch.randn(1, num_tokens, num_v_heads, head_dim_v, device=dev, dtype=dtype)
    g = F.logsigmoid(torch.randn(1, num_tokens, num_v_heads, device=dev, dtype=torch.float32)) / 16
    beta = torch.randn(1, num_tokens, num_v_heads, device=dev, dtype=torch.float32).sigmoid()
    swa = torch.zeros(num_v_heads, dtype=torch.bool, device=dev)
    swa[:math.ceil(swa_ratio * num_v_heads)] = 1
    swa = swa[torch.randperm(num_v_heads, device=dev)]
    g[:, :, ~swa] = 0.0

    cu_global = cu_seqlens if cu_seqlens is not None else [0, num_tokens]
    cu_g = torch.tensor(cu_global, device=dev, dtype=torch.int32)
    part = num_tokens // W
    lo, hi = rank * part, (rank + 1) * part
    ctx = build_cp_context(cu_g, group=dist.group.WORLD)
    ql, kl, vl, gl, bl = q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi]

    if rank == 0:
        print(f"Shape: T={num_tokens} (per-rank {part}) W={W} Hk={num_k_heads} "
              f"Hv={num_v_heads} cu={cu_global}")

    # ---- 端到端 total（走公共入口；barrier 同步墙钟，稳定）----
    #   cp: 分派到 CPChunkGatedDeltaRuleFunction；baseline: 同片非 CP（auto_cp=False）。
    #   输入不带 requires_grad，autograd 开销可忽略。
    t_cp = bench_dist(lambda: chunk_gated_delta_rule_qla(
        ql, kl, vl, gl, bl, scale=scale, cp_context=ctx))
    t_base = bench_dist(lambda: chunk_gated_delta_rule_qla(
        ql, kl, vl, gl, bl, scale=scale, cu_seqlens=ctx.cu_seqlens, auto_cp=False))

    # ---- 阶段拆解要用的中间量（g_c/A/warmup），提前算好（fla 段的 our_pre_wall 也要用）----
    g_c = chunk_local_cumsum(gl, cu_seqlens=ctx.cu_seqlens, chunk_size=CHUNK)
    A = kkt_solve(kl, bl, cu_seqlens=ctx.cu_seqlens, chunk_size=CHUNK)
    cu_cpu = ctx.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    nw = torch.empty((N, num_v_heads), dtype=ctx.cu_seqlens.dtype, device=dev)
    for i in range(N):
        nw[i, :] = (cu_cpu[i + 1] - cu_cpu[i] + CHUNK - 1) // CHUNK

    # ---- 可选：fla 自己的 inter-card CP（同一本地切片，公平对比）----
    # 注意：所有含集合通信的 bench_dist 必须都在 torch.profiler(--kernel) **之前**，
    # 否则 CUPTI 与 NCCL 交错会 SIGABRT。故 fla 的 pre_process/整体等墙钟量在此 Phase A 全测掉。
    t_fla_cp = t_fla_base = None
    fla_kern = None  # 存 fla 逐 kernel 所需的中间量与墙钟量，供 Phase B 打印
    if use_fla:
        if FLA_LATEST_PATH and FLA_LATEST_PATH not in sys.path:
            sys.path.insert(0, FLA_LATEST_PATH)  # 优先用 fla-latest（含 fla.ops.cp）
        import fla as _fla
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as chunk_gdn_fla
        from fla.ops.cp import build_cp_context as build_cp_context_fla
        if rank == 0:
            print(f"[FLA] using fla from: {_fla.__file__}")
        ctx_fla = build_cp_context_fla(cu_g, group=dist.group.WORLD)
        t_fla_cp = bench_dist(lambda: chunk_gdn_fla(
            ql, kl, vl, gl, bl, scale=scale, cp_context=ctx_fla))
        t_fla_base = bench_dist(lambda: chunk_gdn_fla(
            ql, kl, vl, gl, bl, scale=scale, cu_seqlens=ctx.cu_seqlens))

        if kernels:
            # 复刻 fla fwd 各 stage（Phase A：含集合通信的 pre_process 用 bench_dist 在此测掉）
            from fla.ops.utils import chunk_local_cumsum as cls_fla
            from fla.ops.utils.constant import RCP_LN2
            from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
            from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
            from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_fwd_h_pre_process
            cuf = ctx.cu_seqlens
            g_f = cls_fla(gl, chunk_size=CHUNK, scale=RCP_LN2, cu_seqlens=cuf)
            w_f, u_f, _A_f = chunk_gated_delta_rule_fwd_intra(k=kl, v=vl, g=g_f, beta=bl, cu_seqlens=cuf, chunk_size=CHUNK)
            init_f = chunk_gated_delta_rule_fwd_h_pre_process(
                k=kl, w=w_f, u=u_f, g=g_f, cu_seqlens=cuf, initial_state=None, context=ctx_fla, chunk_size=CHUNK)
            h_f, vnew_f, _ = chunk_gated_delta_rule_fwd_h(
                k=kl, w=w_f, u=u_f, g=g_f, initial_state=init_f, output_final_state=False, cu_seqlens=cuf, chunk_size=CHUNK)
            # 含集合通信：Phase A bench_dist
            f_pre = bench_dist(lambda: chunk_gated_delta_rule_fwd_h_pre_process(
                k=kl, w=w_f, u=u_f, g=g_f, cu_seqlens=cuf, initial_state=None, context=ctx_fla, chunk_size=CHUNK))
            our_pre_wall = bench_dist(lambda: inter_card_cp_preprocess_fwd(
                k=kl, v=vl, a=A, g=g_c, beta=bl, cp_context=ctx))
            fla_kern = dict(g_f=g_f, w_f=w_f, u_f=u_f, init_f=init_f, h_f=h_f, vnew_f=vnew_f,
                            f_pre=f_pre, our_pre_wall=our_pre_wall)

    def _pre():  # pre_process 聚合一趟（fused_gdr_h 全量）
        return fused_gdr_h(k=kl, v=vl, a=A, g=g_c, b=bl, initial_state=None,
                           output_final_state=True, output_h=False,
                           cu_seqlens=ctx.cu_seqlens, num_warmup_chunks=nw)

    _, ht, mt = _pre()
    hm = pack_hm(ht[-1].float(), mt[-1].float())

    # all_gather 调一次以准备 scan 输入；不单独 bench——do_bench 逐 iter 无 barrier，
    # 两 rank 漂移会把微小集合通信测成毫秒级假值。通信成本由 tax 推导（见下）。
    ag_hm, _ = all_gather_into_tensor(hm, group=dist.group.WORLD)
    pre = ctx.pre_num_ranks if not ctx.is_first_rank else 0
    neigh = ag_hm[rank - pre: rank] if pre > 0 else ag_hm[:0]

    def _scan():
        S_n, M_n = unpack_hm(neigh, head_dim_v)
        return inter_scan(M_n, S_n) if pre > 0 else None

    t_pre = bench_dist(_pre)
    t_scan = bench_dist(_scan)

    # 为主前向 kernel profile 备好真实 raw_h0（含 all_gather，须所有 rank 参与）
    if not ctx.is_first_rank:
        S_n, M_n = unpack_hm(neigh, head_dim_v)
        init_r = inter_scan(M_n, S_n)
        raw_h0 = torch.zeros((N, num_v_heads, head_dim_k, head_dim_v), dtype=torch.float32, device=dev)
        raw_h0[0] = init_r
    else:
        raw_h0 = None

    if rank != 0:
        return

    tax = t_cp - t_base
    t_comm = tax - t_pre - t_scan  # all_gather + 额外 kernel 启动开销（墙钟残差）
    df = pd.DataFrame({
        "ms (wall, barrier-synced)": {
            "total (baseline, same slice, no cp)": t_base,
            "total (cp, per-rank)": t_cp,
            "  ├ pre_process (fused_gdr_h+M)": t_pre,
            "  ├ all_gather + launch (derived)": t_comm,
            "  └ inter_scan": t_scan,
            "CP tax (cp − baseline)": tax,
        }
    })
    print(df.round(4))
    print(f"CP overhead vs same-slice baseline: {t_cp / t_base:.2f}x  "
          f"(tax≈{tax:.3f}ms：pre_process 额外 h+M 递推 {t_pre:.3f}ms + 通信/启动 {t_comm:.3f}ms)")

    if use_fla:
        print(f"[FLA] baseline(same slice) {t_fla_base:.4f}ms  cp {t_fla_cp:.4f}ms  "
              f"overhead {t_fla_cp / t_fla_base:.2f}x")
        print(f"[QLA vs FLA] cp total: QLA {t_cp:.4f}ms / FLA {t_fla_cp:.4f}ms "
              f"→ {t_fla_cp / t_cp:.2f}x（>1 表示 QLA 更快）；"
              f"baseline: QLA {t_base:.4f} / FLA {t_fla_base:.4f}")

    if kernels:
        # 逐 kernel GPU-time（torch.profiler / CUDA event），三列对比：
        #   整条单卡(T) / 同片非 CP(T/W) / inter-card CP(T/W)。
        # 只对本地算子用 torch.profiler（包 all_gather 会 SIGABRT）；all_gather/scan 用墙钟。
        # rank0 单独跑（无集合通信）。每 stage 一个融合 tilelang kernel，profile 'total' 即其 GPU 时间。
        print("\n=== 逐 kernel（GPU time, CUDA-event, ms）：整条单卡(T) vs 同片非CP(T/W) vs inter-card CP(T/W) ===")
        NAN = float("nan")

        def _bench_op(fn):
            return profile_torch(fn, [])["total"]

        def _main(qh, kh, vh, ah, gh, bh, h0, cu):
            return fused_gdr_fwd(
                q=qh, k=kh, v=vh, a=ah, g=gh, b=bh, scale=scale,
                initial_state=h0, output_final_state=False, output_h=False,
                output_o=True, cu_seqlens=cu, cp_seq_map=None, raw_cu_seqlens=None)

        # --- 整条单卡(T)：全局张量、无 CP ---
        g_c_full = chunk_local_cumsum(g, cu_seqlens=cu_g, chunk_size=CHUNK)
        A_full = kkt_solve(k, beta, cu_seqlens=cu_g, chunk_size=CHUNK)
        f_cumsum = _bench_op(lambda: chunk_local_cumsum(g, cu_seqlens=cu_g, chunk_size=CHUNK))
        f_kkt = _bench_op(lambda: kkt_solve(k, beta, cu_seqlens=cu_g, chunk_size=CHUNK))
        f_main = _bench_op(lambda: _main(q, k, v, A_full, g_c_full, beta, None, cu_g))

        # --- 同片(T/W) 的本地 kernel（非 CP 与 CP 共享 cumsum/kkt/main）---
        p_cumsum = _bench_op(lambda: chunk_local_cumsum(gl, cu_seqlens=ctx.cu_seqlens, chunk_size=CHUNK))
        p_kkt = _bench_op(lambda: kkt_solve(kl, bl, cu_seqlens=ctx.cu_seqlens, chunk_size=CHUNK))
        p_pre = _bench_op(_pre)
        p_main_base = _bench_op(lambda: _main(ql, kl, vl, A, g_c, bl, None, ctx.cu_seqlens))
        p_main_cp = _bench_op(lambda: _main(ql, kl, vl, A, g_c, bl, raw_h0, ctx.cu_seqlens))

        full_col = {
            "cumsum": f_cumsum, "kkt_solve": f_kkt,
            "pre_process (prepare_h+M)": NAN, "main (fused_gdr_fwd)": f_main,
            "all_gather": NAN, "inter_scan": NAN,
        }
        full_col["TOTAL"] = f_cumsum + f_kkt + f_main
        base_col = {
            "cumsum": p_cumsum, "kkt_solve": p_kkt,
            "pre_process (prepare_h+M)": NAN, "main (fused_gdr_fwd)": p_main_base,
            "all_gather": NAN, "inter_scan": NAN,
        }
        base_col["TOTAL"] = p_cumsum + p_kkt + p_main_base
        cp_col = {
            "cumsum": p_cumsum, "kkt_solve": p_kkt,
            "pre_process (prepare_h+M)": p_pre, "main (fused_gdr_fwd)": p_main_cp,
            "all_gather": t_comm, "inter_scan": t_scan,
        }
        cp_col["TOTAL"] = p_cumsum + p_kkt + p_pre + p_main_cp + t_comm + t_scan
        print(pd.DataFrame({
            f"整条单卡(T={num_tokens})": full_col,
            f"同片非CP(T/W={part})": base_col,
            "inter-card CP(T/W)": cp_col,
        }).round(4))
        print(f"forward halving：整条 main {f_main:.4f} → 每卡 main {p_main_base:.4f} "
              f"（≈1/{f_main / max(p_main_base, 1e-9):.1f}）")
        print(f"CP 墙钟(全序列, W 卡并行 = per-rank total) vs 整条单卡：{cp_col['TOTAL']:.4f} vs "
              f"{full_col['TOTAL']:.4f} → {full_col['TOTAL'] / cp_col['TOTAL']:.2f}x"
              f"（>1 表示 CP 更快；W={W} 下 pre_process 税可能吃掉 halving）")

        if use_fla:
            # fla 逐 kernel：本地 stage 用 torch.profiler；pre_process(含 all_gather) 与 our_pre_wall
            # 已在 Phase A 用 bench_dist 测过（存 fla_kern），此处**不再碰集合通信**（避免 CUPTI×NCCL SIGABRT）。
            from fla.ops.utils import chunk_local_cumsum as cls_fla
            from fla.ops.utils.constant import RCP_LN2
            from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
            from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
            from fla.ops.common.chunk_o import chunk_fwd_o
            cuf = ctx.cu_seqlens
            g_f, w_f, u_f = fla_kern["g_f"], fla_kern["w_f"], fla_kern["u_f"]
            init_f, h_f, vnew_f = fla_kern["init_f"], fla_kern["h_f"], fla_kern["vnew_f"]
            fla_cumsum = _bench_op(lambda: cls_fla(gl, chunk_size=CHUNK, scale=RCP_LN2, cu_seqlens=cuf))
            fla_intra = _bench_op(lambda: chunk_gated_delta_rule_fwd_intra(k=kl, v=vl, g=g_f, beta=bl, cu_seqlens=cuf, chunk_size=CHUNK))
            fla_fwdh = _bench_op(lambda: chunk_gated_delta_rule_fwd_h(
                k=kl, w=w_f, u=u_f, g=g_f, initial_state=init_f, output_final_state=False, cu_seqlens=cuf, chunk_size=CHUNK))
            fla_o = _bench_op(lambda: chunk_fwd_o(q=ql, k=kl, v=vnew_f, h=h_f, g=g_f, scale=scale, cu_seqlens=cuf, chunk_size=CHUNK))
            f_pre = fla_kern["f_pre"]
            our_pre_wall = fla_kern["our_pre_wall"]

            fla_rows = {
                "cumsum": fla_cumsum,
                "wy_intra (kkt+solve+wu)": fla_intra,
                "pre_process (S_ext+M+allgather+merge, wall)": f_pre,
                "fwd_h (main h)": fla_fwdh,
                "fwd_o (output)": fla_o,
            }
            fla_rows["TOTAL"] = sum(fla_rows.values())
            print("\n--- FLA 逐 kernel（ms；pre_process 为墙钟含通信，其余 CUDA-event GPU time）---")
            print(pd.DataFrame({"fla CP(T/W)": fla_rows}).round(4))
            print(f"[pre_process 对比, 墙钟含通信]  QLA {our_pre_wall:.4f}ms  vs  FLA {f_pre:.4f}ms  "
                  f"→ FLA {our_pre_wall / f_pre:.2f}x 更省（QLA 复用 fused_gdr_h 全量算 M；FLA 专用融合 kernel）")


def main():
    parser = argparse.ArgumentParser(description="Profile 单层 inter-card CP 前向")
    parser.add_argument("--seqlen", "--num-tokens", type=int, default=16384, help="全局 T")
    parser.add_argument("--nvh", "--num-v-heads", type=int, default=16)
    parser.add_argument("--nkh", "--num-k-heads", type=int, default=0, help="0 = 同 nvh")
    parser.add_argument("--cu-seqlens", type=str, default=None,
                        help="全局 cu_seqlens，如 0-8192-16384（默认单条 [0,T]）")
    parser.add_argument("--data-dtype", type=str, default="bfloat16")
    parser.add_argument("--swa-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fla", action="store_true", help="同时 profile fla 自己的 inter-card CP 做对比")
    parser.add_argument("--kernel", action="store_true", help="逐 kernel GPU-time 分解（torch.profiler）")
    args = parser.parse_args()
    if args.nkh <= 0:
        args.nkh = args.nvh
    cu = [int(x) for x in args.cu_seqlens.split("-")] if args.cu_seqlens else None

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", dist.get_rank())))
    if dist.get_rank() == 0:
        print("-" * 72)
    profile_cp_forward(
        num_tokens=args.seqlen, num_k_heads=args.nkh, num_v_heads=args.nvh,
        cu_seqlens=cu, data_dtype=args.data_dtype,
        swa_ratio=args.swa_ratio, random_seed=args.seed, use_fla=args.fla, kernels=args.kernel,
    )
    if dist.get_rank() == 0:
        print("-" * 72)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
