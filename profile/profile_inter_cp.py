# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""单层 inter-card CP 前向的 kernel profiling。

每个 baseline 是一个独立函数，内部用 ``with timer.mark('tag')`` 标注各步，
外层 ``timer.bench()`` 控制 warmup+rep 迭代并对各 tag 求均值。

用法（需多卡）::

    torchrun --nproc_per_node=2 profile/profile_inter_cp.py --seqlen 16384 --nvh 16
    torchrun --nproc_per_node=4 profile/profile_inter_cp.py --seqlen 32768 --nvh 8 --nkh 2
    torchrun --nproc_per_node=2 profile/profile_inter_cp.py --seqlen 16384 --nvh 16 --fla
    torchrun --nproc_per_node=2 profile/profile_inter_cp.py --set develop
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from cuda_timer import CudaTimer
from flash_qla import chunk_gated_delta_rule as chunk_gated_delta_rule_qla
from flash_qla.utils import l2norm
from flash_qla.ops.utils import chunk_local_cumsum
from flash_qla.ops.gated_delta_rule.chunk import kkt_solve, fused_gdr_fwd
from flash_qla.ops.gated_delta_rule.chunk.cp import (
    build_cp_context,
    inter_card_cp_prepare_hm,
    inter_card_cp_all_gather_hm,
    inter_card_cp_correct_initial_states,
)

CHUNK = 64
FLA_LATEST_PATH = os.environ.get("FLA_LATEST", "/cpfs02/user/cenxi.lx/code/fla-latest")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_inputs(
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int = 128,
    head_dim_v: int = 128,
    data_dtype: str = "bfloat16",
    swa_ratio: float = 0.75,
    random_seed: int = 42,
    cu_seqlens: list[int] | None = None,
):
    dev = f"cuda:{torch.cuda.current_device()}"
    dtype = getattr(torch, data_dtype)
    torch.manual_seed(random_seed)

    q = l2norm(torch.randn(1, num_tokens, num_k_heads, head_dim_k, device=dev, dtype=dtype))
    k = l2norm(torch.randn(1, num_tokens, num_k_heads, head_dim_k, device=dev, dtype=dtype))
    v = torch.randn(1, num_tokens, num_v_heads, head_dim_v, device=dev, dtype=dtype)
    g = F.logsigmoid(torch.randn(1, num_tokens, num_v_heads, device=dev, dtype=torch.float32)) / 16
    beta = torch.randn(1, num_tokens, num_v_heads, device=dev, dtype=torch.float32).sigmoid()

    swa = torch.zeros(num_v_heads, dtype=torch.bool, device=dev)
    swa[:math.ceil(swa_ratio * num_v_heads)] = 1
    swa = swa[torch.randperm(num_v_heads, device=dev)]
    g[:, :, ~swa] = 0.0

    cu_g = torch.tensor(cu_seqlens or [0, num_tokens], device=dev, dtype=torch.int32)
    scale = head_dim_k ** -0.5

    return dict(q=q, k=k, v=v, g=g, beta=beta, cu_g=cu_g, scale=scale)


def slice_inputs(inputs: dict, lo: int, hi: int):
    return dict(
        q=inputs["q"][:, lo:hi],
        k=inputs["k"][:, lo:hi],
        v=inputs["v"][:, lo:hi],
        g=inputs["g"][:, lo:hi],
        beta=inputs["beta"][:, lo:hi],
    )



# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_full_seq(timer: CudaTimer, inputs: dict):
    """整条单卡(T)：全局序列单卡前向。"""
    cu_g, scale = inputs["cu_g"], inputs["scale"]

    def run(t):
        with t.mark("full/cumsum"):
            g_c = chunk_local_cumsum(inputs["g"], cu_seqlens=cu_g, chunk_size=CHUNK)
        with t.mark("full/kkt_solve"):
            A = kkt_solve(inputs["k"], inputs["beta"], cu_seqlens=cu_g, chunk_size=CHUNK)
        with t.mark("full/fused_fwd"):
            fused_gdr_fwd(
                q=inputs["q"], k=inputs["k"], v=inputs["v"], a=A, g=g_c, b=inputs["beta"],
                scale=scale, initial_state=None, output_final_state=False, output_h=False,
                output_o=True, cu_seqlens=cu_g, cp_seq_map=None, raw_cu_seqlens=None,
            )

    timer.bench(run, timer)


def baseline_no_cp(timer: CudaTimer, local: dict, cu: torch.Tensor, scale: float):
    """同片非 CP：rank-local 切片单卡前向。"""

    def run(t):
        with t.mark("base/cumsum"):
            g_c = chunk_local_cumsum(local["g"], cu_seqlens=cu, chunk_size=CHUNK)
        with t.mark("base/kkt_solve"):
            A = kkt_solve(local["k"], local["beta"], cu_seqlens=cu, chunk_size=CHUNK)
        with t.mark("base/fused_fwd"):
            fused_gdr_fwd(
                q=local["q"], k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
                scale=scale, initial_state=None, output_final_state=False, output_h=False,
                output_o=True, cu_seqlens=cu, cp_seq_map=None, raw_cu_seqlens=None,
            )

    timer.bench(run, timer)


def baseline_fla_cp(timer: CudaTimer, local: dict, cu_g: torch.Tensor, scale: float):
    """fla 自己的 inter-card CP，逐阶段拆解。"""
    if FLA_LATEST_PATH and FLA_LATEST_PATH not in sys.path:
        sys.path.insert(0, FLA_LATEST_PATH)
    import fla as _fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as chunk_gdn_fla
    from fla.ops.cp import build_cp_context as build_cp_context_fla
    from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_fwd_h_pre_process
    from fla.ops.utils import chunk_local_cumsum as cls_fla
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o

    rank = dist.get_rank()
    if rank == 0:
        print(f"[FLA] using fla from: {_fla.__file__}")

    ctx_fla = build_cp_context_fla(cu_g, group=dist.group.WORLD)
    cuf = ctx_fla.cu_seqlens

    def run_e2e(t):
        with t.mark("fla/e2e"):
            chunk_gdn_fla(
                local["q"], local["k"], local["v"], local["g"], local["beta"],
                scale=scale, cp_context=ctx_fla,
            )

    timer.bench(run_e2e, timer)

    def run_stages(t):
        with t.mark("fla/cumsum"):
            chunk_indices = prepare_chunk_indices(cuf, CHUNK)
            g_c = cls_fla(local["g"], chunk_size=CHUNK, scale=RCP_LN2,
                          cu_seqlens=cuf, chunk_indices=chunk_indices)
        with t.mark("fla/intra_fwd"):
            w, u, A = chunk_gated_delta_rule_fwd_intra(
                k=local["k"], v=local["v"], g=g_c, beta=local["beta"],
                cu_seqlens=cuf, chunk_indices=chunk_indices, chunk_size=CHUNK,
            )
        with t.mark("fla/cp_preprocess"):
            init_state = chunk_gated_delta_rule_fwd_h_pre_process(
                k=local["k"], w=w, u=u, g=g_c,
                cu_seqlens=cuf, initial_state=None,
                context=ctx_fla, state_v_first=False, chunk_size=CHUNK,
            )
        with t.mark("fla/fwd_h"):
            h, v_new, _ = chunk_gated_delta_rule_fwd_h(
                k=local["k"], w=w, u=u, g=g_c,
                initial_state=init_state, output_final_state=False,
                cu_seqlens=cuf, chunk_indices=chunk_indices,
                state_v_first=False, chunk_size=CHUNK,
            )
        with t.mark("fla/fwd_o"):
            chunk_fwd_o(
                q=local["q"], k=local["k"], v=v_new, h=h, g=g_c,
                scale=scale, cu_seqlens=cuf, chunk_indices=chunk_indices,
                state_v_first=False, chunk_size=CHUNK,
            )

    timer.bench(run_stages, timer)


# ---------------------------------------------------------------------------
# CP forward stages
# ---------------------------------------------------------------------------

def cp_fwd_stages(timer: CudaTimer, local: dict, ctx, inputs: dict):
    """QLA inter-card CP 前向各阶段拆解。"""
    scale = inputs["scale"]
    cu = ctx.cu_seqlens
    Hv = local["v"].shape[2]
    K = local["k"].shape[3]
    V = local["v"].shape[3]
    cu_cpu = ctx.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1

    def run(t):
        with t.mark("cp/cumsum"):
            g_c = chunk_local_cumsum(local["g"], cu_seqlens=cu, chunk_size=CHUNK)
        with t.mark("cp/kkt_solve"):
            A = kkt_solve(local["k"], local["beta"], cu_seqlens=cu, chunk_size=CHUNK)
        with t.mark("cp/prepare_h"):
            S_ext_r, M_r = inter_card_cp_prepare_hm(
                k=local["k"], v=local["v"], a=A, g=g_c, beta=local["beta"],
                cp_context=ctx, state_v_first=False,
            )
            S_ext_r = S_ext_r.float()
            M_r = M_r.float()
        with t.mark("cp/all_gather"):
            S_neigh, M_neigh = inter_card_cp_all_gather_hm(
                S_ext_r=S_ext_r, M_r=M_r, V=V, group=ctx.group, cp_context=ctx,
            )
        with t.mark("cp/inter_scan"):
            if not ctx.is_first_rank:
                init_r = inter_card_cp_correct_initial_states(
                    S_neigh=S_neigh, M_neigh=M_neigh, Hv=Hv, K=K, V=V,
                )
                raw_h0 = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=local["k"].device)
                raw_h0[0] = init_r
            else:
                raw_h0 = None
        with t.mark("cp/main_fwd"):
            fused_gdr_fwd(
                q=local["q"], k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
                scale=scale, initial_state=raw_h0, output_final_state=False, output_h=False,
                output_o=True, cu_seqlens=cu, cp_seq_map=None, raw_cu_seqlens=None,
            )

    timer.bench(run, timer)

    def run_e2e(t):
        with t.mark("cp/e2e"):
            chunk_gated_delta_rule_qla(
                local["q"], local["k"], local["v"], local["g"], local["beta"],
                scale=scale, cp_context=ctx,
            )

    timer.bench(run_e2e, timer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def profile_one(
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    rank: int,
    W: int,
    timer: CudaTimer,
    use_fla: bool = False,
    data_dtype: str = "bfloat16",
    swa_ratio: float = 0.75,
    random_seed: int = 42,
    cu_seqlens: list[int] | None = None,
    **_extra,
):
    T = num_tokens
    assert T % W == 0, f"T={T} 必须能被 world_size={W} 整除"
    part = T // W
    lo, hi = rank * part, (rank + 1) * part

    inputs = generate_inputs(
        num_tokens=T, num_k_heads=num_k_heads, num_v_heads=num_v_heads,
        data_dtype=data_dtype, swa_ratio=swa_ratio,
        random_seed=random_seed, cu_seqlens=cu_seqlens,
    )
    local = slice_inputs(inputs, lo, hi)
    ctx = build_cp_context(inputs["cu_g"], group=dist.group.WORLD)

    if rank == 0:
        print(f"Shape: T={T} (per-rank {part}) W={W} Hk={num_k_heads} "
              f"Hv={num_v_heads} cu={cu_seqlens or [0, T]} swa={swa_ratio}")

    timer.reset()

    baseline_full_seq(timer, inputs)
    baseline_no_cp(timer, local, ctx.cu_seqlens, inputs["scale"])
    if use_fla:
        baseline_fla_cp(timer, local, inputs["cu_g"], inputs["scale"])
    cp_fwd_stages(timer, local, ctx, inputs)

    if rank == 0:
        import pandas as pd
        r = timer.report()
        NAN = float("nan")

        def _g(tag):
            return r.get(tag, NAN)

        rows = OrderedDict()
        if use_fla:
            for row in rows.values():
                row["fla_cp"] = NAN
        rows["cumsum"]       = {"full": _g("full/cumsum"),    "base": _g("base/cumsum"),    "qla_cp": _g("cp/cumsum")}
        rows["kkt_solve"]    = {"full": _g("full/kkt_solve"), "base": _g("base/kkt_solve"), "qla_cp": _g("cp/kkt_solve")}
        if use_fla:
            rows["cp_preprocess"] = {"full": NAN, "base": NAN, "qla_cp": NAN, "fla_cp": _g("fla/cp_preprocess")}
        rows["prepare_h"]    = {"full": NAN,                  "base": NAN,                  "qla_cp": _g("cp/prepare_h")}
        rows["all_gather"]   = {"full": NAN,                  "base": NAN,                  "qla_cp": _g("cp/all_gather")}
        rows["inter_scan"]   = {"full": NAN,                  "base": NAN,                  "qla_cp": _g("cp/inter_scan")}
        if use_fla:
            rows["fused_fwd"]    = {"full": _g("full/fused_fwd"), "base": _g("base/fused_fwd"), "qla_cp": _g("cp/main_fwd"), "fla_cp": _g("fla/intra_fwd") + _g("fla/fwd_h") + _g("fla/fwd_o")}
        else:
            rows["fused_fwd"]    = {"full": _g("full/fused_fwd"), "base": _g("base/fused_fwd"), "qla_cp": _g("cp/main_fwd")}

        if use_fla:
            rows["cumsum"]["fla_cp"] = _g("fla/cumsum")

        full_total = sum(r.get(f"full/{k}", 0) for k in ("cumsum", "kkt_solve", "fused_fwd"))
        base_total = sum(r.get(f"base/{k}", 0) for k in ("cumsum", "kkt_solve", "fused_fwd"))
        rows["TOTAL"] = {"full": full_total, "base": base_total, "qla_cp": _g("cp/e2e")}
        if use_fla:
            rows["TOTAL"]["fla_cp"] = _g("fla/e2e")

        col_labels = {
            "full": f"整条单卡(T={T})",
            "base": f"同片非CP(T/W={part})",
            "qla_cp": f"QLA CP(T/W={part})",
        }
        if use_fla:
            col_labels["fla_cp"] = f"FLA CP(T/W={part})"

        df = pd.DataFrame(rows).T.reindex(columns=col_labels.keys())
        df.columns = [col_labels[c] for c in df.columns]
        print(f"\n{'='*72}")
        print(df.round(4).to_string())
        print(f"{'='*72}")

        cp_e2e = r.get("cp/e2e", 0)
        print(f"CP tax vs no-CP:  {cp_e2e - base_total:.4f} ms ({cp_e2e / max(base_total, 1e-9):.2f}x)")
        print(f"CP vs full-seq:   {full_total / max(cp_e2e, 1e-9):.2f}x speedup")
        if use_fla:
            fla_e2e = r.get("fla/e2e", 0)
            print(f"QLA vs FLA CP:    {fla_e2e / max(cp_e2e, 1e-9):.2f}x (>1 = QLA faster)")


def main():
    parser = argparse.ArgumentParser(description="Profile 单层 inter-card CP 前向")
    parser.add_argument("--set", type=str, default=None,
                        help="Preset name (loads from profile/settings/{set}.csv)")
    parser.add_argument("--seqlen", "--num-tokens", type=int, default=16384, help="全局 T")
    parser.add_argument("--nvh", "--num-v-heads", type=int, default=16)
    parser.add_argument("--nkh", "--num-k-heads", type=int, default=0, help="0 = 同 nvh")
    parser.add_argument("--cu-seqlens", type=str, default=None,
                        help="全局 cu_seqlens，如 0-8192-16384（默认单条 [0,T]）")
    parser.add_argument("--data-dtype", type=str, default="bfloat16")
    parser.add_argument("--swa-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fla", action="store_true", help="同时 profile fla 的 inter-card CP")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    if args.nkh <= 0:
        args.nkh = args.nvh

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", dist.get_rank())))
    rank = dist.get_rank()
    W = dist.get_world_size()

    timer = CudaTimer(warmup=args.warmup, rep=args.rep)

    if args.set is not None:
        import pandas as pd
        preset = pd.read_csv(
            os.path.join(PROJECT_ROOT, "profile", "settings", f"{args.set}.csv")
        )
        for i, row in preset.iterrows():
            if rank == 0:
                print("-" * 72)
            torch.cuda.empty_cache()
            data = row.to_dict()
            if "cu_seqlens" in data and isinstance(data["cu_seqlens"], str):
                data["cu_seqlens"] = list(map(int, data["cu_seqlens"].split("-")))
            cfg = dict(
                num_tokens=int(data.get("num_tokens", args.seqlen)),
                num_k_heads=int(data.get("num_k_heads", args.nkh)),
                num_v_heads=int(data.get("num_v_heads", args.nvh)),
                data_dtype=str(data.get("data_dtype", args.data_dtype)),
                swa_ratio=float(data.get("swa_ratio", args.swa_ratio)),
                random_seed=int(data.get("random_seed", args.seed)),
                cu_seqlens=data.get("cu_seqlens"),
            )
            profile_one(rank=rank, W=W, timer=timer, use_fla=args.fla, **cfg)
    else:
        if rank == 0:
            print("-" * 72)
        cu = [int(x) for x in args.cu_seqlens.split("-")] if args.cu_seqlens else None
        profile_one(
            num_tokens=args.seqlen, num_k_heads=args.nkh, num_v_heads=args.nvh,
            rank=rank, W=W, timer=timer, use_fla=args.fla,
            data_dtype=args.data_dtype, swa_ratio=args.swa_ratio,
            random_seed=args.seed, cu_seqlens=cu,
        )

    if rank == 0:
        print("-" * 72)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
