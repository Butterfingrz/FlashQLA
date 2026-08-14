# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""单层 inter+intra CP 前向+后向的逐 kernel/step profiling。

对比三种模式（同一 rank-local 切片上）：
  - intra:      单卡内 intra CP（auto_cp=True，无 inter）
  - inter:      纯 inter-card CP（build_cp_context）
  - inter_intra:卡间 inter + 卡内 intra（build_cp_context enable_inter+enable_intra）

每个模式内部用 ``with timer.mark('tag')`` 标注各步，外层 ``timer.bench()``
控制 warmup+rep 迭代并对各 tag 求均值，最后按 tag 对齐打印对比表。重点观察
inter_intra 相对 intra/inter 新增的 aggregate 与两级 correct 的耗时占比。

用法（需多卡）::

    torchrun --nproc_per_node=2 profile/profile_inter_intra_cp.py --seqlen 65536 --nvh 16
    torchrun --nproc_per_node=4 profile/profile_inter_intra_cp.py --seqlen 131072 --nvh 8 --nkh 2
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

import torch
import torch.distributed as dist

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from cuda_timer import CudaTimer
from flash_qla import chunk_gated_delta_rule as chunk_gdr
from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector
from flash_qla.ops.gated_delta_rule.chunk import (
    kkt_solve, fused_gdr_fwd, fused_gdr_bwd, fused_gdr_h, fused_gdr_dh,
    correct_initial_states, correct_terminal_states, aggregate_card_state,
    get_warmup_chunks, get_warmup_chunks_bidi, CHUNK_SIZE,
)
from flash_qla.ops.gated_delta_rule.chunk.cp import (
    build_cp_context,
)
from flash_qla.ops.gated_delta_rule.chunk.cp.comm import (
    all_gather_into_tensor, pack_hm, unpack_hm,
)

# reuse the shared input generator / slicer from the inter profile
from profile_inter_cp import generate_inputs, slice_inputs

CHUNK = 64


def _ii_warmup_fallback(ctx, g_c, chunk_size=CHUNK):
    """inter+intra warmup/fallback (mirrors cp/preprocess.py): g-dependent bidi
    warmup with the first/last cp sub-chunks of each raw seq forced to full
    (they carry the inter boundary and must always be warmed up + corrected).
    Returns fwd/bwd warmup counts and fallback masks."""
    cp_cu = ctx.intra_cp_cu_seqlens
    seq_map_r2c = ctx.seq_map_r2c
    nwh, nwb, fbf, fbb = get_warmup_chunks_bidi(
        g=g_c, cu_seqlens=cp_cu, ht_mask_fwd=ctx.ht_mask,
        ht_mask_bwd=ctx.ht_mask_bwd, chunk_size=chunk_size,
    )
    first_end = seq_map_r2c[1]
    last_start = seq_map_r2c[-2]
    first_full = ((cp_cu[1:first_end + 1] - cp_cu[:first_end] + chunk_size - 1) // chunk_size).unsqueeze(-1)
    last_full = ((cp_cu[last_start + 1:] - cp_cu[last_start:-1] + chunk_size - 1) // chunk_size).unsqueeze(-1)
    for nw, fb in ((nwh, fbf), (nwb, fbb)):
        fb[:first_end] = True
        fb[last_start:] = True
        nw[:first_end] = first_full
        nw[last_start:] = last_full
    return nwh, nwb, fbf, fbb


# ---------------------------------------------------------------------------
# e2e baselines (per-rank local slice)
# ---------------------------------------------------------------------------
def _e2e(timer, tag, d, scale, cp_context, auto_cp, cu_seqlens=None):
    def fwd(t):
        with t.mark(f"{tag}/fwd_e2e"):
            chunk_gdr(d["q"], d["k"], d["v"], d["g"], d["beta"],
                      scale=scale, cp_context=cp_context, auto_cp=auto_cp, cu_seqlens=cu_seqlens)

    def bwd(t):
        q = d["q"].clone().requires_grad_(True)
        k = d["k"].clone().requires_grad_(True)
        v = d["v"].clone().requires_grad_(True)
        g = d["g"].clone().requires_grad_(True)
        b = d["beta"].clone().requires_grad_(True)
        # forward outside the timed region so bwd_e2e measures backward only
        o, _ = chunk_gdr(q, k, v, g, b, scale=scale, cp_context=cp_context,
                         auto_cp=auto_cp, cu_seqlens=cu_seqlens)
        loss = o.sum()
        with t.mark(f"{tag}/bwd_e2e"):
            loss.backward()

    timer.bench(fwd, timer)
    if fused_gdr_bwd is not None:
        timer.bench(bwd, timer)


# ---------------------------------------------------------------------------
# per-stage breakdown, shared tag scheme "{mode}::{stage}" so the three modes
# align in one table. Missing stages stay NaN.
# ---------------------------------------------------------------------------
def intra_stages(timer, full, scale, cu):
    """切分前单卡：整条序列在单卡上跑 intra CP（或 intra 未触发时的裸单卡）。"""
    ictx = build_cp_context(cu, enable_intra=True, num_v_heads=full["v"].shape[2], chunk_size=CHUNK_SIZE)
    use_intra = ictx.is_intra
    cp_cu = ictx.intra_cp_cu_seqlens if use_intra else cu

    def fwd(t):
        with t.mark("intra::cumsum"):
            g_c = chunk_local_cumsum(full["g"], cu_seqlens=cu, chunk_size=CHUNK)
        with t.mark("intra::kkt_solve"):
            A = kkt_solve(full["k"], full["beta"], cu_seqlens=cu, chunk_size=CHUNK)
        h0 = None
        if use_intra:
            with t.mark("intra::warmup"):
                nwh, fbf = get_warmup_chunks(g=g_c, cu_seqlens=cp_cu, ht_mask=ictx.ht_mask, chunk_size=CHUNK)
            with t.mark("intra::prepare_h"):
                _, ht, mt = fused_gdr_h(k=full["k"], v=full["v"], a=A, g=g_c, b=full["beta"],
                                        initial_state=None, output_final_state=True, output_h=False,
                                        cu_seqlens=cp_cu, num_warmup_chunks=nwh, state_v_first=False)
            with t.mark("intra::intra_correct"):
                h0 = correct_initial_states(raw_h0=None, ht_buffer=ht, mt_buffer=mt,
                                            fallback_mask=fbf, seq_map_r2c=ictx.seq_map_r2c, state_v_first=False)
        with t.mark("intra::main_fwd"):
            fused_gdr_fwd(q=full["q"], k=full["k"], v=full["v"], a=A, g=g_c, b=full["beta"],
                          scale=scale, initial_state=h0, output_final_state=False, output_h=False,
                          output_o=True, cu_seqlens=cp_cu,
                          cp_seq_map=ictx.seq_map_c2r if use_intra else None,
                          raw_cu_seqlens=cu if use_intra else None)

    timer.bench(fwd, timer)


def inter_stages(timer, local, ctx, scale, prefix="inter"):
    """切分后纯 inter-card CP，逐阶段。prefix 让退化的 inter_intra 也能复用。"""
    cu = ctx.cu_seqlens
    Hv = local["v"].shape[2]
    K, V = local["k"].shape[3], local["v"].shape[3]
    N = ctx.num_seqs
    cu_cpu = ctx.cu_seqlens_cpu.tolist()
    rank = dist.get_rank(group=ctx.group)
    pre = ctx.pre_num_ranks
    dev = local["k"].device

    def fwd(t):
        with t.mark(f"{prefix}::cumsum"):
            g_c = chunk_local_cumsum(local["g"], cu_seqlens=cu, chunk_size=CHUNK)
        with t.mark(f"{prefix}::kkt_solve"):
            A = kkt_solve(local["k"], local["beta"], cu_seqlens=cu, chunk_size=CHUNK)
        with t.mark(f"{prefix}::prepare_h"):
            num_warmup = torch.zeros((N, Hv), dtype=cu.dtype, device=dev)
            num_warmup[-1, :] = (cu_cpu[-1] - cu_cpu[-2] + CHUNK - 1) // CHUNK
            _, ht, mt = fused_gdr_h(k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
                                    initial_state=None, output_final_state=True, output_h=False,
                                    cu_seqlens=cu, num_warmup_chunks=num_warmup, state_v_first=False)
        with t.mark(f"{prefix}::all_gather"):
            hm = pack_hm(ht[-1].float(), mt[-1].float())
            ag_hm, _ = all_gather_into_tensor(hm, group=ctx.group)
            S_buf, M_buf = unpack_hm(ag_hm[rank - pre: rank + 1], V)
        with t.mark(f"{prefix}::inter_correct"):
            if not ctx.is_first_rank:
                sm, fb = ctx.get_fwd_scan_tensors(Hv)
                all_h0 = correct_initial_states(raw_h0=None, ht_buffer=S_buf, mt_buffer=M_buf,
                                                fallback_mask=fb, seq_map_r2c=sm, state_v_first=False)
                card_h0 = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=dev)
                card_h0[0] = all_h0[S_buf.shape[0] - 1]
            else:
                card_h0 = None
        with t.mark(f"{prefix}::main_fwd"):
            fused_gdr_fwd(q=local["q"], k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
                          scale=scale, initial_state=card_h0, output_final_state=False, output_h=False,
                          output_o=True, cu_seqlens=cu, cp_seq_map=None, raw_cu_seqlens=None)

    timer.bench(fwd, timer)


# ---------------------------------------------------------------------------
# inter_intra stage-by-stage (the new path)
# ---------------------------------------------------------------------------
def inter_intra_stages(timer, local, ctx, scale):
    """切分后 inter+intra 合并，逐阶段。intra 未触发时退化为 inter 拆解。"""
    if not ctx.is_intra:
        inter_stages(timer, local, ctx, scale, prefix="inter_intra")
        return
    cu = ctx.cu_seqlens
    Hv = local["v"].shape[2]
    K, V = local["k"].shape[3], local["v"].shape[3]
    N = ctx.num_seqs
    cp_cu = ctx.intra_cp_cu_seqlens
    seq_map_r2c = ctx.seq_map_r2c
    rank = dist.get_rank(group=ctx.group)
    pre = ctx.pre_num_ranks
    dev = local["k"].device

    def fwd(t):
        with t.mark("inter_intra::cumsum"):
            g_c = chunk_local_cumsum(local["g"], cu_seqlens=cu, chunk_size=CHUNK)
        with t.mark("inter_intra::kkt_solve"):
            A = kkt_solve(local["k"], local["beta"], cu_seqlens=cu, chunk_size=CHUNK)
        with t.mark("inter_intra::warmup"):
            nwh, _, fbf, _ = _ii_warmup_fallback(ctx, g_c)
        with t.mark("inter_intra::prepare_h"):
            _, ht, mt = fused_gdr_h(k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
                                    initial_state=None, output_final_state=True, output_h=False,
                                    cu_seqlens=cp_cu, num_warmup_chunks=nwh, state_v_first=False)
        with t.mark("inter_intra::aggregate"):
            S_card, M_card = aggregate_card_state(ht, mt, fbf, seq_map_r2c, state_v_first=False, compute_m=True)
        with t.mark("inter_intra::all_gather"):
            hm = pack_hm(S_card[-1], M_card[-1])
            ag_hm, _ = all_gather_into_tensor(hm, group=ctx.group)
            S_buf, M_buf = unpack_hm(ag_hm[rank - pre: rank + 1], V)
        with t.mark("inter_intra::inter_correct"):
            if not ctx.is_first_rank:
                sm, fb = ctx.get_fwd_scan_tensors(Hv)
                all_h0 = correct_initial_states(raw_h0=None, ht_buffer=S_buf, mt_buffer=M_buf,
                                                fallback_mask=fb, seq_map_r2c=sm, state_v_first=False)
                card_h0 = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=dev)
                card_h0[0] = all_h0[S_buf.shape[0] - 1]
            else:
                card_h0 = None
        with t.mark("inter_intra::intra_correct"):
            cp_h0 = correct_initial_states(raw_h0=card_h0, ht_buffer=ht, mt_buffer=mt,
                                           fallback_mask=fbf, seq_map_r2c=seq_map_r2c, state_v_first=False)
        with t.mark("inter_intra::main_fwd"):
            fused_gdr_fwd(q=local["q"], k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
                          scale=scale, initial_state=cp_h0, output_final_state=False, output_h=False,
                          output_o=True, cu_seqlens=cp_cu, cp_seq_map=ctx.seq_map_c2r, raw_cu_seqlens=cu)

    timer.bench(fwd, timer)


# ---------------------------------------------------------------------------
# backward per-stage breakdown (tags "{mode}::b_{stage}"), mirrors the forward
# structure. fwd artifacts (cp_h0/mt/M_card/o) are precomputed untimed.
# ---------------------------------------------------------------------------
def intra_stages_bwd(timer, full, scale, cu):
    if fused_gdr_bwd is None:
        return
    ictx = build_cp_context(cu, enable_intra=True, num_v_heads=full["v"].shape[2], chunk_size=CHUNK_SIZE, is_bwd=True)
    use_intra = ictx.is_intra
    cp_cu = ictx.intra_cp_cu_seqlens if use_intra else cu
    Hg, Hv = full["k"].shape[2], full["v"].shape[2]
    g_c = chunk_local_cumsum(full["g"], cu_seqlens=cu, chunk_size=CHUNK)
    A = kkt_solve(full["k"], full["beta"], cu_seqlens=cu, chunk_size=CHUNK)
    cp_h0 = None
    nwb = fbb = None
    if use_intra:
        nwh, nwb, fbf, fbb = get_warmup_chunks_bidi(g=g_c, cu_seqlens=cp_cu,
            ht_mask_fwd=ictx.ht_mask, ht_mask_bwd=ictx.ht_mask_bwd, chunk_size=CHUNK)
        _, ht, mt = fused_gdr_h(k=full["k"], v=full["v"], a=A, g=g_c, b=full["beta"],
            initial_state=None, output_final_state=True, output_h=False,
            cu_seqlens=cp_cu, num_warmup_chunks=nwh, state_v_first=False)
        cp_h0 = correct_initial_states(raw_h0=None, ht_buffer=ht, mt_buffer=mt,
            fallback_mask=fbf, seq_map_r2c=ictx.seq_map_r2c, state_v_first=False)
    o, _, _ = fused_gdr_fwd(q=full["q"], k=full["k"], v=full["v"], a=A, g=g_c, b=full["beta"],
        scale=scale, initial_state=cp_h0, output_final_state=False, output_h=False, output_o=True,
        cu_seqlens=cp_cu, cp_seq_map=ictx.seq_map_c2r if use_intra else None,
        raw_cu_seqlens=cu if use_intra else None)
    do = torch.ones_like(o)

    def bwd(t):
        dht = None
        if use_intra:
            with t.mark("intra::b_prepare_dh"):
                _, dh_buf = fused_gdr_dh(q=full["q"], k=full["k"], a=A, g=g_c, b=full["beta"], do=do,
                    dht=None, output_dh0=True, output_dh=False, scale=scale,
                    cu_seqlens=cp_cu, num_warmup_chunks=nwb, state_v_first=False)
            with t.mark("intra::b_intra_correct"):
                dht = correct_terminal_states(raw_dht=None, dht_buffer=dh_buf, mt_buffer=mt.float(),
                    fallback_mask=fbb, seq_map_r2c=ictx.seq_map_r2c, state_v_first=False)
        with t.mark("intra::b_recompute_h"):
            h, _, _ = fused_gdr_h(k=full["k"], v=full["v"], a=A, g=g_c, b=full["beta"],
                initial_state=cp_h0, output_final_state=False, output_h=True,
                cu_seqlens=cp_cu, state_v_first=False)
        with t.mark("intra::b_main_bwd"):
            dq, dk, dv, dg, db, _ = fused_gdr_bwd(q=full["q"], k=full["k"], v=full["v"], a=A, g=g_c,
                b=full["beta"], do=do, dht=dht, h=h, scale=scale, cu_seqlens=cp_cu, state_v_first=False)
        with t.mark("intra::b_postprocess"):
            if Hg < Hv:
                dq = group_reduce_vector(dq, Hg); dk = group_reduce_vector(dk, Hg)
            chunk_local_cumsum(dg, chunk_size=CHUNK, reverse=True, cu_seqlens=cu)

    timer.bench(bwd, timer)


def inter_stages_bwd(timer, local, ctx, scale, prefix="inter"):
    if fused_gdr_bwd is None:
        return
    cu = ctx.cu_seqlens
    Hv, Hg = local["v"].shape[2], local["k"].shape[2]
    K, V = local["k"].shape[3], local["v"].shape[3]
    N = ctx.num_seqs
    cu_cpu = ctx.cu_seqlens_cpu.tolist()
    rank = dist.get_rank(group=ctx.group)
    post = ctx.post_num_ranks
    dev = local["k"].device
    g_c = chunk_local_cumsum(local["g"], cu_seqlens=cu, chunk_size=CHUNK)
    A = kkt_solve(local["k"], local["beta"], cu_seqlens=cu, chunk_size=CHUNK)
    nw = torch.zeros((N, Hv), dtype=cu.dtype, device=dev)
    nw[-1, :] = (cu_cpu[-1] - cu_cpu[-2] + CHUNK - 1) // CHUNK
    nw[0, :] = (cu_cpu[1] - cu_cpu[0] + CHUNK - 1) // CHUNK
    _, ht, mt = fused_gdr_h(k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
        initial_state=None, output_final_state=True, output_h=False,
        cu_seqlens=cu, num_warmup_chunks=nw, state_v_first=False)
    M_first = mt[0].float()
    hm = pack_hm(ht[-1].float(), mt[-1].float())
    ag, _ = all_gather_into_tensor(hm, group=ctx.group)
    S_buf, M_buf = unpack_hm(ag[rank - ctx.pre_num_ranks: rank + 1], V)
    raw_h0 = None
    if not ctx.is_first_rank:
        sm, fb = ctx.get_fwd_scan_tensors(Hv)
        a0 = correct_initial_states(raw_h0=None, ht_buffer=S_buf, mt_buffer=M_buf,
            fallback_mask=fb, seq_map_r2c=sm, state_v_first=False)
        raw_h0 = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=dev); raw_h0[0] = a0[S_buf.shape[0] - 1]
    o, _, _ = fused_gdr_fwd(q=local["q"], k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
        scale=scale, initial_state=raw_h0, output_final_state=False, output_h=False, output_o=True,
        cu_seqlens=cu, cp_seq_map=None, raw_cu_seqlens=None)
    do = torch.ones_like(o)

    def bwd(t):
        with t.mark(f"{prefix}::b_prepare_dh"):
            nwb = torch.zeros((N, Hv), dtype=cu.dtype, device=dev)
            nwb[0, :] = (cu_cpu[1] - cu_cpu[0] + CHUNK - 1) // CHUNK
            _, dh0_buf = fused_gdr_dh(q=local["q"], k=local["k"], a=A, g=g_c, b=local["beta"], do=do,
                dht=None, output_dh0=True, output_dh=False, scale=scale,
                cu_seqlens=cu, num_warmup_chunks=nwb, state_v_first=False)
        with t.mark(f"{prefix}::b_all_gather"):
            hmb = pack_hm(dh0_buf[0].float(), M_first)
            agb, _ = all_gather_into_tensor(hmb, group=ctx.group)
            dS_buf, dM_buf = unpack_hm(agb[rank: rank + 1 + post], V)
        with t.mark(f"{prefix}::b_inter_correct"):
            corrected_dht = None
            if not ctx.is_last_rank:
                sm, fb = ctx.get_bwd_scan_tensors(Hv)
                cdt = correct_terminal_states(raw_dht=None, dht_buffer=dS_buf, mt_buffer=dM_buf,
                    fallback_mask=fb, seq_map_r2c=sm, state_v_first=False)
                corrected_dht = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=dev)
                corrected_dht[N - 1] = cdt[0]
        with t.mark(f"{prefix}::b_recompute_h"):
            h, _, _ = fused_gdr_h(k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
                initial_state=raw_h0, output_final_state=False, output_h=True,
                cu_seqlens=cu, state_v_first=False)
        with t.mark(f"{prefix}::b_main_bwd"):
            dq, dk, dv, dg, db, _ = fused_gdr_bwd(q=local["q"], k=local["k"], v=local["v"], a=A, g=g_c,
                b=local["beta"], do=do, dht=corrected_dht, h=h, scale=scale, cu_seqlens=cu, state_v_first=False)
        with t.mark(f"{prefix}::b_postprocess"):
            if Hg < Hv:
                dq = group_reduce_vector(dq, Hg); dk = group_reduce_vector(dk, Hg)
            chunk_local_cumsum(dg, chunk_size=CHUNK, reverse=True, cu_seqlens=cu)

    timer.bench(bwd, timer)


def inter_intra_stages_bwd(timer, local, ctx, scale):
    if fused_gdr_bwd is None:
        return
    if not ctx.is_intra:
        inter_stages_bwd(timer, local, ctx, scale, prefix="inter_intra")
        return
    cu = ctx.cu_seqlens
    Hv, Hg = local["v"].shape[2], local["k"].shape[2]
    K, V = local["k"].shape[3], local["v"].shape[3]
    N = ctx.num_seqs
    cp_cu = ctx.intra_cp_cu_seqlens
    seq_map_r2c = ctx.seq_map_r2c
    rank = dist.get_rank(group=ctx.group)
    pre, post = ctx.pre_num_ranks, ctx.post_num_ranks
    dev = local["k"].device
    g_c = chunk_local_cumsum(local["g"], cu_seqlens=cu, chunk_size=CHUNK)
    # fwd warmup/fallback set up the (untimed) fwd artifacts; bwd ones drive the timed bwd.
    nwh, nwb, fbf, fbb = _ii_warmup_fallback(ctx, g_c)
    A = kkt_solve(local["k"], local["beta"], cu_seqlens=cu, chunk_size=CHUNK)
    _, ht, mt = fused_gdr_h(k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
        initial_state=None, output_final_state=True, output_h=False,
        cu_seqlens=cp_cu, num_warmup_chunks=nwh, state_v_first=False)
    S_card, M_card = aggregate_card_state(ht, mt, fbf, seq_map_r2c, state_v_first=False, compute_m=True)
    hm = pack_hm(S_card[-1], M_card[-1]); ag, _ = all_gather_into_tensor(hm, group=ctx.group)
    S_buf, M_buf = unpack_hm(ag[rank - pre: rank + 1], V)
    card_h0 = None
    if not ctx.is_first_rank:
        sm, fb = ctx.get_fwd_scan_tensors(Hv)
        a0 = correct_initial_states(raw_h0=None, ht_buffer=S_buf, mt_buffer=M_buf,
            fallback_mask=fb, seq_map_r2c=sm, state_v_first=False)
        card_h0 = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=dev); card_h0[0] = a0[S_buf.shape[0] - 1]
    cp_h0 = correct_initial_states(raw_h0=card_h0, ht_buffer=ht, mt_buffer=mt,
        fallback_mask=fbf, seq_map_r2c=seq_map_r2c, state_v_first=False)
    o, _, _ = fused_gdr_fwd(q=local["q"], k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
        scale=scale, initial_state=cp_h0, output_final_state=False, output_h=False, output_o=True,
        cu_seqlens=cp_cu, cp_seq_map=ctx.seq_map_c2r, raw_cu_seqlens=cu)
    do = torch.ones_like(o)
    mtf = mt.float()

    def bwd(t):
        with t.mark("inter_intra::b_prepare_dh"):
            _, dh_buf = fused_gdr_dh(q=local["q"], k=local["k"], a=A, g=g_c, b=local["beta"], do=do,
                dht=None, output_dh0=True, output_dh=False, scale=scale,
                cu_seqlens=cp_cu, num_warmup_chunks=nwb, state_v_first=False)
        with t.mark("inter_intra::b_aggregate"):
            dS_card, _ = aggregate_card_state(dh_buf, mtf, fbb, seq_map_r2c, state_v_first=False,
                reverse=True, transpose_m=True, compute_m=False)
        with t.mark("inter_intra::b_all_gather"):
            hmb = pack_hm(dS_card[0], M_card[0]); agb, _ = all_gather_into_tensor(hmb, group=ctx.group)
            dS_buf, dM_buf = unpack_hm(agb[rank: rank + 1 + post], V)
        with t.mark("inter_intra::b_inter_correct"):
            card_dht = torch.zeros((N, Hv, K, V), dtype=torch.float32, device=dev)
            if not ctx.is_last_rank:
                sm, fb = ctx.get_bwd_scan_tensors(Hv)
                cdt = correct_terminal_states(raw_dht=None, dht_buffer=dS_buf, mt_buffer=dM_buf,
                    fallback_mask=fb, seq_map_r2c=sm, state_v_first=False)
                card_dht[N - 1] = cdt[0]
        with t.mark("inter_intra::b_intra_correct"):
            cp_dht = correct_terminal_states(raw_dht=card_dht, dht_buffer=dh_buf, mt_buffer=mtf,
                fallback_mask=fbb, seq_map_r2c=seq_map_r2c, state_v_first=False)
        with t.mark("inter_intra::b_recompute_h"):
            h, _, _ = fused_gdr_h(k=local["k"], v=local["v"], a=A, g=g_c, b=local["beta"],
                initial_state=cp_h0, output_final_state=False, output_h=True,
                cu_seqlens=cp_cu, state_v_first=False)
        with t.mark("inter_intra::b_main_bwd"):
            dq, dk, dv, dg, db, _ = fused_gdr_bwd(q=local["q"], k=local["k"], v=local["v"], a=A, g=g_c,
                b=local["beta"], do=do, dht=cp_dht, h=h, scale=scale, cu_seqlens=cp_cu, state_v_first=False)
        with t.mark("inter_intra::b_postprocess"):
            if Hg < Hv:
                dq = group_reduce_vector(dq, Hg); dk = group_reduce_vector(dk, Hg)
            chunk_local_cumsum(dg, chunk_size=CHUNK, reverse=True, cu_seqlens=cu)

    timer.bench(bwd, timer)


def main():
    p = argparse.ArgumentParser(description="Profile inter+intra CP 逐 kernel/step")
    p.add_argument("--seqlen", "--num-tokens", type=int, default=65536)
    p.add_argument("--nvh", "--num-v-heads", type=int, default=16)
    p.add_argument("--nkh", "--num-k-heads", type=int, default=0)
    p.add_argument("--cu-seqlens", type=str, default=None)
    p.add_argument("--data-dtype", type=str, default="bfloat16")
    p.add_argument("--swa-ratio", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rep", type=int, default=30)
    args = p.parse_args()
    if args.nkh <= 0:
        args.nkh = args.nvh

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", dist.get_rank())))
    rank, W = dist.get_rank(), dist.get_world_size()
    T = args.seqlen
    assert T % W == 0, f"T={T} must be divisible by world_size={W}"
    part = T // W
    lo, hi = rank * part, (rank + 1) * part

    cu = [int(x) for x in args.cu_seqlens.split("-")] if args.cu_seqlens else None
    inputs = generate_inputs(num_tokens=T, num_k_heads=args.nkh, num_v_heads=args.nvh,
                             data_dtype=args.data_dtype, swa_ratio=args.swa_ratio,
                             random_seed=args.seed, cu_seqlens=cu)
    local = slice_inputs(inputs, lo, hi)
    scale = inputs["scale"]

    ctx_inter = build_cp_context(inputs["cu_g"], group=dist.group.WORLD, enable_inter=True)
    ctx_ii = build_cp_context(inputs["cu_g"], group=dist.group.WORLD, num_v_heads=local["v"].shape[2],
                              chunk_size=CHUNK_SIZE, enable_inter=True, enable_intra=True)

    if rank == 0:
        print(f"T={T} W={W} per-rank={part} Hk={args.nkh} Hv={args.nvh} "
              f"ii.is_intra={ctx_ii.is_intra} cu={cu or [0, T]}")

    # intra baseline = 切分前单卡：整条全局序列在单卡上（full tensors, 不切分）
    full = {kk: inputs[kk] for kk in ("q", "k", "v", "g", "beta")}

    timer = CudaTimer(warmup=args.warmup, rep=args.rep)
    timer.reset()

    # e2e：intra 走整条单卡(T)；inter / inter_intra 走各卡切片(T/W)
    _e2e(timer, "intra", full, scale, cp_context=None, auto_cp=True, cu_seqlens=inputs["cu_g"])
    _e2e(timer, "inter", local, scale, cp_context=ctx_inter, auto_cp=False)
    _e2e(timer, "inter_intra", local, scale, cp_context=ctx_ii, auto_cp=False)

    # 逐阶段拆解（三模式都拆，便于横向比较）
    intra_stages(timer, full, scale, inputs["cu_g"])
    inter_stages(timer, local, ctx_inter, scale, prefix="inter")
    inter_intra_stages(timer, local, ctx_ii, scale)
    # backward 逐阶段
    intra_stages_bwd(timer, full, scale, inputs["cu_g"])
    inter_stages_bwd(timer, local, ctx_inter, scale, prefix="inter")
    inter_intra_stages_bwd(timer, local, ctx_ii, scale)

    if rank == 0:
        r = timer.report()
        NAN = float("nan")
        g = lambda tag: r.get(tag, NAN)

        try:
            import pandas as pd
        except ImportError:
            pd = None

        print(f"\n{'='*72}\nE2E (ms)  [intra=整条单卡T={T} | inter/inter_intra=切片T/W={part}]:")
        e2e_rows = OrderedDict()
        e2e_rows["fwd"] = {m: g(f"{m}/fwd_e2e") for m in ("intra", "inter", "inter_intra")}
        e2e_rows["bwd"] = {m: g(f"{m}/bwd_e2e") for m in ("intra", "inter", "inter_intra")}
        if pd is not None:
            print(pd.DataFrame(e2e_rows).T.round(4).to_string())
        else:
            for kk, vv in e2e_rows.items():
                print(kk, {a: round(b, 4) for a, b in vv.items()})

        # 逐阶段对齐表：行=阶段，列=三模式（不适用的阶段留 NaN）
        stages = ["cumsum", "kkt_solve", "warmup", "prepare_h", "aggregate",
                  "all_gather", "inter_correct", "intra_correct", "main_fwd"]
        modes = ["intra", "inter", "inter_intra"]
        rows = OrderedDict()
        for st in stages:
            rows[st] = {m: g(f"{m}::{st}") for m in modes}
        # 各模式 stage 求和作为 TOTAL（忽略 NaN）
        rows["TOTAL"] = {m: sum(v for v in (g(f"{m}::{st}") for st in stages) if v == v)
                         for m in modes}
        ii_mode = "intra ON" if ctx_ii.is_intra else "intra OFF→inter"
        print(f"\n{'='*72}\nForward 逐阶段 (ms)  [inter_intra: {ii_mode}]:")
        if pd is not None:
            df = pd.DataFrame(rows).T.reindex(columns=modes)
            print(df.round(4).to_string())
        else:
            for st, vv in rows.items():
                print(f"  {st:16s}", {a: round(b, 4) for a, b in vv.items()})

        if fused_gdr_bwd is not None:
            bstages = ["prepare_dh", "aggregate", "all_gather", "inter_correct",
                       "intra_correct", "recompute_h", "main_bwd", "postprocess"]
            brows = OrderedDict()
            for st in bstages:
                brows[st] = {m: g(f"{m}::b_{st}") for m in modes}
            brows["TOTAL"] = {m: sum(v for v in (g(f"{m}::b_{st}") for st in bstages) if v == v)
                              for m in modes}
            print(f"\n{'='*72}\nBackward 逐阶段 (ms)  [inter_intra: {ii_mode}]:")
            if pd is not None:
                print(pd.DataFrame(brows).T.reindex(columns=modes).round(4).to_string())
            else:
                for st, vv in brows.items():
                    print(f"  {st:16s}", {a: round(b, 4) for a, b in vv.items()})

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
