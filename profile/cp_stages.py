# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Stage-by-stage transcription of the CP forward/backward, for profiling.

One composition covers all four modes -- ``(is_inter, is_intra)`` on the context picks
which stages run -- so ``none`` / ``intra`` / ``inter`` / ``inter_intra`` are timed by
the *same* code path and their columns line up stage by stage:

=================  ====  =====  =====  ===========
stage              none  intra  inter  inter_intra
=================  ====  =====  =====  ===========
cumsum              x      x      x        x
kkt_solve           x      x      x        x
warmup                     x               x
prepare_h                  x      x        x
aggregate                         x*       x
all_gather                        x        x
inter_correct                     x        x
intra_correct              x               x
main_fwd            x      x      x        x
=================  ====  =====  =====  ===========

(``*`` pure inter has no aggregation to do; the stage is a no-op pass-through and is
timed as such so the column stays aligned.)

This is a **hand-written mirror** of :mod:`flash_qla.ops.gated_delta_rule.chunk.cp.preprocess`
plus the surrounding driver in ``chunk/__init__.py``, kept deliberately outside the
production code so profiling needs no hooks there. The cost of that choice is drift, so
``profile_cp.py`` ends every run with a parity step: it re-runs this composition
untimed and diffs the outputs against the real ``chunk_gated_delta_rule``. Keep the two
in sync -- when you change ``preprocess.py``, change this file and let the parity step
confirm it.

The ``region`` argument is what makes one composition serve three purposes: pass
:func:`timer_regions` to time each stage, :func:`nvtx_regions` to annotate an nsys
trace, or :func:`null_regions` for the parity re-run.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist

from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector
from flash_qla.ops.gated_delta_rule.chunk import (
    CHUNK_SIZE,
    kkt_solve,
    fused_gdr_fwd,
    fused_gdr_bwd,
    fused_gdr_h,
    fused_gdr_dh,
    correct_initial_states,
    correct_terminal_states,
    aggregate_card_state,
    get_warmup_chunks_bidi,
)
from flash_qla.ops.gated_delta_rule.chunk.cp.comm import (
    all_gather_into_tensor,
    pack_hm,
    unpack_hm,
)

#: Stage order in the printed tables. Also the set of tags a mode may emit.
FWD_STAGES = (
    "cumsum", "kkt_solve", "warmup", "prepare_h", "aggregate",
    "all_gather", "inter_correct", "intra_correct", "main_fwd",
)
BWD_STAGES = (
    "prepare_dh", "aggregate", "all_gather", "inter_correct",
    "intra_correct", "recompute_h", "main_bwd", "postprocess",
)

#: Backward tags are prefixed so the two directions never collide on a shared stage
#: name (``aggregate``, ``all_gather``, ``inter_correct``, ``intra_correct``).
BWD_TAG_PREFIX = "b_"


# ---------------------------------------------------------------------------
# Regions: how a stage boundary is recorded
# ---------------------------------------------------------------------------
def null_regions():
    """No instrumentation -- for the untimed parity re-run."""
    return lambda name: nullcontext()


def timer_regions(timer, mode: str, *, bwd: bool = False):
    """CUDA-event timing under the tag ``{mode}::{stage}`` (``{mode}::b_{stage}``)."""
    prefix = f"{mode}::{BWD_TAG_PREFIX}" if bwd else f"{mode}::"
    return lambda name: timer.mark(prefix + name)


def nvtx_regions(mode: str, *, bwd: bool = False):
    """NVTX ranges, for ``nsys``."""
    prefix = f"{mode}/{'bwd' if bwd else 'fwd'}/"

    @contextmanager
    def region(name):
        torch.cuda.nvtx.range_push(prefix + name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    return region


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
@dataclass
class FwdArtifacts:
    """Everything the backward needs, i.e. this file's stand-in for ``CPCache``."""

    g_c: torch.Tensor
    a: torch.Tensor
    o: torch.Tensor
    cp_h0: torch.Tensor | None = None
    mt: torch.Tensor | None = None            # per-chunk M (is_intra)
    m_first: torch.Tensor | None = None       # first local seq's M (is_inter)
    num_warmup_bwd: torch.Tensor | None = None
    fallback_bwd: torch.Tensor | None = None


def _inter_warmup_counts(ctx, num_v_heads: int, device, *, reverse: bool):
    """Pure-inter warmup counts: only the boundary sequences need a full warmup.

    Forward warms up the *last* local sequence (its terminal state leaves this card)
    and, when the card holds more than one sequence, the *first* one too (its M is what
    the backward gather needs). Backward only needs the first.
    """
    cu_cpu = ctx.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    counts = torch.zeros((N, num_v_heads), dtype=ctx.cu_seqlens.dtype, device=device)

    def full(i):
        return (cu_cpu[i + 1] - cu_cpu[i] + CHUNK_SIZE - 1) // CHUNK_SIZE

    if reverse:
        counts[0, :] = full(0)
    else:
        counts[-1, :] = full(N - 1)
        if N > 1:
            counts[0, :] = full(0)
    return counts


def _force_inter_boundaries_full(ctx, num_warmup, fallback, chunk_size=CHUNK_SIZE):
    """inter+intra: the first/last intra sub-chunks of each raw sequence carry the
    inter boundary, so they must always be warmed up and corrected regardless of what
    the decay-based heuristic decided (mirrors ``cp_preprocess_fwd``)."""
    cp_cu = ctx.intra_cp_cu_seqlens
    first_end = ctx.seq_map_r2c[1]
    last_start = ctx.seq_map_r2c[-2]
    first_full = (
        (cp_cu[1:first_end + 1] - cp_cu[:first_end] + chunk_size - 1) // chunk_size
    ).unsqueeze(-1)
    last_full = (
        (cp_cu[last_start + 1:] - cp_cu[last_start:-1] + chunk_size - 1) // chunk_size
    ).unsqueeze(-1)
    fallback[:first_end] = True
    fallback[last_start:] = True
    num_warmup[:first_end] = first_full
    num_warmup[last_start:] = last_full


def cp_fwd(
    d: dict,
    ctx,
    scale: float,
    region,
    *,
    initial_state: torch.Tensor | None = None,
    state_v_first: bool = False,
) -> FwdArtifacts:
    """One CP forward, stage by stage. ``d`` holds ``q/k/v/g/beta`` for this card."""
    q, k, v, g, beta = d["q"], d["k"], d["v"], d["g"], d["beta"]
    is_inter, is_intra = ctx.is_inter, ctx.is_intra
    Hv = v.shape[2]
    cu = ctx.cu_seqlens
    cp_cu = ctx.intra_cp_cu_seqlens if is_intra else cu

    with region("cumsum"):
        g_c = chunk_local_cumsum(g, cu_seqlens=cu, chunk_size=CHUNK_SIZE)
    with region("kkt_solve"):
        a = kkt_solve(k=k, b=beta, cu_seqlens=cu, chunk_size=CHUNK_SIZE)

    ht = mt = fallback = None
    num_warmup_bwd = fallback_bwd = m_first = None

    if is_intra:
        with region("warmup"):
            # The bidirectional variant costs one extra kernel but hands the backward
            # its warmup plan for free, which is why production caches it. The inter
            # boundary forcing is folded into the kernel (force_inter_boundaries),
            # mirroring production's cp_preprocess_fwd -- no host-side D2H/sync, so this
            # stage now reflects what the api path actually pays (was: a Python
            # _force_inter_boundaries_full doing per-boundary .item() reads).
            num_warmup, num_warmup_bwd, fallback, fallback_bwd = get_warmup_chunks_bidi(
                g=g_c, cu_seqlens=cp_cu, ht_mask_fwd=ctx.ht_mask,
                ht_mask_bwd=ctx.ht_mask_bwd, chunk_size=CHUNK_SIZE,
                seq_map_r2c=ctx.seq_map_r2c if is_inter else None,
                force_inter_boundaries=is_inter,
            )
        with region("prepare_h"):
            _, ht, mt = fused_gdr_h(
                k=k, v=v, a=a, g=g_c, b=beta, initial_state=None,
                output_final_state=True, output_h=False, cu_seqlens=cp_cu,
                num_warmup_chunks=num_warmup, state_v_first=state_v_first,
            )
    elif is_inter:
        with region("prepare_h"):
            num_warmup = _inter_warmup_counts(ctx, Hv, k.device, reverse=False)
            _, ht, mt = fused_gdr_h(
                k=k, v=v, a=a, g=g_c, b=beta, initial_state=initial_state,
                output_final_state=True, output_h=False, cu_seqlens=cu,
                num_warmup_chunks=num_warmup, state_v_first=state_v_first,
            )

    if is_inter:
        with region("aggregate"):
            if is_intra:
                # per-chunk (h, M) -> per-raw-sequence (h, M) for this card
                h_seq, m_seq = aggregate_card_state(
                    ht, mt, fallback, ctx.seq_map_r2c,
                    state_v_first=state_v_first, compute_m=True,
                )
            else:
                h_seq, m_seq = ht, mt
        m_first = m_seq[0]
        with region("all_gather"):
            hm = pack_hm(h_seq[-1], m_seq[-1])
            ag_hm, _ = all_gather_into_tensor(hm, group=ctx.group)
            rank = dist.get_rank(group=ctx.group)
            h_buf, m_buf = unpack_hm(
                ag_hm[rank - ctx.pre_num_ranks: rank + 1], h_seq[-1], m_seq[-1]
            )
        with region("inter_correct"):
            card_h0 = initial_state
            if not ctx.is_first_rank:
                seq_map_scan, fb_scan = ctx.get_fwd_scan_tensors(Hv)
                cp_h0_all = correct_initial_states(
                    raw_h0=None, ht_buffer=h_buf, mt_buffer=m_buf,
                    fallback_mask=fb_scan, seq_map_r2c=seq_map_scan,
                    state_v_first=state_v_first,
                )
                card_h0 = (
                    torch.zeros(h_seq.shape, dtype=torch.float32, device=h_seq.device)
                    if initial_state is None else initial_state.clone()
                )
                card_h0[0] = cp_h0_all[h_buf.shape[0] - 1]
    else:
        card_h0 = initial_state

    if is_intra:
        with region("intra_correct"):
            cp_h0 = correct_initial_states(
                raw_h0=card_h0, ht_buffer=ht, mt_buffer=mt,
                fallback_mask=fallback, seq_map_r2c=ctx.seq_map_r2c,
                state_v_first=state_v_first,
            )
    else:
        cp_h0 = card_h0

    with region("main_fwd"):
        o, _, _ = fused_gdr_fwd(
            q=q, k=k, v=v, a=a, g=g_c, b=beta, scale=scale,
            initial_state=cp_h0, output_final_state=False, output_h=False,
            output_o=True, cu_seqlens=cp_cu,
            cp_seq_map=ctx.seq_map_c2r, raw_cu_seqlens=cu,
            state_v_first=state_v_first,
        )

    return FwdArtifacts(
        g_c=g_c, a=a, o=o, cp_h0=cp_h0, mt=mt, m_first=m_first,
        num_warmup_bwd=num_warmup_bwd, fallback_bwd=fallback_bwd,
    )


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------
def cp_bwd(
    d: dict,
    ctx,
    scale: float,
    region,
    fwd: FwdArtifacts,
    *,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    state_v_first: bool = False,
) -> dict:
    """One CP backward, stage by stage, reusing ``fwd``'s artifacts as the cache."""
    q, k, v, beta = d["q"], d["k"], d["v"], d["beta"]
    g_c, a = fwd.g_c, fwd.a
    is_inter, is_intra = ctx.is_inter, ctx.is_intra
    Hg, Hv = k.shape[2], v.shape[2]
    cu = ctx.cu_seqlens
    cp_cu = ctx.intra_cp_cu_seqlens if is_intra else cu

    dh = None
    if is_intra:
        with region("prepare_dh"):
            _, dh = fused_gdr_dh(
                q=q, k=k, a=a, g=g_c, b=beta, do=do, dht=None,
                output_dh0=True, output_dh=False, scale=scale, cu_seqlens=cp_cu,
                num_warmup_chunks=fwd.num_warmup_bwd, state_v_first=state_v_first,
            )
    elif is_inter:
        with region("prepare_dh"):
            num_warmup = _inter_warmup_counts(ctx, Hv, k.device, reverse=True)
            _, dh = fused_gdr_dh(
                q=q, k=k, a=a, g=g_c, b=beta, do=do, dht=dht,
                output_dh0=True, output_dh=False, scale=scale, cu_seqlens=cu,
                num_warmup_chunks=num_warmup, state_v_first=state_v_first,
            )

    if is_inter:
        with region("aggregate"):
            if is_intra:
                # reverse scan: per-chunk dh -> per-raw-sequence dh. The M product is
                # the forward one (cached), so compute_m stays off.
                dh_seq, _ = aggregate_card_state(
                    dh, fwd.mt, fwd.fallback_bwd, ctx.seq_map_r2c,
                    state_v_first=state_v_first, reverse=True, transpose_m=True,
                    compute_m=False,
                )
            else:
                dh_seq = dh
        N = dh_seq.shape[0]
        with region("all_gather"):
            hm = pack_hm(dh_seq[0], fwd.m_first)
            ag_hm, _ = all_gather_into_tensor(hm, group=ctx.group)
            rank = dist.get_rank(group=ctx.group)
            dh_buf, m_buf = unpack_hm(
                ag_hm[rank: rank + 1 + ctx.post_num_ranks], dh_seq[0], fwd.m_first
            )
        with region("inter_correct"):
            if dht is None and ctx.is_last_rank:
                card_dht = None
            else:
                card_dht = (
                    torch.zeros(dh_seq.shape, dtype=torch.float32, device=dh_seq.device)
                    if dht is None else dht.clone()
                )
                if not ctx.is_last_rank:
                    seq_map_scan, fb_scan = ctx.get_bwd_scan_tensors(Hv)
                    cp_dht_all = correct_terminal_states(
                        raw_dht=None, dht_buffer=dh_buf, mt_buffer=m_buf,
                        fallback_mask=fb_scan, seq_map_r2c=seq_map_scan,
                        state_v_first=state_v_first,
                    )
                    card_dht[N - 1] = cp_dht_all[0]
    else:
        card_dht = dht

    if is_intra:
        with region("intra_correct"):
            cp_dht = correct_terminal_states(
                # No `.float()`: production (`preprocess.py`) passes `mt` as stored, so
                # upcasting here would time a different kernel config (fp32-M instead
                # of bf16-M) than the one it is supposed to mirror.
                raw_dht=card_dht, dht_buffer=dh, mt_buffer=fwd.mt,
                fallback_mask=fwd.fallback_bwd, seq_map_r2c=ctx.seq_map_r2c,
                state_v_first=state_v_first,
            )
    else:
        cp_dht = card_dht

    with region("recompute_h"):
        h, _, _ = fused_gdr_h(
            k=k, v=v, a=a, g=g_c, b=beta, initial_state=fwd.cp_h0,
            output_final_state=False, output_h=True, cu_seqlens=cp_cu,
            state_v_first=state_v_first,
        )
    with region("main_bwd"):
        dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
            q=q, k=k, v=v, a=a, g=g_c, b=beta, do=do, dht=cp_dht, h=h,
            scale=scale, cu_seqlens=cp_cu, state_v_first=state_v_first,
        )
    with region("postprocess"):
        if dh0 is None or initial_state is None:
            dh0 = None
        else:
            if is_intra:
                dh0 = dh0[ctx.seq_map_r2c[:-1].long()]
            if is_inter and not ctx.is_first_rank:
                dh0[0] = 0
        if Hg < Hv:
            dq = group_reduce_vector(dq, Hg)
            dk = group_reduce_vector(dk, Hg)
        dg = chunk_local_cumsum(dg, chunk_size=CHUNK_SIZE, reverse=True, cu_seqlens=cu)

    return dict(dq=dq, dk=dk, dv=dv, dg=dg, db=db, dh0=dh0)


def bwd_available() -> bool:
    return fused_gdr_bwd is not None and fused_gdr_dh is not None
