# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
from __future__ import annotations

import torch
import torch.distributed as dist

from .comm import all_gather_into_tensor, pack_hm, unpack_hm


# ============================================================================
# inter-card preprocess
# ============================================================================
def inter_cp_preprocess_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cp_context,
    state_v_first: bool = False,
    output_mt_first: bool = False,
    initial_state: torch.Tensor | None = None,
):
    from flash_qla.ops.gated_delta_rule.chunk import fused_gdr_h, correct_initial_states

    group = cp_context.group
    cu = cp_context.cu_seqlens
    cu_cpu = cp_context.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    Hv = v.shape[2]
    K = k.shape[3]
    V = v.shape[3]
    chunk_size = a.shape[-1]

    num_warmup = torch.zeros((N, Hv), dtype=cu.dtype, device=k.device)
    num_warmup[-1, :] = (cu_cpu[-1] - cu_cpu[-2] + chunk_size - 1) // chunk_size
    if output_mt_first and N > 1:
        num_warmup[0, :] = (cu_cpu[1] - cu_cpu[0] + chunk_size - 1) // chunk_size

    _, ht, mt = fused_gdr_h(
        k=k, v=v, a=a, g=g, b=beta,
        initial_state=initial_state, output_final_state=True, output_h=False,
        cu_seqlens=cu, num_warmup_chunks=num_warmup, state_v_first=state_v_first,
    )
    S_ext_r = ht[-1].float()
    M_r = mt[-1].float()
    M_r_first = mt[0] if output_mt_first else None

    hm = pack_hm(S_ext_r, M_r)                          # [H, K, V+K]
    ag_hm, _ = all_gather_into_tensor(hm, group=group)  # [W, H, K, V+K]
    rank = dist.get_rank(group=group)
    pre = cp_context.pre_num_ranks
    S_buf, M_buf = unpack_hm(ag_hm[rank - pre: rank + 1], V)

    if cp_context.is_first_rank:
        if output_mt_first:
            return initial_state, M_r_first
        return initial_state

    seq_map, fb_mask = cp_context.get_fwd_scan_tensors(Hv)
    cp_h0_all = correct_initial_states(
        raw_h0=None, ht_buffer=S_buf, mt_buffer=M_buf,
        fallback_mask=fb_mask, seq_map_r2c=seq_map, state_v_first=state_v_first,
    )

    if initial_state is None:
        raw_h0 = torch.zeros((N, Hv, V, K) if state_v_first else (N, Hv, K, V), dtype=torch.float32, device=k.device)
    else:
        raw_h0 = initial_state.clone()
    raw_h0[0] = cp_h0_all[S_buf.shape[0] - 1]

    if output_mt_first:
        return raw_h0, M_r_first
    return raw_h0


def inter_cp_preprocess_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cp_context,
    state_v_first: bool = False,
    M_r_first: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
):
    from flash_qla.ops.gated_delta_rule.chunk import fused_gdr_dh, correct_terminal_states

    group = cp_context.group
    cu = cp_context.cu_seqlens
    cu_cpu = cp_context.cu_seqlens_cpu.tolist()
    N = len(cu_cpu) - 1
    Hv = v.shape[2]
    K = k.shape[3]
    V = v.shape[3]
    chunk_size = a.shape[-1]

    assert M_r_first is not None, "M_r_first must be provided for inter-card CP backward."
    
    num_warmup = torch.zeros((N, Hv), dtype=cu.dtype, device=k.device)
    num_warmup[0, :] = (cu_cpu[1] - cu_cpu[0] + chunk_size - 1) // chunk_size
    _, dh0_buffer = fused_gdr_dh(
        q=q, k=k, a=a, g=g, b=beta, do=do,
        dht=dht, output_dh0=True, output_dh=False, scale=scale,
        cu_seqlens=cu, num_warmup_chunks=num_warmup, state_v_first=state_v_first,
    )
    dS_ext_r = dh0_buffer[0].float()

    M_r_f = M_r_first.float()
    hm = pack_hm(dS_ext_r, M_r_f)
    ag_hm, _ = all_gather_into_tensor(hm, group=group)
    rank = dist.get_rank(group=group)
    post = cp_context.post_num_ranks
    dS_buf, M_buf = unpack_hm(ag_hm[rank: rank + 1 + post], V)

    if cp_context.is_last_rank:
        return None

    seq_map, fb_mask = cp_context.get_bwd_scan_tensors(Hv)
    cp_dht_all = correct_terminal_states(
        raw_dht=None, dht_buffer=dS_buf, mt_buffer=M_buf,
        fallback_mask=fb_mask, seq_map_r2c=seq_map, state_v_first=state_v_first,
    )
    terminal_state_r = cp_dht_all[0]

    if dht is None:
        corrected_dht = torch.zeros((N, Hv, V, K) if state_v_first else (N, Hv, K, V), dtype=torch.float32, device=k.device)
    else:
        corrected_dht = dht.clone()
    corrected_dht[N - 1] = terminal_state_r
    return corrected_dht


# ============================================================================
# intra-card preprocess
# ============================================================================
def intra_cp_preprocess_fwd(
    cp_context,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    raw_h0: torch.Tensor,
    warmup_threshold: float = -10.0,
    state_v_first: bool = False,
    enable_fwd_cp_cache: bool = False,
):
    from flash_qla.ops.gated_delta_rule.chunk import (
        get_warmup_chunks, get_warmup_chunks_bidi, fused_gdr_h, correct_initial_states,
    )

    if not cp_context.use_intra_cp:
        return raw_h0, None

    chunk_size = a.shape[-1]
    cp_cu_seqlens = cp_context.intra_cp_cu_seqlens
    seq_map_r2c = cp_context.seq_map_r2c
    ht_mask = cp_context.ht_mask
    ht_mask_bwd = cp_context.ht_mask_bwd

    if enable_fwd_cp_cache:
        num_warmup_chunks, num_warmup_chunks_bwd, fallback_mask, fallback_mask_bwd = get_warmup_chunks_bidi(
            g=g,
            cu_seqlens=cp_cu_seqlens,
            ht_mask_fwd=ht_mask,
            ht_mask_bwd=ht_mask_bwd,
            chunk_size=chunk_size,
            threshold=warmup_threshold,
        )
    else:
        num_warmup_chunks, fallback_mask = get_warmup_chunks(
            g=g,
            cu_seqlens=cp_cu_seqlens,
            ht_mask=ht_mask,
            chunk_size=chunk_size,
            threshold=warmup_threshold,
        )  # [cp_batch_size, num_v_heads]
        num_warmup_chunks_bwd, fallback_mask_bwd = None, None

    _, ht, mt = fused_gdr_h(
        k=k,
        v=v,
        a=a,
        g=g,
        b=b,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        cu_seqlens=cp_cu_seqlens,
        num_warmup_chunks=num_warmup_chunks,
        state_v_first=state_v_first,
    )  # [cp_batch_size, num_v_heads, k_head_dim, v_head_dim]

    cp_h0 = correct_initial_states(
        raw_h0=raw_h0,
        ht_buffer=ht,
        mt_buffer=mt,
        fallback_mask=fallback_mask,
        seq_map_r2c=seq_map_r2c,
        state_v_first=state_v_first,
    )

    if enable_fwd_cp_cache:
        cp_cache = (cp_h0, mt, fallback_mask_bwd, num_warmup_chunks_bwd)
    else:
        cp_cache = None
    return cp_h0, cp_cache


def intra_cp_preprocess_bwd(
    cp_context,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    raw_h0: torch.Tensor,
    q: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    state_v_first: bool = False,
    cp_cache: tuple | None = None,
):
    from flash_qla.ops.gated_delta_rule.chunk import (
        get_warmup_chunks_bidi, fused_gdr_h, fused_gdr_dh, correct_initial_states,
        correct_terminal_states,
    )

    if fused_gdr_dh is None:
        raise NotImplementedError(
            "Backward pass (CP) is not implemented for SM120 (Blackwell). "
            "Only forward pass is supported on this architecture."
        )

    if not cp_context.use_intra_cp:
        return raw_h0, dht

    chunk_size = a.shape[-1]
    cp_cu_seqlens = cp_context.intra_cp_cu_seqlens
    seq_map_r2c = cp_context.seq_map_r2c
    ht_mask = cp_context.ht_mask
    ht_mask_bwd = cp_context.ht_mask_bwd

    if cp_cache is not None:
        # Use cached forward CP artifacts
        cp_h0, mt_buffer, fallback_bwd, num_warmup_bwd = cp_cache
    else:
        num_warmup_h, num_warmup_bwd, fallback_fwd, fallback_bwd = get_warmup_chunks_bidi(
            g=g, cu_seqlens=cp_cu_seqlens,
            ht_mask_fwd=ht_mask, ht_mask_bwd=ht_mask_bwd,
            chunk_size=chunk_size,
        )

        _, ht_buffer, mt_buffer = fused_gdr_h(
            k=k, v=v, a=a, g=g, b=b,
            initial_state=None,
            output_final_state=True,
            output_h=False,
            cu_seqlens=cp_cu_seqlens,
            num_warmup_chunks=num_warmup_h,
            state_v_first=state_v_first,
        )

        cp_h0 = correct_initial_states(
            raw_h0=raw_h0,
            ht_buffer=ht_buffer,
            mt_buffer=mt_buffer,
            fallback_mask=fallback_fwd,
            seq_map_r2c=seq_map_r2c,
            state_v_first=state_v_first,
        )

    _, dht_buffer = fused_gdr_dh(
        q=q, k=k, a=a, g=g, b=b, do=do,
        dht=None,
        output_dh0=True,
        output_dh=False,
        scale=scale,
        cu_seqlens=cp_cu_seqlens,
        num_warmup_chunks=num_warmup_bwd,
        state_v_first=state_v_first,
    )

    cp_dht = correct_terminal_states(
        raw_dht=dht,
        dht_buffer=dht_buffer,
        mt_buffer=mt_buffer.float(),
        fallback_mask=fallback_bwd,
        seq_map_r2c=seq_map_r2c,
        state_v_first=state_v_first,
    )

    return cp_h0, cp_dht


# ============================================================================
# inter+intra-card preprocess (inter across cards, intra within each card)
# ============================================================================
# The inter+intra path disables the intra warmup optimization: every cp sub-chunk
# goes through the full correction scan (fallback=True everywhere, num_warmup = the
# sub-chunk's full length). This is exact (no warmup truncation) and keeps the
# card-level aggregate consistent with the per-chunk correct scans. Because those
# tensors are then a pure function of the static intra partition, they are
# precomputed once in ``build_inter_intra_cp_context`` (``cp_context.intra_num_warmup``
# / ``cp_context.intra_fallback``) rather than via a per-step warmup kernel.
def inter_intra_cp_preprocess_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cp_context,
    state_v_first: bool = False,
    initial_state: torch.Tensor | None = None,
):
    from flash_qla.ops.gated_delta_rule.chunk import (
        fused_gdr_h, correct_initial_states, aggregate_card_state,
    )

    group = cp_context.group
    Hv = v.shape[2]
    V = v.shape[3]
    chunk_size = a.shape[-1]

    # Degenerate: this card's local sequence is too short for an intra split.
    # It behaves exactly like pure inter-card CP, so delegate. Note the packed
    # (S*, M) tensor has the same shape/dtype as the full path, so a mixed set of
    # degenerate/full ranks still all-gather consistently.
    if not cp_context.use_intra_cp:
        raw_h0, M_r_first = inter_cp_preprocess_fwd(
            k=k, v=v, a=a, g=g, beta=beta, cp_context=cp_context,
            state_v_first=state_v_first, output_mt_first=True, initial_state=initial_state,
        )
        return raw_h0, ("inter_intra_degenerate", M_r_first, raw_h0)

    cp_cu = cp_context.intra_cp_cu_seqlens
    seq_map_r2c = cp_context.seq_map_r2c
    N_local = cp_context.num_seqs

    # full-fallback warmup/mask, precomputed on the context (no per-step kernel).
    num_warmup = cp_context.intra_num_warmup
    fallback = cp_context.intra_fallback

    # Step 1: intra prepare_h -> per-intra-chunk S* (ht) and M (mt).
    _, ht, mt = fused_gdr_h(
        k=k, v=v, a=a, g=g, b=beta,
        initial_state=None, output_final_state=True, output_h=False,
        cu_seqlens=cp_cu, num_warmup_chunks=num_warmup, state_v_first=state_v_first,
    )

    # Step 2: aggregate the card's intra chunks -> per-raw-seq card-level S*, M.
    S_card, M_card = aggregate_card_state(
        ht, mt, fallback, seq_map_r2c, state_v_first=state_v_first, compute_m=True,
    )

    # Steps 3-4: inter all-gather + correct (mirror inter_cp_preprocess_fwd; the
    # cross-card boundary is this card's LAST local raw sequence).
    S_ext_r = S_card[-1]
    M_r = M_card[-1]
    hm = pack_hm(S_ext_r, M_r)
    ag_hm, _ = all_gather_into_tensor(hm, group=group)
    rank = dist.get_rank(group=group)
    pre = cp_context.pre_num_ranks
    S_buf, M_buf = unpack_hm(ag_hm[rank - pre: rank + 1], V)

    if cp_context.is_first_rank:
        card_inter_h0 = initial_state
    else:
        seq_map, fb_mask = cp_context.get_fwd_scan_tensors(Hv)
        cp_h0_all = correct_initial_states(
            raw_h0=None, ht_buffer=S_buf, mt_buffer=M_buf,
            fallback_mask=fb_mask, seq_map_r2c=seq_map, state_v_first=state_v_first,
        )
        if initial_state is None:
            card_inter_h0 = torch.zeros(
                (N_local, Hv, V, k.shape[3]) if state_v_first else (N_local, Hv, k.shape[3], V),
                dtype=torch.float32, device=k.device,
            )
        else:
            card_inter_h0 = initial_state.clone()
        card_inter_h0[0] = cp_h0_all[S_buf.shape[0] - 1]

    # Step 5: intra correct -> per-intra-chunk corrected initial states, seeded by
    # the inter-level card state. This feeds the main fused_gdr_fwd kernel.
    cp_h0 = correct_initial_states(
        raw_h0=card_inter_h0, ht_buffer=ht, mt_buffer=mt,
        fallback_mask=fallback, seq_map_r2c=seq_map_r2c, state_v_first=state_v_first,
    )

    # warmup/fallback live on the context, so the bwd cache only needs h0/M/M_card.
    cp_cache = ("inter_intra", cp_h0, mt, M_card)
    return cp_h0, cp_cache


def inter_intra_cp_preprocess_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    cp_context,
    state_v_first: bool = False,
    cp_cache: tuple | None = None,
):
    from flash_qla.ops.gated_delta_rule.chunk import (
        fused_gdr_dh, correct_terminal_states, aggregate_card_state,
    )

    if fused_gdr_dh is None:
        raise NotImplementedError(
            "Backward pass (inter+intra CP) is not implemented for SM120 (Blackwell). "
            "Only forward pass is supported on this architecture."
        )

    # Degenerate card: pure inter-card backward.
    if cp_cache is not None and cp_cache[0] == "inter_intra_degenerate":
        _, M_r_first, fwd_h0 = cp_cache
        corrected_dht = inter_cp_preprocess_bwd(
            q=q, k=k, v=v, a=a, g=g, beta=beta, do=do, scale=scale,
            cp_context=cp_context, state_v_first=state_v_first,
            M_r_first=M_r_first, dht=dht,
        )
        if dht is not None and corrected_dht is None:
            corrected_dht = dht
        return fwd_h0, corrected_dht

    _, cp_h0, mt, M_card = cp_cache

    group = cp_context.group
    Hv = v.shape[2]
    V = v.shape[3]
    cp_cu = cp_context.intra_cp_cu_seqlens
    seq_map_r2c = cp_context.seq_map_r2c
    N_local = cp_context.num_seqs
    # full-fallback warmup/mask from the context (same as forward).
    num_warmup = cp_context.intra_num_warmup
    fallback = cp_context.intra_fallback

    # Step 1: intra prepare_dh -> per-intra-chunk dh*.
    _, dh_buffer = fused_gdr_dh(
        q=q, k=k, a=a, g=g, b=beta, do=do,
        dht=None, output_dh0=True, output_dh=False, scale=scale,
        cu_seqlens=cp_cu, num_warmup_chunks=num_warmup, state_v_first=state_v_first,
    )

    # Step 2: aggregate (reverse) -> per-raw-seq card dS*. dh_buffer is fp32, so mt
    # is upcast to match (the aggregate kernel uses one buffer dtype). The card-level
    # M is the forward M_card (reused from cache), so no M recompute here.
    dS_card, _ = aggregate_card_state(
        dh_buffer, mt.float(), fallback, seq_map_r2c, state_v_first=state_v_first,
        reverse=True, transpose_m=True, compute_m=False,
    )

    # Steps 3-4: inter all-gather + correct_terminal (mirror inter_cp_preprocess_bwd;
    # boundary is this card's FIRST local raw sequence for the backward direction).
    dS_ext_r = dS_card[0]
    M_r_f = M_card[0].float()
    hm = pack_hm(dS_ext_r, M_r_f)
    ag_hm, _ = all_gather_into_tensor(hm, group=group)
    rank = dist.get_rank(group=group)
    post = cp_context.post_num_ranks
    dS_buf, M_buf = unpack_hm(ag_hm[rank: rank + 1 + post], V)

    if dht is None:
        card_inter_dht = torch.zeros(
            (N_local, Hv, V, k.shape[3]) if state_v_first else (N_local, Hv, k.shape[3], V),
            dtype=torch.float32, device=k.device,
        )
    else:
        card_inter_dht = dht.clone()

    if not cp_context.is_last_rank:
        seq_map, fb_mask = cp_context.get_bwd_scan_tensors(Hv)
        cp_dht_all = correct_terminal_states(
            raw_dht=None, dht_buffer=dS_buf, mt_buffer=M_buf,
            fallback_mask=fb_mask, seq_map_r2c=seq_map, state_v_first=state_v_first,
        )
        card_inter_dht[N_local - 1] = cp_dht_all[0]

    # Step 5: intra correct_terminal -> per-intra-chunk dh, seeded by the inter
    # card terminal gradient.
    cp_dht = correct_terminal_states(
        raw_dht=card_inter_dht, dht_buffer=dh_buffer, mt_buffer=mt.float(),
        fallback_mask=fallback, seq_map_r2c=seq_map_r2c, state_v_first=state_v_first,
    )

    return cp_h0, cp_dht


def cp_preprocess_fwd(
    cp_context,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
    enable_fwd_cp_cache: bool,
):
    if cp_context.is_inter_intra_cp_enabled:
        return inter_intra_cp_preprocess_fwd(
            k=k, v=v, a=a, g=g, beta=beta, cp_context=cp_context,
            state_v_first=state_v_first, initial_state=initial_state,
        )

    if cp_context.is_inter_cp_enabled:
        raw_h0, M_r_first = inter_cp_preprocess_fwd(
            k=k, v=v, a=a, g=g, beta=beta, cp_context=cp_context,
            state_v_first=state_v_first, output_mt_first=True, initial_state=initial_state,
        )
        return raw_h0, ("inter", M_r_first, raw_h0)

    h0, intra_cache = intra_cp_preprocess_fwd(
        cp_context, k=k, v=v, a=a, g=g, b=beta, raw_h0=initial_state,
        state_v_first=state_v_first, enable_fwd_cp_cache=enable_fwd_cp_cache,
    )
    cp_cache = ("intra", intra_cache) if intra_cache is not None else None
    return h0, cp_cache


def cp_preprocess_bwd(
    cp_context,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
    cp_cache,
):
    if cp_context.is_inter_intra_cp_enabled:
        return inter_intra_cp_preprocess_bwd(
            q=q, k=k, v=v, a=a, g=g, beta=beta, do=do, dht=dht, scale=scale,
            cp_context=cp_context, state_v_first=state_v_first, cp_cache=cp_cache,
        )

    if cp_context.is_inter_cp_enabled:
        _, M_r_first, fwd_h0 = cp_cache if (cp_cache is not None and cp_cache[0] == "inter") else (None, None, None)
        corrected_dht = inter_cp_preprocess_bwd(
            q=q, k=k, v=v, a=a, g=g, beta=beta, do=do, scale=scale,
            cp_context=cp_context, state_v_first=state_v_first,
            M_r_first=M_r_first, dht=dht,
        )

        if dht is not None and corrected_dht is None:
            corrected_dht = dht
        return fwd_h0, corrected_dht

    intra_cache = cp_cache[1] if (cp_cache is not None and cp_cache[0] == "intra") else None
    return intra_cp_preprocess_bwd(
        cp_context, k=k, v=v, a=a, g=g, b=beta, raw_h0=initial_state,
        q=q, do=do, dht=dht, scale=scale,
        state_v_first=state_v_first, cp_cache=intra_cache,
    )


def finalize_dh0(dh0, cp_context, has_initial_state: bool):
    if dh0 is None or not has_initial_state:
        return None
    if cp_context.is_inter_cp_enabled:
        if not cp_context.is_first_rank:
            dh0[0] = 0
    elif cp_context.is_intra_cp_enabled:
        # TODO store dh0 in fused_bwd kernel
        dh0 = dh0[cp_context.seq_map_r2c[:-1].long()]
    elif cp_context.is_inter_intra_cp_enabled:
        # intra remap first (per-chunk -> per-raw-seq), then inter boundary zero
        # (first local seq's h0 grad belongs to the previous card).
        if cp_context.use_intra_cp:
            dh0 = dh0[cp_context.seq_map_r2c[:-1].long()]
        if not cp_context.is_first_rank:
            dh0[0] = 0
    return dh0
