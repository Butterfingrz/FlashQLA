# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""L2: the CP support kernels against plain-torch references.

These are the five entry points the CP pre/post-processing is built out of::

    get_warmup_chunks        how far back each cp segment must warm up (one direction)
    get_warmup_chunks_bidi   the same, both directions, in one launch
    correct_initial_states   forward scan: per-chunk entering state
    correct_terminal_states  the same scan, reversed and transposed (backward)
    aggregate_card_state     the same scan again, keeping only the per-raw-seq result
                             plus the ordered M product (SM100/SM103 only)

The e2e tests catch *that* CP is wrong; these catch *where*. Every reference below is a
literal transcription of the kernel's index arithmetic, so a changed convention (floor vs
ceil, which token of a chunk is sampled, which side M multiplies on) fails here with a
readable diff instead of showing up as a slightly-off gradient three layers up.

Note ``correct_initial_states``, ``correct_terminal_states`` and ``aggregate_card_state``
are three configurations of one kernel (``tilelang_correct_h0``), so they share one
reference scan here too -- the same way they share one kernel body.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cp_arch import supports
from flash_qla.ops.gated_delta_rule.chunk import (
    CHUNK_SIZE,
    get_warmup_chunks,
    get_warmup_chunks_bidi,
    correct_initial_states,
    correct_terminal_states,
    aggregate_card_state,
)

DEVICE = "cuda"
THRESHOLD = -10.0          # the kernels' default log-decay cutoff
K = V = 128                # the scan kernels assert this
SEED = 20260819

# The scan kernels' gemms may run on tensor cores (TF32-class precision), so the
# comparison is a loose relative-max against a float64 reference. A transposed operand or
# an off-by-one index is O(1) wrong, which this still catches by orders of magnitude.
SCAN_RTOL = 2e-2


# ===========================================================================
# Synthetic cp layouts
#
# `segments` is the number of cp segments per raw sequence, so [1, 3, 2] means three raw
# sequences split into 1, 3 and 2 cp segments. Interior segments are chunk-aligned (the
# intra split only cuts on chunk boundaries); the last segment of a raw sequence may be
# ragged, which is what makes the floor-vs-ceil difference between the two warmup kernels
# observable.
# ===========================================================================
LAYOUTS = [
    pytest.param([1], id="one-seq-unsplit"),
    pytest.param([2, 2, 2], id="uniform-split"),
    pytest.param([1, 3, 2], id="mixed-split"),
    pytest.param([5], id="deep-split"),
]


def _build_layout(segments, *, ragged_tail):
    """-> (cu_seqlens, seq_map_r2c, ht_mask_fwd, ht_mask_bwd) for a synthetic split."""
    lengths = []
    seq_map = [0]
    for n_seg in segments:
        for j in range(n_seg):
            length = (j % 3 + 1) * CHUNK_SIZE
            if ragged_tail and j == n_seg - 1:
                length += 13
            lengths.append(length)
        seq_map.append(len(lengths))

    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    n_cp = len(lengths)

    ht_fwd = torch.zeros(n_cp, dtype=torch.bool, device=DEVICE)
    ht_bwd = torch.zeros(n_cp, dtype=torch.bool, device=DEVICE)
    for i in range(len(segments)):
        ht_fwd[seq_map[i + 1] - 1] = True   # last cp segment of the raw seq
        ht_bwd[seq_map[i]] = True           # first cp segment of the raw seq

    return (
        torch.tensor(cu, dtype=torch.int32, device=DEVICE),
        torch.tensor(seq_map, dtype=torch.int32, device=DEVICE),
        ht_fwd,
        ht_bwd,
    )


def _make_g(cu, num_heads, *, seed=SEED):
    """Per-token log decay, cumulative within each chunk -- the layout the warmup kernels
    expect (they sample one token per chunk and read its within-chunk cumulative sum).

    Head 0 decays slowly (never crosses the threshold, so it falls back), the last head
    decays fast (crosses after one chunk), and the rest are spread in between, so a single
    call exercises both sides of the branch.
    """
    total = int(cu[-1].item())
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    base = -torch.rand(1, total, num_heads, generator=gen, device=DEVICE) * 0.5 - 0.01
    scale = torch.logspace(
        -2, 1.2, num_heads, device=DEVICE
    ) if num_heads > 1 else torch.ones(1, device=DEVICE)
    g = base * scale
    # cumulative within each chunk of each segment
    cu_list = cu.tolist()
    for lo, hi in zip(cu_list[:-1], cu_list[1:]):
        for c0 in range(lo, hi, CHUNK_SIZE):
            c1 = min(c0 + CHUNK_SIZE, hi)
            g[0, c0:c1] = torch.cumsum(g[0, c0:c1], dim=0)
    return g.float()


# ===========================================================================
# References
# ===========================================================================
def _ref_warmup(g, cu, ht_mask, *, reverse):
    """Transcription of ``tilelang_get_warmup_chunks_kernel``.

    ``num_iters`` floors (unlike the bidi kernel, which ceils), and where ``ht_mask`` is
    set the kernel writes only ``num_warmup=0`` -- it leaves that row of
    ``fallback_mask`` untouched, so the caller must not read it. Returned as None here to
    make that explicit.
    """
    cu_list = cu.tolist()
    n_cp = len(cu_list) - 1
    num_heads = g.shape[2]
    num_warmup = torch.zeros(n_cp, num_heads, dtype=cu.dtype, device=g.device)
    fallback = torch.zeros(n_cp, num_heads, dtype=torch.bool, device=g.device)
    valid = torch.ones(n_cp, num_heads, dtype=torch.bool, device=g.device)

    for bb in range(n_cp):
        start, end = cu_list[bb], cu_list[bb + 1]
        num_iters = (end - start) // CHUNK_SIZE
        if ht_mask[bb]:
            num_warmup[bb] = 0
            valid[bb] = False
            continue
        num_warmup[bb] = num_iters
        fallback[bb] = True
        cumsum = torch.zeros(num_heads, dtype=torch.float64, device=g.device)
        for i_s in range(num_iters):
            idx = start + (i_s + 1) * CHUNK_SIZE - 1 if reverse else end - i_s * CHUNK_SIZE - 1
            cumsum += g[0, idx].double()
            hit = (cumsum < THRESHOLD) & (num_warmup[bb] == num_iters)
            num_warmup[bb][hit] = i_s + 1
            fallback[bb][hit] = False
    return num_warmup, fallback, valid


def _ref_warmup_bidi(g, cu, ht_mask_fwd, ht_mask_bwd):
    """Transcription of ``tilelang_get_warmup_chunks_bidi_kernel``.

    Differences from the single-direction kernel that matter: ``num_iters`` *ceils*, the
    backward sample index is clamped to the last token, ``fallback`` is written on every
    row, and the returned forward warmup count is ``max(n_fwd, n_bwd)`` -- one prepare_h
    launch has to cover both directions' needs.
    """
    cu_list = cu.tolist()
    n_cp = len(cu_list) - 1
    H = g.shape[2]
    dev = g.device
    n_fwd = torch.zeros(n_cp, H, dtype=cu.dtype, device=dev)
    n_bwd = torch.zeros(n_cp, H, dtype=cu.dtype, device=dev)
    f_fwd = torch.zeros(n_cp, H, dtype=torch.bool, device=dev)
    f_bwd = torch.zeros(n_cp, H, dtype=torch.bool, device=dev)

    for bb in range(n_cp):
        start, end = cu_list[bb], cu_list[bb + 1]
        num_iters = -(-(end - start) // CHUNK_SIZE)
        if ht_mask_fwd[bb]:
            n_fwd[bb], f_fwd[bb] = 0, False
        else:
            n_fwd[bb], f_fwd[bb] = num_iters, True
        if ht_mask_bwd[bb]:
            n_bwd[bb], f_bwd[bb] = 0, False
        else:
            n_bwd[bb], f_bwd[bb] = num_iters, True

        cs_fwd = torch.zeros(H, dtype=torch.float64, device=dev)
        cs_bwd = torch.zeros(H, dtype=torch.float64, device=dev)
        for i_s in range(num_iters):
            cs_fwd += g[0, end - i_s * CHUNK_SIZE - 1].double()
            hit = (cs_fwd < THRESHOLD) & (n_fwd[bb] == num_iters)
            n_fwd[bb][hit] = i_s + 1
            f_fwd[bb][hit] = False

            bwd_idx = min(start + (i_s + 1) * CHUNK_SIZE - 1, end - 1)
            cs_bwd += g[0, bwd_idx].double()
            hit = (cs_bwd < THRESHOLD) & (n_bwd[bb] == num_iters)
            n_bwd[bb][hit] = i_s + 1
            f_bwd[bb][hit] = False

    return torch.maximum(n_fwd, n_bwd), n_bwd, f_fwd, f_bwd


def _ref_scan(
    ht, mt, fallback, seq_map, *,
    raw_h0=None, state_v_first=False, reverse=False, transpose_m=False,
):
    """Transcription of ``tilelang_correct_h0``'s ``scan_body``, all three modes at once.

    Returns ``(cp_h0, h_card, m_card)``:

    * ``cp_h0[idx]`` -- the state *entering* cp segment ``idx`` (correct mode),
    * ``h_card[bb]`` -- the state *leaving* the last segment of raw seq ``bb``
      (aggregate mode),
    * ``m_card[bb]`` -- the ordered M product over the raw sequence's segments. A
      non-fallback segment zeroes the accumulator, and because later segments multiply
      into it, that zero is permanent: M is non-zero only when *every* segment of the
      sequence is a fallback. That is the intended meaning -- if any chunk's history is
      already dead, no correction can transfer across the sequence at all.
    """
    ht = ht.double()
    mt = mt.double()
    n_cp, H = ht.shape[0], ht.shape[1]
    n_raw = seq_map.shape[0] - 1
    seq_map = seq_map.tolist()

    cp_h0 = torch.zeros_like(ht)
    h_card = torch.zeros((n_raw,) + ht.shape[1:], dtype=torch.float64, device=ht.device)
    m_card = torch.zeros((n_raw, H, K, K), dtype=torch.float64, device=ht.device)

    for bb in range(n_raw):
        lo, hi = seq_map[bb], seq_map[bb + 1]
        order = range(hi - 1, lo - 1, -1) if reverse else range(lo, hi)
        for h in range(H):
            state = (
                raw_h0[bb, h].double() if raw_h0 is not None
                else torch.zeros_like(ht[0, h])
            )
            m_run = torch.eye(K, dtype=torch.float64, device=ht.device)
            for idx in order:
                cp_h0[idx, h] = state
                prev, m = state, mt[idx, h]
                state = ht[idx, h].clone()
                if fallback[idx, h]:
                    if state_v_first:
                        state += prev @ (m if transpose_m else m.T)
                    else:
                        state += (m.T if transpose_m else m) @ prev
                    m_run = m @ m_run
                else:
                    m_run = torch.zeros_like(m_run)
            h_card[bb, h] = state
            m_card[bb, h] = m_run
    return cp_h0, h_card, m_card


def _make_scan_inputs(seq_map, n_cp, H, *, state_v_first, fallback_pattern, seed=SEED):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    shape = (n_cp, H, V, K) if state_v_first else (n_cp, H, K, V)
    ht = torch.randn(shape, generator=gen, device=DEVICE, dtype=torch.float32) * 0.1
    # spectral norm well under 1 so the M product decays instead of blowing up
    mt = torch.randn(
        (n_cp, H, K, K), generator=gen, device=DEVICE, dtype=torch.float32
    ) * (0.5 / K ** 0.5)
    if fallback_pattern == "all":
        fallback = torch.ones(n_cp, H, dtype=torch.bool, device=DEVICE)
    elif fallback_pattern == "none":
        fallback = torch.zeros(n_cp, H, dtype=torch.bool, device=DEVICE)
    else:
        fallback = torch.rand(n_cp, H, generator=gen, device=DEVICE) < 0.7
    return ht, mt, fallback


def _rel_max(actual, expected):
    diff = (actual.double() - expected.double()).abs().max().item()
    return diff / (expected.double().abs().max().item() + 1e-30)


FALLBACK_PATTERNS = ["all", "none", "random"]


# ===========================================================================
# get_warmup_chunks
# ===========================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("segments", LAYOUTS)
@pytest.mark.parametrize("ragged_tail", [False, True], ids=["aligned", "ragged"])
@pytest.mark.parametrize("reverse", [False, True], ids=["fwd", "reverse"])
@pytest.mark.parametrize("num_heads", [4, 16])
def test_get_warmup_chunks(segments, ragged_tail, reverse, num_heads):
    cu, _, ht_fwd, ht_bwd = _build_layout(segments, ragged_tail=ragged_tail)
    ht_mask = ht_bwd if reverse else ht_fwd
    g = _make_g(cu, num_heads)

    num_warmup, fallback = get_warmup_chunks(
        g=g, cu_seqlens=cu, ht_mask=ht_mask, chunk_size=CHUNK_SIZE, reverse=reverse,
    )
    ref_warmup, ref_fallback, valid = _ref_warmup(g, cu, ht_mask, reverse=reverse)

    assert torch.equal(num_warmup, ref_warmup), (
        f"num_warmup mismatch\ngot:\n{num_warmup}\nwant:\n{ref_warmup}"
    )
    # `fallback_mask` rows under ht_mask are never written by this kernel, so only the
    # rest are meaningful; see _ref_warmup.
    assert torch.equal(fallback[valid], ref_fallback[valid]), (
        f"fallback mismatch\ngot:\n{fallback}\nwant:\n{ref_fallback}\nvalid:\n{valid}"
    )


@pytest.mark.gpu
def test_get_warmup_chunks_covers_both_branches():
    """The layered g above must actually produce a mix of fallback True and False --
    otherwise the comparison tests would pass while only covering one branch."""
    cu, _, ht_fwd, _ = _build_layout([1, 3, 2], ragged_tail=False)
    g = _make_g(cu, 16)
    _, fallback = get_warmup_chunks(
        g=g, cu_seqlens=cu, ht_mask=ht_fwd, chunk_size=CHUNK_SIZE,
    )
    interior = fallback[~ht_fwd]
    assert interior.any() and not interior.all(), (
        f"expected both fallback outcomes, got\n{fallback}"
    )


# ===========================================================================
# get_warmup_chunks_bidi
# ===========================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("segments", LAYOUTS)
@pytest.mark.parametrize("ragged_tail", [False, True], ids=["aligned", "ragged"])
@pytest.mark.parametrize("num_heads", [4, 16])
def test_get_warmup_chunks_bidi(segments, ragged_tail, num_heads):
    cu, _, ht_fwd, ht_bwd = _build_layout(segments, ragged_tail=ragged_tail)
    g = _make_g(cu, num_heads)

    n_h, n_bwd, f_fwd, f_bwd = get_warmup_chunks_bidi(
        g=g, cu_seqlens=cu, ht_mask_fwd=ht_fwd, ht_mask_bwd=ht_bwd,
        chunk_size=CHUNK_SIZE,
    )
    r_h, r_bwd, r_f_fwd, r_f_bwd = _ref_warmup_bidi(g, cu, ht_fwd, ht_bwd)

    assert torch.equal(n_h, r_h), f"num_warmup_h\ngot:\n{n_h}\nwant:\n{r_h}"
    assert torch.equal(n_bwd, r_bwd), f"num_warmup_bwd\ngot:\n{n_bwd}\nwant:\n{r_bwd}"
    assert torch.equal(f_fwd, r_f_fwd), f"fallback_fwd\ngot:\n{f_fwd}\nwant:\n{r_f_fwd}"
    assert torch.equal(f_bwd, r_f_bwd), f"fallback_bwd\ngot:\n{f_bwd}\nwant:\n{r_f_bwd}"


@pytest.mark.gpu
def test_get_warmup_chunks_bidi_h_covers_both_directions():
    """``num_warmup_h`` is deliberately the elementwise max of the two directions: one
    prepare_h launch has to warm up enough for the forward *and* the backward scan."""
    cu, _, ht_fwd, ht_bwd = _build_layout([1, 3, 2], ragged_tail=False)
    g = _make_g(cu, 16)
    n_h, n_bwd, _, _ = get_warmup_chunks_bidi(
        g=g, cu_seqlens=cu, ht_mask_fwd=ht_fwd, ht_mask_bwd=ht_bwd,
        chunk_size=CHUNK_SIZE,
    )
    # the layout is chunk-aligned, so the single-direction kernel's forward count is
    # directly comparable (floor == ceil there)
    n_fwd_only, _ = get_warmup_chunks(
        g=g, cu_seqlens=cu, ht_mask=ht_fwd, chunk_size=CHUNK_SIZE,
    )
    assert torch.all(n_h >= n_bwd), f"num_warmup_h={n_h}\nnum_warmup_bwd={n_bwd}"
    assert torch.all(n_h >= n_fwd_only), f"num_warmup_h={n_h}\nnum_warmup_fwd={n_fwd_only}"
    assert (n_h > n_bwd).any(), "expected the forward direction to dominate somewhere"


@pytest.mark.gpu
@pytest.mark.parametrize("segments", LAYOUTS)
def test_get_warmup_chunks_bidi_agrees_with_single_on_aligned_layouts(segments):
    """On chunk-aligned segments floor == ceil, so the bidi kernel's forward half must
    reproduce the single-direction kernel exactly (fallback included). This is what pins
    the two kernels to one convention."""
    cu, _, ht_fwd, ht_bwd = _build_layout(segments, ragged_tail=False)
    g = _make_g(cu, 8)

    _, f_single = get_warmup_chunks(
        g=g, cu_seqlens=cu, ht_mask=ht_fwd, chunk_size=CHUNK_SIZE,
    )
    _, _, f_bidi, _ = get_warmup_chunks_bidi(
        g=g, cu_seqlens=cu, ht_mask_fwd=ht_fwd, ht_mask_bwd=ht_bwd,
        chunk_size=CHUNK_SIZE,
    )
    interior = ~ht_fwd
    assert torch.equal(f_bidi[interior], f_single[interior]), (
        f"fallback_fwd disagrees\nbidi:\n{f_bidi}\nsingle:\n{f_single}"
    )


# ===========================================================================
# correct_initial_states / correct_terminal_states
# ===========================================================================
_FWD_BUF_DTYPES = [
    pytest.param(torch.float32, torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, torch.bfloat16, id="prod-bf16"),
]
_BWD_BUF_DTYPES = [
    pytest.param(torch.float32, torch.float32, id="fp32"),
    pytest.param(torch.float32, torch.bfloat16, id="prod-mixed"),
]

@pytest.mark.gpu
@pytest.mark.parametrize("segments", LAYOUTS)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_raw_h0", [False, True], ids=["zero_seed", "raw_h0"])
@pytest.mark.parametrize("fallback_pattern", FALLBACK_PATTERNS)
@pytest.mark.parametrize("h_dtype, m_dtype", _FWD_BUF_DTYPES)
def test_correct_initial_states(
    segments, state_v_first, use_raw_h0, fallback_pattern, h_dtype, m_dtype
):
    H = 4
    cu, seq_map, _, _ = _build_layout(segments, ragged_tail=False)
    n_cp = cu.shape[0] - 1
    ht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first, fallback_pattern=fallback_pattern,
    )
    ht, mt = ht.to(h_dtype), mt.to(m_dtype)
    raw_shape = (len(segments), H, V, K) if state_v_first else (len(segments), H, K, V)
    raw_h0 = torch.randn(raw_shape, device=DEVICE, dtype=torch.float32) if use_raw_h0 else None

    out = correct_initial_states(
        raw_h0=raw_h0, ht_buffer=ht, mt_buffer=mt, fallback_mask=fallback,
        seq_map_r2c=seq_map, state_v_first=state_v_first,
    )
    ref, _, _ = _ref_scan(
        ht, mt, fallback, seq_map, raw_h0=raw_h0, state_v_first=state_v_first,
    )
    err = _rel_max(out, ref)
    assert out.dtype == torch.float32, "the main kernel's seed must stay fp32"
    assert err <= SCAN_RTOL, f"cp_h0 rel_max={err:.2e} > {SCAN_RTOL:g}"


@pytest.mark.gpu
@pytest.mark.parametrize("segments", LAYOUTS)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("use_raw_dht", [False, True], ids=["zero_seed", "raw_dht"])
@pytest.mark.parametrize("fallback_pattern", FALLBACK_PATTERNS)
@pytest.mark.parametrize("h_dtype, m_dtype", _BWD_BUF_DTYPES)
def test_correct_terminal_states(
    segments, state_v_first, use_raw_dht, fallback_pattern, h_dtype, m_dtype
):
    """The backward twin: same scan, walked in reverse, with M transposed."""
    H = 4
    cu, seq_map, _, _ = _build_layout(segments, ragged_tail=False)
    n_cp = cu.shape[0] - 1
    dht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first,
        fallback_pattern=fallback_pattern, seed=SEED + 1,
    )
    dht, mt = dht.to(h_dtype), mt.to(m_dtype)
    raw_shape = (len(segments), H, V, K) if state_v_first else (len(segments), H, K, V)
    raw_dht = torch.randn(raw_shape, device=DEVICE, dtype=torch.float32) if use_raw_dht else None

    out = correct_terminal_states(
        raw_dht=raw_dht, dht_buffer=dht, mt_buffer=mt, fallback_mask=fallback,
        seq_map_r2c=seq_map, state_v_first=state_v_first,
    )
    ref, _, _ = _ref_scan(
        dht, mt, fallback, seq_map, raw_h0=raw_dht,
        state_v_first=state_v_first, reverse=True, transpose_m=True,
    )
    err = _rel_max(out, ref)
    assert out.dtype == torch.float32, "the main kernel's seed must stay fp32"
    assert err <= SCAN_RTOL, f"cp_dht rel_max={err:.2e} > {SCAN_RTOL:g}"


@pytest.mark.gpu
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_correct_initial_states_no_fallback_is_pure_shift(state_v_first):
    """With fallback all False nothing folds in: the state entering segment ``i`` is just
    the previous segment's ``ht``, and the first one is the raw seed. A dedicated case
    because it is the shape of the answer when the decay has already killed the history.
    """
    H = 4
    cu, seq_map, _, _ = _build_layout([2, 2, 2], ragged_tail=False)
    n_cp = cu.shape[0] - 1
    ht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first, fallback_pattern="none",
    )
    raw_shape = (3, H, V, K) if state_v_first else (3, H, K, V)
    raw_h0 = torch.randn(raw_shape, device=DEVICE, dtype=torch.float32)

    out = correct_initial_states(
        raw_h0=raw_h0, ht_buffer=ht, mt_buffer=mt, fallback_mask=fallback,
        seq_map_r2c=seq_map, state_v_first=state_v_first,
    )
    seq_map_list = seq_map.tolist()
    for bb in range(3):
        lo, hi = seq_map_list[bb], seq_map_list[bb + 1]
        assert _rel_max(out[lo], raw_h0[bb]) <= SCAN_RTOL, f"seq {bb} seed"
        for idx in range(lo + 1, hi):
            assert _rel_max(out[idx], ht[idx - 1]) <= SCAN_RTOL, f"seq {bb} segment {idx}"


# ===========================================================================
# aggregate_card_state (inter+intra only)
# ===========================================================================
@pytest.mark.gpu
@pytest.mark.cp_inter_intra
@pytest.mark.parametrize("segments", LAYOUTS)
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("fallback_pattern", FALLBACK_PATTERNS)
@pytest.mark.parametrize(
    "reverse, transpose_m", [(False, False), (True, True)], ids=["fwd", "bwd"],
)
def test_aggregate_card_state(segments, state_v_first, fallback_pattern, reverse, transpose_m):
    H = 4
    cu, seq_map, _, _ = _build_layout(segments, ragged_tail=False)
    n_cp = cu.shape[0] - 1
    ht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first,
        fallback_pattern=fallback_pattern, seed=SEED + 2,
    )

    h_card, m_card = aggregate_card_state(
        ht, mt, fallback, seq_map, state_v_first=state_v_first,
        reverse=reverse, transpose_m=transpose_m, compute_m=True,
    )
    _, ref_h, ref_m = _ref_scan(
        ht, mt, fallback, seq_map, state_v_first=state_v_first,
        reverse=reverse, transpose_m=transpose_m,
    )
    err_h = _rel_max(h_card, ref_h)
    assert err_h <= SCAN_RTOL, f"h_card rel_max={err_h:.2e} > {SCAN_RTOL:g}"
    err_m = _rel_max(m_card, ref_m)
    assert err_m <= SCAN_RTOL, f"m_card rel_max={err_m:.2e} > {SCAN_RTOL:g}"


@pytest.mark.gpu
@pytest.mark.cp_inter_intra
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
def test_aggregate_card_state_without_m(state_v_first):
    """``compute_m=False`` is the backward dh aggregate, which reuses the forward
    ``m_card``: it must return None for M and the same h_card as the compute_m run."""
    H = 4
    cu, seq_map, _, _ = _build_layout([1, 3, 2], ragged_tail=False)
    n_cp = cu.shape[0] - 1
    ht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first, fallback_pattern="random",
    )
    h_with, m_with = aggregate_card_state(
        ht, mt, fallback, seq_map, state_v_first=state_v_first, compute_m=True,
    )
    h_without, m_without = aggregate_card_state(
        ht, mt, fallback, seq_map, state_v_first=state_v_first, compute_m=False,
    )
    assert m_with is not None and m_without is None
    # `compute_m` selects a separately compiled kernel (extra shared buffers, different
    # scheduling), but the h recurrence itself is identical, so the two agree bit-for-bit
    # in practice; 1e-5 only leaves room for a future scheduling change to reorder the
    # same arithmetic. It is *not* an accuracy budget -- a real divergence here means one
    # of the two variants is computing the wrong thing, which is exactly how the missing
    # barrier in test_aggregate_card_state_survives_a_non_fallback_iteration first showed
    # up. Do not loosen it.
    err = _rel_max(h_without, h_with)
    assert err <= 1e-5, f"h_card must not depend on compute_m, rel_max={err:.2e}"


@pytest.mark.gpu
@pytest.mark.cp_inter_intra
@pytest.mark.parametrize("state_v_first", [False, True], ids=["kv", "vk"])
@pytest.mark.parametrize("compute_m", [True, False], ids=["m", "no_m"])
def test_aggregate_card_state_survives_a_non_fallback_iteration(state_v_first, compute_m):
    """Regression: the scan's per-iteration shared-memory load must be synchronised on
    *every* iteration, not only on the ones that take the fallback branch.

    The scan loads ``ht[idx]`` into shared memory and then reads it back into the
    accumulator fragment. TileLang used to sink the ``__syncthreads()`` between those two
    into the ``if fallback_mask[idx, bh]`` block (the first shared *read* it saw), so a
    segment with ``fallback=False`` read the tile while other threads were still writing
    it -- garbage, at O(1) relative error, non-deterministic across processes.

    The 48-case matrix above missed it: at ``H=4`` on a 4-segment layout only a handful of
    (seq, head) pairs are exposed and most launches came back clean. This pins the trigger
    directly: many heads, and a fallback mask whose *last* segment is the non-fallback one,
    which is where the corruption is most likely (the accumulator is read out immediately
    afterwards, with no later iteration to overwrite it).
    """
    H = 32
    cu, seq_map, _, _ = _build_layout([4, 4, 4, 4], ragged_tail=False)
    n_cp = cu.shape[0] - 1
    ht, mt, _ = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=state_v_first, fallback_pattern="all",
    )
    fallback = torch.ones(n_cp, H, dtype=torch.bool, device=DEVICE)
    for lo, hi in zip(seq_map.tolist()[:-1], seq_map.tolist()[1:]):
        fallback[hi - 1] = False

    h_card, _ = aggregate_card_state(
        ht, mt, fallback, seq_map, state_v_first=state_v_first, compute_m=compute_m,
    )
    _, ref_h, _ = _ref_scan(ht, mt, fallback, seq_map, state_v_first=state_v_first)
    # A non-fallback last segment means h_card is a plain copy of ht[last], so this is
    # exact -- but keep the loose bound so the failure mode reads as "O(1) wrong".
    err = _rel_max(h_card, ref_h)
    assert err <= SCAN_RTOL, f"h_card rel_max={err:.2e} > {SCAN_RTOL:g}"


@pytest.mark.gpu
@pytest.mark.cp_inter_intra
@pytest.mark.parametrize("reset_at", [0, 2, 3], ids=["first", "middle", "last"])
def test_aggregate_card_state_m_is_all_or_nothing(reset_at):
    """M is the transfer matrix for propagating a correction *across* a whole raw
    sequence, so it is either the full ordered product or exactly zero.

    A non-fallback segment clears the accumulator, and later segments multiply into it,
    so a single reset anywhere kills M permanently -- correctly, because a dead chunk
    means nothing from before the sequence can reach past it. This pins that behaviour
    down: without it, an implementation that resumed accumulating after a reset would
    still pass the reference comparison only if the reference had the same bug.
    """
    H = 2
    n_seg = 4
    cu, seq_map, _, _ = _build_layout([n_seg], ragged_tail=False)
    n_cp = cu.shape[0] - 1
    ht, mt, fallback = _make_scan_inputs(
        seq_map, n_cp, H, state_v_first=False, fallback_pattern="all",
    )

    # all fallback -> the full ordered product
    _, m_full = aggregate_card_state(
        ht, mt, fallback, seq_map, state_v_first=False, compute_m=True,
    )
    expected = torch.eye(K, dtype=torch.float64, device=DEVICE)
    for idx in range(n_cp):
        expected = mt[idx, 0].double() @ expected
    err = _rel_max(m_full[0, 0], expected)
    assert err <= SCAN_RTOL, f"full M product rel_max={err:.2e} > {SCAN_RTOL:g}"

    # one reset anywhere -> zero
    fallback_reset = fallback.clone()
    fallback_reset[reset_at] = False
    _, m_reset = aggregate_card_state(
        ht, mt, fallback_reset, seq_map, state_v_first=False, compute_m=True,
    )
    assert torch.count_nonzero(m_reset) == 0, (
        f"a non-fallback segment at {reset_at} must zero M, got "
        f"max|M|={m_reset.abs().max().item():.3e}"
    )


# ===========================================================================
# Coupling to the real context
# ===========================================================================
@pytest.mark.gpu
def test_warmup_kernels_accept_a_real_intra_context():
    """The synthetic layouts above are hand-built; this one comes from the production
    context builder, so a change in the ht_mask / cu_seqlens conventions shows up here
    rather than only in the e2e run."""
    import cp_common as C

    case = C.CPCase(layout="offset", tokens_per_card=4096, num_k_heads=4, num_v_heads=4)
    inp = C.make_inputs(case, 1, DEVICE)
    ctx = C.CP_MODES["intra"].make_ctx(
        inp.cu_g, num_v_heads=case.num_v_heads, force_intra_cp=True, is_bwd=False,
    )
    assert ctx.is_intra, "expected the forced intra split"

    cp_cu = ctx.intra_cp_cu_seqlens
    g = inp.g.float()
    n_h, n_bwd, f_fwd, f_bwd = get_warmup_chunks_bidi(
        g=g, cu_seqlens=cp_cu, ht_mask_fwd=ctx.ht_mask, ht_mask_bwd=ctx.ht_mask_bwd,
        chunk_size=C.CHUNK_SIZE,
    )
    r_h, r_bwd, r_f_fwd, r_f_bwd = _ref_warmup_bidi(g, cp_cu, ctx.ht_mask, ctx.ht_mask_bwd)
    assert torch.equal(n_h, r_h)
    assert torch.equal(n_bwd, r_bwd)
    assert torch.equal(f_fwd, r_f_fwd)
    assert torch.equal(f_bwd, r_f_bwd)

    # the counts must never exceed the segment's own chunk count
    seg_chunks = ((cp_cu[1:] - cp_cu[:-1] + C.CHUNK_SIZE - 1) // C.CHUNK_SIZE).unsqueeze(-1)
    assert torch.all(n_h <= seg_chunks), f"num_warmup_h exceeds segment length\n{n_h}"
    assert torch.all(n_bwd <= seg_chunks), f"num_warmup_bwd exceeds segment length\n{n_bwd}"
