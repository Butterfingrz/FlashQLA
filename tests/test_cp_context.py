# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""L1: CP context invariants -- metadata only, no kernel launches.

These are the cheapest and most diagnostic CP tests: everything here is a property of
the index tensors ``build_cp_context`` produces, so a failure names a broken invariant
instead of a numerical mismatch 40 kernels later. They run in seconds and need no
multi-GPU spawn.

A GPU still has to be *present* (the context builders are ``@tensor_cache``d, which
asserts ``torch.cuda.is_available()``, and the index tensors live on the device), but
no compute kernel is compiled or launched.
"""
import os
import sys

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context
from flash_qla.ops.gated_delta_rule.chunk.cp.context import (
    _calc_inter_cp_seqs,
    _calc_intra_cp_seqs,
)
from flash_qla.ops.gated_delta_rule.chunk.cp.preprocess import _assert_inter_intra_supported

from cp_common import GRID, layout_cu_seqlens

DEV = "cuda:0"


# ===========================================================================
# helpers
# ===========================================================================
def _intra_ctx(cu_list, num_v_heads=8, is_bwd=False, force_intra_cp=True):
    cu = torch.tensor(cu_list, dtype=torch.int32, device=DEV)
    return build_cp_context(
        cu, enable_intra=True, num_v_heads=num_v_heads,
        chunk_size=CHUNK_SIZE, is_bwd=is_bwd, force_intra_cp=force_intra_cp,
    )


def _check_intra_invariants(ctx, cu_list):
    """Every structural property the intra kernels rely on, in one place."""
    assert ctx.is_intra, "expected an active intra split"
    n_raw = len(cu_list) - 1

    cp_cu = ctx.intra_cp_cu_seqlens.tolist()
    r2c = ctx.seq_map_r2c.tolist()
    c2r = ctx.seq_map_c2r.tolist()
    ht = ctx.ht_mask.tolist()
    ht_bwd = ctx.ht_mask_bwd.tolist()
    n_cp = len(cp_cu) - 1

    # --- shapes ---
    assert len(r2c) == n_raw + 1, f"seq_map_r2c should be per-raw-seq+1: {len(r2c)} vs {n_raw + 1}"
    assert len(c2r) == n_cp, f"seq_map_c2r should be per-cp-seq: {len(c2r)} vs {n_cp}"
    assert len(ht) == n_cp and len(ht_bwd) == n_cp

    # --- cp_cu refines cu: strictly increasing, same endpoints, superset of boundaries ---
    assert cp_cu[0] == cu_list[0] and cp_cu[-1] == cu_list[-1]
    assert all(cp_cu[i] < cp_cu[i + 1] for i in range(n_cp)), f"non-monotonic cp_cu: {cp_cu}"
    assert set(cu_list).issubset(set(cp_cu)), (
        f"raw boundaries {sorted(set(cu_list) - set(cp_cu))} missing from the cp partition"
    )

    # --- r2c / c2r are mutual inverses ---
    assert r2c[0] == 0 and r2c[-1] == n_cp
    assert all(r2c[i] < r2c[i + 1] for i in range(n_raw)), f"non-monotonic r2c: {r2c}"
    for i in range(n_raw):
        assert cp_cu[r2c[i]] == cu_list[i], (
            f"seq {i}: r2c points at token {cp_cu[r2c[i]]}, raw start is {cu_list[i]}"
        )
        for c in range(r2c[i], r2c[i + 1]):
            assert c2r[c] == i, f"cp seq {c} maps to raw {c2r[c]}, expected {i}"

    # --- ht_mask marks the last cp segment of each raw seq; ht_mask_bwd the first ---
    assert sum(ht) == n_raw, f"ht_mask has {sum(ht)} True, expected one per raw seq ({n_raw})"
    assert sum(ht_bwd) == n_raw, f"ht_mask_bwd has {sum(ht_bwd)} True, expected {n_raw}"
    for i in range(n_raw):
        assert ht[r2c[i + 1] - 1], f"seq {i}: ht_mask not set on its last cp segment"
        assert ht_bwd[r2c[i]], f"seq {i}: ht_mask_bwd not set on its first cp segment"

    # --- a split raw seq is cut into equal pieces (last one shorter or equal) ---
    for i in range(n_raw):
        seg = [cp_cu[c + 1] - cp_cu[c] for c in range(r2c[i], r2c[i + 1])]
        if len(seg) > 1:
            assert len(set(seg[:-1])) == 1, f"seq {i}: uneven leading segments {seg}"
            assert seg[-1] <= seg[0], f"seq {i}: trailing segment {seg[-1]} > {seg[0]}"
            step = seg[0]
            assert step % CHUNK_SIZE == 0, f"seq {i}: split step {step} not chunk-aligned"
            n_chunks = step // CHUNK_SIZE
            assert n_chunks & (n_chunks - 1) == 0, (
                f"seq {i}: split step is {n_chunks} chunks, expected a power of two"
            )


# ===========================================================================
# intra-card context
# ===========================================================================
_INTRA_LAYOUTS = [
    [0, 8192],
    [0, 2048, 8192],
    [0, 1024, 1536, 8192],
    [0, 4096, 8192],
    [0, 2048, 4096, 6144, 8192],
    [0, 1031, 8192],                       # ragged: boundary off the chunk grid
    [0, 64, 8192],                         # single-chunk leading sequence
    [0, 8192, 8256],                       # single-chunk trailing sequence
    [0, 100, 8192],                        # sub-chunk leading sequence
]


@pytest.mark.gpu
@pytest.mark.parametrize("cu_list", _INTRA_LAYOUTS, ids=lambda c: f"n{len(c) - 1}-T{c[-1]}")
@pytest.mark.parametrize("is_bwd", [False, True], ids=["fwd", "bwd"])
def test_intra_context_invariants(cu_list, is_bwd):
    ctx = _intra_ctx(cu_list, is_bwd=is_bwd, force_intra_cp=True)
    _check_intra_invariants(ctx, cu_list)
    assert not ctx.is_inter
    assert ctx.num_seqs == len(cu_list) - 1


@pytest.mark.gpu
def test_intra_context_degenerate_split_is_still_consistent():
    """``force_intra_cp`` on a sequence too short to split must yield one cp segment per
    raw sequence -- ``is_intra`` is on, but the partition is the identity."""
    cu_list = [0, CHUNK_SIZE, 3 * CHUNK_SIZE]
    ctx = _intra_ctx(cu_list, force_intra_cp=True)
    _check_intra_invariants(ctx, cu_list)
    assert ctx.intra_cp_cu_seqlens.tolist() == cu_list, "short sequences should not be split"
    assert ctx.seq_map_r2c.tolist() == list(range(len(cu_list)))


@pytest.mark.gpu
def test_force_intra_cp_tristate():
    """``force_intra_cp`` bypasses the heuristic; ``enable_intra=False`` still wins.

    Also pins the thing the old ``use_cp=True`` debug override hid: with the heuristic
    in charge, a configuration it dislikes really does come back with ``is_intra=False``
    and no intra tensors at all.
    """
    cu = torch.tensor([0, 4 * CHUNK_SIZE], dtype=torch.int32, device=DEV)
    kw = dict(num_v_heads=64, chunk_size=CHUNK_SIZE)

    forced = build_cp_context(cu, enable_intra=True, force_intra_cp=True, **kw)
    assert forced.is_intra
    assert forced.intra_cp_cu_seqlens is not None

    # 64 heads over 4 chunks: every arch threshold says no.
    auto = build_cp_context(cu, enable_intra=True, force_intra_cp=False, **kw)
    assert not auto.is_intra, "heuristic unexpectedly enabled intra CP for Be*H=64, 4 chunks"
    assert auto.intra_cp_cu_seqlens is None and auto.seq_map_r2c is None
    assert auto.ht_mask is None and auto.ht_mask_bwd is None
    assert auto.cu_seqlens is not None, "the raw cu_seqlens must survive a disabled split"

    # force_intra_cp is a no-op when intra is not enabled at all.
    off = build_cp_context(cu, enable_intra=False, force_intra_cp=True, **kw)
    assert not off.is_intra and not off.is_inter


@pytest.mark.gpu
@pytest.mark.parametrize("force", [False, True], ids=["auto", "forced"])
def test_copy_for_backward_carries_intra_fields(force):
    """``copy_for_backward`` must carry every field the backward preprocess reads, and
    must not invent intra tensors when the split is off."""
    ctx = _intra_ctx([0, 32768], num_v_heads=4, force_intra_cp=force)
    copied = ctx.copy_for_backward()

    assert copied.is_intra == ctx.is_intra and copied.is_inter == ctx.is_inter
    for name in ("cu_seqlens", "intra_cp_cu_seqlens", "seq_map_r2c", "seq_map_c2r",
                 "ht_mask", "ht_mask_bwd"):
        orig, copy = getattr(ctx, name), getattr(copied, name)
        if orig is None:
            assert copy is None, f"{name}: copy invented a tensor"
            continue
        assert copy is not None, f"{name}: dropped by copy_for_backward"
        assert copy is not orig, f"{name}: copy_for_backward returned the same tensor"
        assert torch.equal(copy, orig), f"{name}: copy differs from the original"

    if ctx.is_intra:
        _check_intra_invariants(copied, ctx.cu_seqlens.tolist())


@pytest.mark.gpu
def test_intra_context_shares_cache_across_calls():
    """The context builders are ``@tensor_cache``d on the *same* tensor object, which is
    what lets forward and backward reuse one partition instead of rebuilding it."""
    cu = torch.tensor([0, 32768], dtype=torch.int32, device=DEV)
    a = build_cp_context(cu, enable_intra=True, num_v_heads=4,
                         chunk_size=CHUNK_SIZE, force_intra_cp=True)
    b = build_cp_context(cu, enable_intra=True, num_v_heads=4,
                         chunk_size=CHUNK_SIZE, force_intra_cp=True)
    assert a is b, "identical arguments should hit the tensor_cache"


# ===========================================================================
# inter-card context
# ===========================================================================
def _inter_ctxs(cu_list, world_size):
    """Per-rank inter contexts, built without a process group (explicit rank/world)."""
    cu = torch.tensor(cu_list, dtype=torch.int32, device=DEV)
    return [
        _calc_inter_cp_seqs(cu, world_size=world_size, rank=r, group=None)
        for r in range(world_size)
    ]


@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 3, 4])
@pytest.mark.parametrize(
    "layout",
    ["single", "per_card", "offset", "three", "tail", "offset_ragged"],
)
def test_inter_partition_accounts_for_every_token(layout, world_size):
    """Union of the per-rank local partitions must reconstruct the global one."""
    cu_list = layout_cu_seqlens(layout, world_size, tokens_per_card=GRID * CHUNK_SIZE)
    total = cu_list[-1]
    part = total // world_size
    ctxs = _inter_ctxs(cu_list, world_size)

    boundaries = set()
    for rank, ctx in enumerate(ctxs):
        local = ctx.cu_seqlens_cpu.tolist()
        assert local[0] == 0 and local[-1] == part, (
            f"rank {rank}: local partition covers {local[-1]} tokens, expected {part}"
        )
        assert all(local[i] < local[i + 1] for i in range(len(local) - 1))
        boundaries.update(rank * part + b for b in local)

    # Every global sequence boundary is a boundary on the card that contains it, and the
    # only extra boundaries are the card edges themselves.
    card_edges = {r * part for r in range(world_size)} | {total}
    assert set(cu_list).issubset(boundaries | {total})
    assert boundaries.issubset(set(cu_list) | card_edges), (
        f"unexpected local boundaries: {sorted(boundaries - set(cu_list) - card_edges)}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_inter_topology_sequence_spans_all_cards(world_size):
    """One sequence over every card: rank r has r predecessors and W-1-r successors."""
    total = world_size * 1024
    ctxs = _inter_ctxs([0, total], world_size)
    for rank, ctx in enumerate(ctxs):
        assert ctx.is_inter and not ctx.is_intra
        assert ctx.num_seqs == 1
        assert ctx.pre_num_ranks == rank
        assert ctx.post_num_ranks == world_size - 1 - rank
        assert ctx.is_first_rank == (rank == 0)
        assert ctx.is_last_rank == (rank == world_size - 1)
        assert ctx.pre_num_conv_tokens == rank * (total // world_size)


@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_inter_topology_boundary_on_card_edge(world_size):
    """One sequence per card: no rank shares a sequence, so nothing to correct."""
    per_card = 1024
    cu_list = [i * per_card for i in range(world_size + 1)]
    for rank, ctx in enumerate(_inter_ctxs(cu_list, world_size)):
        assert ctx.num_seqs == 1
        assert ctx.pre_num_ranks == 0 and ctx.post_num_ranks == 0
        assert ctx.is_first_rank and ctx.is_last_rank
        assert ctx.pre_num_conv_tokens == 0


@pytest.mark.gpu
def test_inter_topology_sequence_inside_one_card():
    """A short sequence wholly inside card 0, plus a long one spanning both cards."""
    # card 0 = [0, 512), card 1 = [512, 1024)
    cu_list = [0, 128, 1024]
    c0, c1 = _inter_ctxs(cu_list, world_size=2)

    # rank 0 owns the whole first sequence and the head of the second.
    assert c0.cu_seqlens_cpu.tolist() == [0, 128, 512]
    assert c0.num_seqs == 2
    assert c0.is_first_rank, "rank 0 starts the sequence it shares with rank 1"
    assert not c0.is_last_rank
    assert c0.pre_num_ranks == 0 and c0.post_num_ranks == 1
    assert c0.pre_num_conv_tokens == 0

    # rank 1 owns only the tail of the second sequence.
    assert c1.cu_seqlens_cpu.tolist() == [0, 512]
    assert c1.num_seqs == 1
    assert not c1.is_first_rank and c1.is_last_rank
    assert c1.pre_num_ranks == 1 and c1.post_num_ranks == 0
    assert c1.pre_num_conv_tokens == 512 - 128, "offset of the card start within its sequence"


@pytest.mark.parametrize("total, world_size", [(1000, 3), (1000, 7), (100, 3)])
def test_inter_cp_indivisible_raises(total, world_size):
    assert total % world_size != 0, "test setup: total must be indivisible"
    cu = torch.tensor([0, total], dtype=torch.int32)
    with pytest.raises(AssertionError, match="divisible by"):
        _calc_inter_cp_seqs(cu, world_size=world_size, rank=0, group=None)


def test_inter_cp_divisible_ok():
    cu = torch.tensor([0, 1000], dtype=torch.int32)  # 1000 % 4 == 0
    ctx = _calc_inter_cp_seqs(cu, world_size=4, rank=0, group=None)
    assert ctx.is_inter and not ctx.is_intra
    assert ctx.num_seqs >= 1


# ===========================================================================
# combined inter+intra context
# ===========================================================================
class _StubGroup:
    """Stands in for a ``ProcessGroup`` while building a context.

    Building a combined (inter, intra) context runs no collectives -- it only asks the
    group for its size and rank -- so a stub lets this L1 module cover the merge path at
    world sizes 2/3/4 in a single process, with no spawn and no network interface
    assumptions. The real multi-process behaviour is covered by ``test_cp_e2e.py``.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank


@pytest.fixture
def stub_group(monkeypatch):
    def make(world_size: int, rank: int) -> _StubGroup:
        group = _StubGroup(world_size, rank)
        real_ws, real_rank = dist.get_world_size, dist.get_rank
        monkeypatch.setattr(
            dist, "get_world_size",
            lambda group=None: group.world_size if isinstance(group, _StubGroup) else real_ws(group),
        )
        monkeypatch.setattr(
            dist, "get_rank",
            lambda group=None: group.rank if isinstance(group, _StubGroup) else real_rank(group),
        )
        return group

    return make


@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_inter_intra_merge_carries_both_field_groups(stub_group, world_size):
    """The merged context must expose the inter topology *and* an intra partition built
    over the card-local sequence (not the global one)."""
    total = world_size * 32768
    cu = torch.tensor([0, total], dtype=torch.int32, device=DEV)

    for rank in range(world_size):
        group = stub_group(world_size, rank)
        ctx = build_cp_context(
            cu, enable_inter=True, enable_intra=True, group=group,
            num_v_heads=4, chunk_size=CHUNK_SIZE, force_intra_cp=True,
        )
        assert ctx.is_inter and ctx.is_intra

        # inter fields
        assert ctx.group is group
        assert ctx.cu_seqlens_cpu.tolist() == [0, total // world_size]
        assert ctx.pre_num_ranks == rank
        assert ctx.post_num_ranks == world_size - 1 - rank
        assert ctx.is_first_rank == (rank == 0)
        assert ctx.is_last_rank == (rank == world_size - 1)

        # intra fields, over the *local* (post-inter-split) sequence
        local_cu = ctx.cu_seqlens.tolist()
        assert local_cu[-1] == total // world_size, "intra split must refine the local view"
        _check_intra_invariants(ctx, local_cu)

        copied = ctx.copy_for_backward()
        assert copied.is_inter and copied.is_intra
        assert copied.group is group
        assert copied.pre_num_ranks == ctx.pre_num_ranks
        assert copied.post_num_ranks == ctx.post_num_ranks
        assert copied.is_first_rank == ctx.is_first_rank
        assert copied.is_last_rank == ctx.is_last_rank
        _check_intra_invariants(copied, local_cu)


@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_inter_intra_degenerates_to_pure_inter(stub_group, world_size):
    """When the intra heuristic declines, an inter+intra context must come back as
    ``(is_inter=True, is_intra=False)`` -- i.e. behave exactly like pure inter."""
    cu = torch.tensor([0, world_size * 4 * CHUNK_SIZE], dtype=torch.int32, device=DEV)
    ctx = build_cp_context(
        cu, enable_inter=True, enable_intra=True, group=stub_group(world_size, 0),
        num_v_heads=64, chunk_size=CHUNK_SIZE, force_intra_cp=False,
    )
    assert ctx.is_inter and not ctx.is_intra
    assert ctx.intra_cp_cu_seqlens is None and ctx.seq_map_r2c is None
    assert ctx.cu_seqlens_cpu is not None, "inter fields must survive the degeneration"
    assert ctx.pre_num_ranks == 0 and ctx.post_num_ranks == world_size - 1


# ===========================================================================
# arch capability guard
# ===========================================================================
@pytest.mark.parametrize(
    "is_inter, is_intra",
    [(True, False), (False, True), (False, False)],
    ids=["inter", "intra", "none"],
)
def test_inter_intra_guard_allows_single_mode_without_kernel(is_inter, is_intra):
    """Only the *combined* mode needs ``aggregate_card_state``."""
    _assert_inter_intra_supported(is_inter, is_intra, None)


def test_inter_intra_guard_rejects_missing_kernel():
    with pytest.raises(NotImplementedError, match="aggregate_card_state"):
        _assert_inter_intra_supported(True, True, None)


def test_inter_intra_guard_accepts_present_kernel():
    _assert_inter_intra_supported(True, True, object())
