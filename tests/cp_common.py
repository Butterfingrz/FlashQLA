# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Shared CP (context-parallel) scaffolding for tests **and** profiling.

One place that knows about CP modes, so nothing downstream has to re-derive them.
``build_cp_context`` already unifies the four modes behind ``(is_inter, is_intra)``;
this module mirrors that with a small registry plus a shared case matrix, input
generator and oracle:

* :data:`CP_MODES` -- ``none`` / ``intra`` / ``inter`` / ``inter_intra``, each knowing
  how to build its context, whether it needs a process group, its minimum world
  size, and which :mod:`cp_arch` capability gates it.
* :func:`case_matrix` -- base config + one-factor-at-a-time (OFAT) over the axes that
  actually change kernel behaviour, plus a few full-combination smoke cases. Token
  counts are **per card**, so the global length is ``world_size * tokens_per_card``
  and stays divisible by the world size (which inter-card CP requires) for W=3 too.
* :func:`run_once` / :func:`run_reference` -- a single forward+backward driver used for
  both the CP run and the non-CP full-sequence oracle.
* :func:`compare` -- the slicing rules for turning a local CP result into relative
  errors against the global reference.

Both ``tests/test_cp_*.py`` and ``profile/`` import from here; keep it free of pytest
imports so the profile scripts can use it standalone.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cp_arch import ARCH, caps, supports  # noqa: F401  (re-exported for convenience)

from flash_qla.ops.gated_delta_rule import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Tolerance for :func:`rel_max` (max-abs error / max-abs reference) on a bf16 CP run
# against the bf16 non-CP oracle. Both sides carry their own rounding error, so this
# has to clear the *sum* of the two. Measured on SM100 at tokens_per_card=2048, W=2,
# layout ``offset_ragged``: the oracle is 7.95e-3 from a float64 reference, the CP run
# 6.35e-3 (i.e. CP is the *more* accurate of the two), and they land on opposite sides
# of the same element -- so they differ by 1.10e-2 with nothing wrong. A genuine CP
# defect (a dropped warmup chunk, a stale boundary state) is orders of magnitude
# larger than this, so the extra headroom costs no real sensitivity.
RTOL = 2e-2
SEED = 1234
DTYPE = torch.bfloat16
HEAD_DIM_K = 128
HEAD_DIM_V = 128

# Distinct rendezvous ports per (mode, world_size) so consecutive spawns in one
# pytest session never collide on a port still in TIME_WAIT.
_BASE_PORT = 29540

# Sequence layouts are expressed in GRID units per card, so every interior boundary
# lands on a multiple of ``tokens_per_card / GRID`` and the total is exactly
# ``world_size * tokens_per_card`` for any world size.
GRID = 8


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CPMode:
    name: str
    is_inter: bool
    is_intra: bool
    #: ``cp_arch`` capability key that must be true for this mode to run at all.
    capability: str | None
    index: int

    @property
    def needs_dist(self) -> bool:
        return self.is_inter

    @property
    def min_world_size(self) -> int:
        return 2 if self.is_inter else 1

    def port(self, world_size: int) -> str:
        return str(_BASE_PORT + self.index * 8 + world_size)

    def supported(self) -> bool:
        return self.capability is None or supports(self.capability)

    def make_ctx(
        self,
        cu_seqlens: torch.Tensor,
        *,
        num_v_heads: int,
        group=None,
        force_intra_cp: bool = False,
        is_train: bool = False,
    ):
        """Build this mode's context. ``cu_seqlens`` is always the **global** varlen
        offsets; the inter split derives each card's local view from it."""

        if self.is_inter:
            assert group is not None, f"mode {self.name} needs a process group"
        return build_cp_context(
            cu_seqlens,
            enable_inter=self.is_inter,
            enable_intra=self.is_intra,
            group=group,
            num_v_heads=num_v_heads,
            chunk_size=CHUNK_SIZE,
            is_train=is_train,
            force_intra_cp=force_intra_cp,
        )


CP_MODES: dict[str, CPMode] = {
    m.name: m
    for m in [
        CPMode("none", is_inter=False, is_intra=False, capability=None, index=0),
        CPMode("intra", is_inter=False, is_intra=True, capability="intra", index=1),
        CPMode("inter", is_inter=True, is_intra=False, capability="inter", index=2),
        CPMode("inter_intra", is_inter=True, is_intra=True, capability="inter_intra", index=3),
    ]
}

ALL_MODES = list(CP_MODES)


def modes_for(world_size: int, *, supported_only: bool = True) -> list[str]:
    """Mode names runnable at ``world_size`` on this arch."""
    out = []
    for name, mode in CP_MODES.items():
        if world_size < mode.min_world_size:
            continue
        if supported_only and not mode.supported():
            continue
        out.append(name)
    return out


def skip_reason(mode_name: str, world_size: int, *, need_bwd: bool = True) -> str | None:
    """Human-readable reason this (mode, world_size) cannot run here, or ``None``."""
    mode = CP_MODES[mode_name]
    if not mode.supported():
        return f"{mode_name} CP is not supported on {ARCH}"
    if need_bwd and not supports("bwd"):
        return f"backward kernels are not implemented on {ARCH}"
    if world_size < mode.min_world_size:
        return f"{mode_name} CP needs world_size >= {mode.min_world_size}"
    if torch.cuda.device_count() < world_size:
        return f"needs >= {world_size} GPUs, found {torch.cuda.device_count()}"
    return None


# ---------------------------------------------------------------------------
# Sequence layouts
# ---------------------------------------------------------------------------
# Each entry maps (GRID, world_size) -> per-sequence lengths in GRID units.
# With W cards there are ``GRID * W`` units in total, and a card boundary sits
# every ``GRID`` units -- which is what makes the layouts interesting:
#   single    one sequence spanning every card
#   per_card  one sequence per card, every boundary exactly on a card edge
#   offset    first boundary strictly inside card 0
#   three     a short middle sequence wholly inside card 0 (non-boundary sequence)
#   tail      a short final sequence inside the last card
# A ``_ragged`` suffix nudges the interior boundaries off the chunk grid.
_UNIT_LAYOUTS = {
    "single": lambda g, W: [g * W],
    "per_card": lambda g, W: [g] * W,
    "offset": lambda g, W: [g // 2, g * W - g // 2],
    "three": lambda g, W: [g // 2, g // 4, g * W - g // 2 - g // 4],
    "tail": lambda g, W: [g * W - g // 4, g // 4],
}

_RAGGED_SUFFIX = "_ragged"
_RAGGED_NUDGE = 7


def layout_cu_seqlens(layout: str, world_size: int, tokens_per_card: int) -> list[int]:
    """Resolve a named layout into global ``cu_seqlens`` (a plain python list)."""
    assert tokens_per_card % GRID == 0, (
        f"tokens_per_card ({tokens_per_card}) must be a multiple of GRID ({GRID})"
    )
    ragged = layout.endswith(_RAGGED_SUFFIX)
    base = layout[: -len(_RAGGED_SUFFIX)] if ragged else layout
    units = _UNIT_LAYOUTS[base](GRID, world_size)
    unit = tokens_per_card // GRID

    cu = [0]
    for n in units:
        cu.append(cu[-1] + n * unit)
    if ragged:
        # Interior boundaries only: the total must stay world_size * tokens_per_card.
        for i in range(1, len(cu) - 1):
            cu[i] += _RAGGED_NUDGE
    assert cu[-1] == world_size * tokens_per_card
    assert all(cu[i] < cu[i + 1] for i in range(len(cu) - 1)), f"degenerate layout {layout}"
    return cu


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CPCase:
    """One CP correctness/profiling configuration.

    ``tokens_per_card`` is per card on purpose: the global sequence is
    ``world_size * tokens_per_card`` so the same case scales to W=2/3/4 while
    staying divisible by the world size.
    """

    layout: str = "single"
    tokens_per_card: int = 2048
    num_k_heads: int = 8
    num_v_heads: int = 8
    g_scale: float = 1.0 / 16
    state_v_first: bool = False
    use_h0: bool = False
    use_dht: bool = False
    #: Bypass the intra heuristic so the intra path is exercised deterministically.
    #: Ignored by modes without an intra split.
    force_intra_cp: bool = True
    #: Free-form note, kept out of the id.
    note: str = field(default="", compare=False)

    @property
    def id(self) -> str:
        flags = "".join(
            f
            for f, on in [
                ("vk", self.state_v_first),
                ("h0", self.use_h0),
                ("dht", self.use_dht),
                ("auto", not self.force_intra_cp),
            ]
            if on
        )
        flags = f"-{flags}" if flags else ""
        return (
            f"{self.layout}-tpc{self.tokens_per_card}"
            f"-Hk{self.num_k_heads}Hv{self.num_v_heads}-g{self.g_scale:g}{flags}"
        )

    def cu_seqlens(self, world_size: int) -> list[int]:
        return layout_cu_seqlens(self.layout, world_size, self.tokens_per_card)

    def total_tokens(self, world_size: int) -> int:
        return world_size * self.tokens_per_card


BASE_CASE = CPCase()

# One factor at a time off BASE_CASE: each entry isolates a single axis, so a failure
# points at one behaviour instead of a combination.
_OFAT = [
    # sequence layout / boundary alignment
    dict(layout="per_card"),
    dict(layout="offset"),
    dict(layout="three"),
    dict(layout="tail"),
    dict(layout="offset_ragged"),
    # GVA (Hv > Hk)
    dict(num_k_heads=8, num_v_heads=16),
    # head count low enough for the auto heuristic to like intra CP
    dict(num_k_heads=4, num_v_heads=4),
    # heavy decay -> frequent state resets inside a chunk
    dict(g_scale=1.0),
    # state layout
    dict(state_v_first=True),
    # initial state / terminal gradient
    dict(use_h0=True),
    dict(use_dht=True),
    # let the heuristic decide (at this size it usually says "no": covers the
    # degenerate (T,T)->(T,F) path for inter+intra and the plain non-CP path for intra)
    dict(force_intra_cp=False),
]

# A few full combinations, to catch interactions the OFAT sweep cannot see.
_COMBOS = [
    dict(layout="offset", num_v_heads=16, g_scale=1.0, state_v_first=True,
         use_h0=True, use_dht=True, note="all flags on, boundary inside a card"),
    dict(layout="three", state_v_first=True, use_h0=True, use_dht=True,
         note="short middle sequence + all state flags"),
    dict(layout="per_card", use_h0=True, use_dht=True, force_intra_cp=False,
         note="one sequence per card, heuristic-driven intra"),
    # Realistic length: long enough that the SM100 forward heuristic turns intra CP
    # on by itself, so the auto path is covered at the size it was tuned for.
    dict(layout="single", tokens_per_card=16384, num_k_heads=4, num_v_heads=4,
         force_intra_cp=False, note="long sequence, auto intra CP expected on"),
]


def case_matrix() -> list[CPCase]:
    """The mode-independent case matrix: base + OFAT + combos."""
    cases = [BASE_CASE]
    cases += [replace(BASE_CASE, **kw) for kw in _OFAT]
    cases += [replace(BASE_CASE, **kw) for kw in _COMBOS]
    return cases


def cases_for_mode(mode_name: str, world_size: int) -> list[CPCase]:
    """Case matrix specialised for one mode/world size, de-duplicated.

    ``force_intra_cp`` is meaningless without an intra split, and several layouts
    collapse onto each other at ``world_size == 1``; both would otherwise produce
    identical cases running twice.
    """
    mode = CP_MODES[mode_name]
    out: list[CPCase] = []
    seen: set[tuple] = set()
    for case in case_matrix():
        if not mode.is_intra:
            case = replace(case, force_intra_cp=False)
        key = (tuple(case.cu_seqlens(world_size)), case.num_k_heads, case.num_v_heads,
               case.g_scale, case.state_v_first, case.use_h0, case.use_dht,
               case.force_intra_cp)
        if key in seen:
            continue
        seen.add(key)
        out.append(case)
    return out


def find_case(mode_name: str, world_size: int, case_id: str) -> CPCase:
    for case in cases_for_mode(mode_name, world_size):
        if case.id == case_id:
            return case
    raise KeyError(
        f"no case {case_id!r} for mode={mode_name} world_size={world_size}; "
        f"available: {[c.id for c in cases_for_mode(mode_name, world_size)]}"
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class CPInputs:
    """Global (un-sliced, grad-free) inputs for one case at one world size."""

    case: CPCase
    world_size: int
    device: str
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    h0: torch.Tensor | None
    cu_list: list[int]
    cu_g: torch.Tensor
    scale: float

    @property
    def total_tokens(self) -> int:
        return self.cu_list[-1]

    @property
    def num_seqs(self) -> int:
        return len(self.cu_list) - 1

    def rank_bounds(self, rank: int) -> tuple[int, int]:
        part = self.total_tokens // self.world_size
        return rank * part, (rank + 1) * part

    def first_seq_of_rank(self, rank: int) -> int:
        """Index of the first global sequence overlapping this rank's slice."""
        lo, _ = self.rank_bounds(rank)
        return int(torch.searchsorted(self.cu_g[1:], lo, side="right").item())

    def slice_tensors(self, rank: int) -> dict[str, torch.Tensor]:
        lo, hi = self.rank_bounds(rank)
        return {
            name: getattr(self, name)[:, lo:hi]
            for name in ("q", "k", "v", "g", "beta")
        }

    def local_h0(self, rank: int, num_local_seqs: int) -> torch.Tensor | None:
        if self.h0 is None:
            return None
        start = self.first_seq_of_rank(rank)
        return self.h0[start: start + num_local_seqs]


def make_inputs(case: CPCase, world_size: int, device, seed: int = SEED) -> CPInputs:
    """Deterministic global inputs. No ``requires_grad``: :func:`run_once` clones
    leaves per run so the reference and the CP run start from identical values."""
    cu_list = case.cu_seqlens(world_size)
    T = cu_list[-1]
    Hk, Hv = case.num_k_heads, case.num_v_heads
    K, V = HEAD_DIM_K, HEAD_DIM_V
    dev = torch.device(device)

    torch.manual_seed(seed)
    q = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=DTYPE), p=2, dim=-1)
    k = F.normalize(torch.randn(1, T, Hk, K, device=dev, dtype=DTYPE), p=2, dim=-1)
    v = torch.randn(1, T, Hv, V, device=dev, dtype=DTYPE)
    beta = torch.randn(1, T, Hv, device=dev, dtype=torch.float32).sigmoid()
    g = F.logsigmoid(torch.randn(1, T, Hv, device=dev, dtype=torch.float32)) * case.g_scale

    h0 = None
    if case.use_h0:
        n_seqs = len(cu_list) - 1
        shape = (n_seqs, Hv, V, K) if case.state_v_first else (n_seqs, Hv, K, V)
        h0 = torch.randn(*shape, device=dev, dtype=torch.float32) * 0.01

    return CPInputs(
        case=case, world_size=world_size, device=str(dev),
        q=q, k=k, v=v, g=g, beta=beta, h0=h0,
        cu_list=cu_list,
        cu_g=torch.tensor(cu_list, device=dev, dtype=torch.int32),
        scale=K ** -0.5,
    )


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
GRAD_NAMES = ("dq", "dk", "dv", "dg", "db")
_GRAD_OF = {"dq": "q", "dk": "k", "dv": "v", "dg": "g", "db": "beta"}


@dataclass
class CPRunResult:
    o: torch.Tensor
    final_state: torch.Tensor | None
    grads: dict[str, torch.Tensor | None]


def run_once(
    tensors: dict[str, torch.Tensor],
    *,
    h0: torch.Tensor | None,
    scale: float,
    state_v_first: bool,
    use_dht: bool,
    need_grad: bool,
    cu_seqlens: torch.Tensor | None = None,
    cp_context=None,
    auto_cp: bool = True,
    force_intra_cp: bool = False,
) -> CPRunResult:
    """One forward (+backward) through the public API.

    ``cu_seqlens`` + ``auto_cp`` drives the non-CP / auto-intra path; ``cp_context``
    drives an explicitly built context. ``output_final_state`` is always on so the
    final state -- the quantity CP is most likely to get wrong -- is always checked;
    when ``use_dht`` is false it simply stays out of the loss.
    """
    leaves = {
        name: t.detach().clone().requires_grad_(need_grad)
        for name, t in tensors.items()
    }
    h0_leaf = None
    if h0 is not None:
        h0_leaf = h0.detach().clone().requires_grad_(need_grad)

    o, final_state = chunk_gated_delta_rule(
        leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
        scale=scale,
        initial_state=h0_leaf,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_v_first=state_v_first,
        auto_cp=auto_cp,
        cp_context=cp_context,
        force_intra_cp=force_intra_cp,
    )

    grads: dict[str, torch.Tensor | None] = {name: None for name in GRAD_NAMES}
    grads["dh0"] = None
    if need_grad:
        loss = o.float().sum()
        if use_dht and final_state is not None:
            loss = loss + final_state.float().sum()
        loss.backward()
        for gname, tname in _GRAD_OF.items():
            grads[gname] = leaves[tname].grad
        if h0_leaf is not None:
            grads["dh0"] = h0_leaf.grad

    return CPRunResult(o=o, final_state=final_state, grads=grads)


def run_reference(inp: CPInputs, *, need_grad: bool = True) -> CPRunResult:
    """The oracle: one GPU, the whole global sequence, CP disabled."""
    return run_once(
        {name: getattr(inp, name) for name in ("q", "k", "v", "g", "beta")},
        h0=inp.h0,
        scale=inp.scale,
        state_v_first=inp.case.state_v_first,
        use_dht=inp.case.use_dht,
        need_grad=need_grad,
        cu_seqlens=inp.cu_g,
        auto_cp=False,
    )


def run_mode(
    inp: CPInputs,
    mode_name: str,
    rank: int,
    *,
    group=None,
    need_grad: bool = True,
) -> tuple[CPRunResult, object]:
    """Run one rank's slice under ``mode_name``. Returns the result and the context
    (callers assert on ``ctx.is_intra`` / ``ctx.is_inter`` to confirm which path ran)."""
    mode = CP_MODES[mode_name]
    ctx = mode.make_ctx(
        inp.cu_g,
        num_v_heads=inp.case.num_v_heads,
        group=group,
        force_intra_cp=inp.case.force_intra_cp,
        is_train=need_grad,
    )
    local = inp.slice_tensors(rank) if mode.is_inter else {
        name: getattr(inp, name) for name in ("q", "k", "v", "g", "beta")
    }
    h0 = inp.local_h0(rank, ctx.num_seqs) if mode.is_inter else inp.h0
    result = run_once(
        local,
        h0=h0,
        scale=inp.scale,
        state_v_first=inp.case.state_v_first,
        use_dht=inp.case.use_dht,
        need_grad=need_grad,
        cp_context=ctx,
    )
    return result, ctx


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
#: Every quantity :func:`compare` reports, in a fixed order. The key set must not vary
#: with rank or case: :func:`all_reduce_ratios` packs it into a tensor, so ranks
#: disagreeing on the keys would deadlock the collective.
RATIO_KEYS = ("o", "final_state", "dq", "dk", "dv", "dg", "db", "dh0")

#: Placeholder for "this rank has nothing to compare here" -- e.g. the final state of a
#: sequence that continues onto the next card. Negative so a MAX reduction ignores it.
NA = -1.0


def rel_max(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Max-abs error normalised by the reference's max-abs magnitude.

    A near-zero reference does not hide a real error: the ratio blows up instead of
    collapsing to zero, so the assertion still fires.
    """
    err = (actual.float() - expected.float()).abs().max().item()
    ref = expected.float().abs().max().item()
    return err / (ref + 1e-30)


def compare(
    inp: CPInputs,
    local: CPRunResult,
    reference: CPRunResult,
    rank: int,
    *,
    num_local_seqs: int,
    is_inter: bool,
    need_grad: bool = True,
) -> dict[str, float]:
    """Relative errors of one rank's local result against the global reference.

    Encodes the three slicing rules CP correctness depends on:

    * token-indexed tensors (``o`` and all input grads) compare against the
      reference restricted to this rank's token range;
    * a sequence's ``final_state`` is only the true global one on the card where the
      sequence *ends* -- elsewhere it is a partial state, so it is skipped;
    * ``dh0`` for a sequence belongs to the card where the sequence *starts*.

    Always returns every key in :data:`RATIO_KEYS`, using :data:`NA` where this rank has
    nothing to compare.
    """
    lo, hi = inp.rank_bounds(rank) if is_inter else (0, inp.total_tokens)
    start_seq = inp.first_seq_of_rank(rank) if is_inter else 0
    ratios = {key: NA for key in RATIO_KEYS}
    ratios["o"] = rel_max(local.o, reference.o[:, lo:hi])

    def _per_seq(local_t, ref_t, owns_seq) -> float:
        out = NA
        for si in range(num_local_seqs):
            gs = start_seq + si
            if not owns_seq(gs):
                continue
            out = max(out, rel_max(local_t[si], ref_t[gs]))
        return out

    if local.final_state is not None and reference.final_state is not None:
        # A sequence's final state is only the true global one on the card where the
        # sequence ends; elsewhere it is a partial state.
        ratios["final_state"] = _per_seq(
            local.final_state, reference.final_state,
            lambda gs: lo < inp.cu_list[gs + 1] <= hi,
        )

    if not need_grad:
        return ratios

    for gname in GRAD_NAMES:
        lg, rg = local.grads.get(gname), reference.grads.get(gname)
        if lg is not None and rg is not None:
            ratios[gname] = rel_max(lg, rg[:, lo:hi])

    lg, rg = local.grads.get("dh0"), reference.grads.get("dh0")
    if lg is not None and rg is not None:
        # dh0 belongs to the card where the sequence starts.
        ratios["dh0"] = _per_seq(lg, rg, lambda gs: inp.cu_list[gs] >= lo)

    return ratios


def all_reduce_ratios(ratios: dict[str, float], device) -> dict[str, float]:
    """MAX-reduce a ratio dict across ranks so every rank reports the same verdict.

    Keys come from :data:`RATIO_KEYS`, not from the dict, so every rank packs the same
    shape even when they disagree about which quantities they own.
    """
    if not dist.is_initialized():
        return dict(ratios)
    t = torch.tensor([ratios.get(n, NA) for n in RATIO_KEYS],
                     device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return {n: t[i].item() for i, n in enumerate(RATIO_KEYS)}


def worst(ratios: dict[str, float]) -> tuple[str, float]:
    """Name and value of the largest real ratio (``(None, NA)`` if nothing applies)."""
    real = {k: v for k, v in ratios.items() if v >= 0.0}
    if not real:
        return None, NA
    name = max(real, key=real.__getitem__)
    return name, real[name]


def format_ratios(ratios: dict[str, float]) -> str:
    return " ".join(
        f"{k}={ratios[k]:.2e}" for k in RATIO_KEYS
        if k in ratios and ratios[k] >= 0.0
    )


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def init_distributed(rank: int, world_size: int, port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return dist.group.WORLD


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
