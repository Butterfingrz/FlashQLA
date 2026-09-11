# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""One-layer CP profiling across every mode: none / intra / inter / inter_intra.

Each mode is timed twice -- stage by stage (``cp_stages.cp_fwd`` / ``cp_bwd``, so the
cost lands on named stages) and end-to-end through the public autograd API -- and the
results are printed as one table per direction with the modes as columns, so a stage's
cost is directly comparable across modes.

Two scaling protocols, because "how much does CP cost" has two different answers:

``strong``
    ``--seqlen`` is the **global** sequence; the inter modes get ``T/W`` tokens per
    card, and the single-card modes (``none`` / ``intra``) run the whole ``T`` on one
    card. This is the strong-scaling view: inter-vs-none is the end-to-end speedup.
``overhead``
    ``--seqlen`` is the **per-card** sequence (global ``T = seqlen * W``); the
    single-card modes run the same rank-local slice the inter modes do. Same work per
    card in every column, so inter-vs-none is exactly the CP overhead.

``--scaling both`` runs them in sequence. Every run ends with a parity step: the
hand-written stage decomposition in ``cp_stages.py`` is re-run untimed and diffed
against the public API, which is what keeps the transcription honest (see the module
docstring there). ``--no-check`` skips it, ``--check-only`` skips the timing.

Usage (single card)::

    python profile/profile_cp.py --seqlen 32768 --nvh 16

Usage (multi card)::

    torchrun --nproc_per_node=4 profile/profile_cp.py --seqlen 32768 --nvh 16
    torchrun --nproc_per_node=4 profile/profile_cp.py --scaling overhead --fla
    torchrun --nproc_per_node=4 profile/profile_cp.py --set cp
    torchrun --nproc_per_node=4 profile/profile_cp.py --nsys --modes inter_intra
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "profile"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))

import cp_common as C
from utils import CudaTimer
from cp_stages import (
    FWD_STAGES, BWD_STAGES, FwdArtifacts, bwd_available,
    cp_fwd, cp_bwd, null_regions, timer_regions, nvtx_regions, nvtx_range,
)
from flash_qla import chunk_gated_delta_rule
from flash_qla.utils import l2norm
from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE

FLA_LATEST_PATH = os.environ.get("FLA_LATEST", "/cpfs02/user/cenxi.lx/code/fla-latest")

#: How far the hand-written stages may drift from the public API before we call it a bug.
PARITY_RTOL = 1e-3

#: FLA baselines. FLA has no intra-card split, so there are exactly two: a naive
#: single-card run and its own inter-card CP.
FLA_MODES = ("fla_naive", "fla_inter")


# ---------------------------------------------------------------------------
# Inputs
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
    """Global (all-card) inputs. ``swa_ratio`` of the value heads get ``g == 0``, which
    is what a sliding-window head looks like to these kernels -- and the case the
    warmup heuristics care about, since a zero decay never lets a chunk fall back."""
    dev = f"cuda:{torch.cuda.current_device()}"
    dtype = getattr(torch, data_dtype)
    torch.manual_seed(random_seed)

    q = l2norm(torch.randn(1, num_tokens, num_k_heads, head_dim_k, device=dev, dtype=dtype))
    k = l2norm(torch.randn(1, num_tokens, num_k_heads, head_dim_k, device=dev, dtype=dtype))
    v = torch.randn(1, num_tokens, num_v_heads, head_dim_v, device=dev, dtype=dtype)
    g = F.logsigmoid(
        torch.randn(1, num_tokens, num_v_heads, device=dev, dtype=torch.float32)
    ) / 16
    beta = torch.randn(1, num_tokens, num_v_heads, device=dev, dtype=torch.float32).sigmoid()

    swa = torch.zeros(num_v_heads, dtype=torch.bool, device=dev)
    swa[:math.ceil(swa_ratio * num_v_heads)] = 1
    swa = swa[torch.randperm(num_v_heads, device=dev)]
    g[:, :, ~swa] = 0.0

    cu_g = torch.tensor(cu_seqlens or [0, num_tokens], device=dev, dtype=torch.int32)
    return dict(q=q, k=k, v=v, g=g, beta=beta, cu_g=cu_g, scale=head_dim_k ** -0.5)


def slice_inputs(inputs: dict, lo: int, hi: int) -> dict:
    return {name: inputs[name][:, lo:hi] for name in ("q", "k", "v", "g", "beta")}


def whole_inputs(inputs: dict) -> dict:
    return {name: inputs[name] for name in ("q", "k", "v", "g", "beta")}


# ---------------------------------------------------------------------------
# One (mode, protocol) view of the data
# ---------------------------------------------------------------------------
class ModeRun:
    """The tensors, cu_seqlens and context one mode is profiled on.

    Which slice a mode sees depends on the protocol, so this is where the two scaling
    interpretations actually differ -- everything downstream is mode-agnostic.
    """

    def __init__(self, mode_name, inputs, *, rank, world_size, protocol, num_v_heads):
        self.name = mode_name
        self.mode = C.CP_MODES[mode_name]
        self.scale = inputs["scale"]

        part = (inputs["cu_g"][-1].item()) // world_size
        lo, hi = rank * part, (rank + 1) * part

        if self.mode.is_inter:
            # The context does the rank-local rebasing itself.
            self.ctx = self.mode.make_ctx(
                inputs["cu_g"], num_v_heads=num_v_heads,
                group=dist.group.WORLD, force_intra_cp=True,
            )
            self.d = slice_inputs(inputs, lo, hi)
        elif protocol == "overhead":
            # Same tokens the inter modes see, so the columns differ only by CP work.
            local_cu = _rebase_cu(inputs["cu_g"], lo, hi)
            self.ctx = self.mode.make_ctx(
                local_cu, num_v_heads=num_v_heads, force_intra_cp=True,
            )
            self.d = slice_inputs(inputs, lo, hi)
        else:  # strong scaling: the single-card reference is the *whole* sequence
            self.ctx = self.mode.make_ctx(
                inputs["cu_g"], num_v_heads=num_v_heads, force_intra_cp=True,
            )
            self.d = whole_inputs(inputs)

        self.tokens = self.d["q"].shape[1]

    @property
    def label(self) -> str:
        """Mode name plus the path the context actually took -- the intra heuristic can
        decline, and a silently-degenerate column would be misread as a real one."""
        flags = f"{'I' if self.ctx.is_inter else '-'}{'i' if self.ctx.is_intra else '-'}"
        return f"{self.name}[{flags}]"


def _rebase_cu(cu_g: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Global cu_seqlens restricted to ``[lo, hi)`` and shifted to start at 0."""
    cu = cu_g.tolist()
    out = [0]
    for i in range(len(cu) - 1):
        s, e = max(cu[i], lo), min(cu[i + 1], hi)
        if e > s:
            out.append(out[-1] + (e - s))
    return torch.tensor(out, device=cu_g.device, dtype=cu_g.dtype)


# ---------------------------------------------------------------------------
# Timed passes
# ---------------------------------------------------------------------------
def _leaves(d: dict) -> dict:
    return {n: t.detach().clone().requires_grad_(True) for n, t in d.items()}


def time_mode(timer: CudaTimer, run: ModeRun, *, need_bwd: bool):
    """Stage-decomposed and end-to-end timings for one mode, into ``timer``'s tags."""
    ctx, d, scale, name = run.ctx, run.d, run.scale, run.name

    timer.bench(lambda t: cp_fwd(d, ctx, scale, timer_regions(t, name)), timer)

    def e2e_fwd(t):
        with t.mark(f"{name}::e2e_fwd"):
            chunk_gated_delta_rule(
                d["q"], d["k"], d["v"], d["g"], d["beta"],
                scale=scale, output_final_state=False, cp_context=ctx,
            )

    timer.bench(e2e_fwd, timer)

    if not need_bwd:
        return

    def stages_bwd(t):
        # The forward has to happen for the backward to have inputs, but it is not part
        # of what we are measuring -- only the marked regions are accumulated.
        fwd = cp_fwd(d, ctx, scale, null_regions())
        cp_bwd(
            d, ctx, scale, timer_regions(t, name, bwd=True), fwd,
            do=torch.ones_like(fwd.o),
        )

    timer.bench(stages_bwd, timer)

    def e2e_bwd(t):
        leaves = _leaves(d)
        o, _ = chunk_gated_delta_rule(
            leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
            scale=scale, output_final_state=False, cp_context=ctx,
        )
        with t.mark(f"{name}::e2e_bwd"):
            o.sum().backward()

    timer.bench(e2e_bwd, timer)


def time_fla(timer: CudaTimer, inputs: dict, *, rank, world_size, protocol,
             need_bwd: bool):
    """FLA baselines: naive single-card, and FLA's own inter-card CP."""
    if FLA_LATEST_PATH and FLA_LATEST_PATH not in sys.path:
        sys.path.insert(0, FLA_LATEST_PATH)
    import fla as _fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as chunk_gdn_fla

    if rank == 0:
        print(f"[FLA] using fla from: {_fla.__file__}", flush=True)

    scale = inputs["scale"]
    part = inputs["cu_g"][-1].item() // world_size
    lo, hi = rank * part, (rank + 1) * part
    local = slice_inputs(inputs, lo, hi)

    variants = [("fla_naive", local if protocol == "overhead" else whole_inputs(inputs), None)]
    if world_size > 1:
        from fla.ops.cp import build_cp_context as build_cp_context_fla
        variants.append(
            ("fla_inter", local, build_cp_context_fla(inputs["cu_g"], group=dist.group.WORLD))
        )

    for name, d, ctx_fla in variants:
        kwargs = {} if ctx_fla is None else dict(cp_context=ctx_fla)

        def fwd(t, d=d, name=name, kwargs=kwargs):
            with t.mark(f"{name}::e2e_fwd"):
                chunk_gdn_fla(d["q"], d["k"], d["v"], d["g"], d["beta"],
                              scale=scale, **kwargs)

        timer.bench(fwd, timer)

        if not need_bwd:
            continue

        def bwd(t, d=d, name=name, kwargs=kwargs):
            leaves = _leaves(d)
            o, _ = chunk_gdn_fla(
                leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
                scale=scale, **kwargs,
            )
            with t.mark(f"{name}::e2e_bwd"):
                o.sum().backward()

        timer.bench(bwd, timer)


# ---------------------------------------------------------------------------
# Parity: hand-written stages vs the production path
# ---------------------------------------------------------------------------
def _rel_max(actual, expected) -> float:
    if actual is None or expected is None:
        return float("nan")
    a, e = actual.float(), expected.float()
    denom = e.abs().max().clamp_min(1e-12)
    return ((a - e).abs().max() / denom).item()


def check_parity(run: ModeRun, *, need_bwd: bool) -> dict[str, float]:
    """Re-run the stage decomposition untimed and diff it against the public API.

    This is the whole reason it is safe to keep a hand-written copy of the CP
    composition outside the production code: if ``preprocess.py`` changes and
    ``cp_stages.py`` does not, this fails.
    """
    ctx, d, scale = run.ctx, run.d, run.scale

    fwd: FwdArtifacts = cp_fwd(d, ctx, scale, null_regions())
    do = torch.ones_like(fwd.o)
    grads = cp_bwd(d, ctx, scale, null_regions(), fwd, do=do) if need_bwd else {}

    leaves = _leaves(d)
    o_ref, _ = chunk_gated_delta_rule(
        leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["beta"],
        scale=scale, output_final_state=False, cp_context=ctx,
    )
    errs = {"o": _rel_max(fwd.o, o_ref)}
    if need_bwd:
        o_ref.sum().backward()
        for key, leaf in (("dq", "q"), ("dk", "k"), ("dv", "v"),
                          ("dg", "g"), ("db", "beta")):
            errs[key] = _rel_max(grads[key], leaves[leaf].grad)
    return errs


# ---------------------------------------------------------------------------
# nsys mode
# ---------------------------------------------------------------------------
def nsys_run(runs: list[ModeRun], *, warmup: int, rep: int, need_bwd: bool,
             use_cuda_graph: bool):
    """NVTX-annotated passes, no timing -- for ``nsys profile``."""
    passes = []
    for run in runs:
        ctx, d, scale, name = run.ctx, run.d, run.scale, run.name

        def stages(ctx=ctx, d=d, scale=scale, name=name):
            with nvtx_range(f"{name}/stages"):
                fwd = cp_fwd(d, ctx, scale, nvtx_regions(name))
                if need_bwd:
                    cp_bwd(d, ctx, scale, nvtx_regions(name, bwd=True), fwd,
                           do=torch.ones_like(fwd.o))

        def e2e(ctx=ctx, d=d, scale=scale, name=name):
            with nvtx_range(f"{name}/e2e"):
                leaves = _leaves(d) if need_bwd else d
                with nvtx_range("fwd"):
                    o, _ = chunk_gated_delta_rule(
                        leaves["q"], leaves["k"], leaves["v"], leaves["g"],
                        leaves["beta"], scale=scale, output_final_state=False,
                        cp_context=ctx,
                    )
                if need_bwd:
                    with nvtx_range("bwd"):
                        o.sum().backward()

        passes += [stages, e2e]

    if use_cuda_graph:
        graphs = []
        for fn in passes:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs.append(graph)
        torch.cuda.synchronize()
        for i in range(rep):
            with nvtx_range(f"iter_{i}"):
                for graph in graphs:
                    graph.replay()
    else:
        for _ in range(warmup):
            for fn in passes:
                fn()
        torch.cuda.synchronize()
        for i in range(rep):
            with nvtx_range(f"iter_{i}"):
                for fn in passes:
                    fn()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _stage_table(report: dict, runs: list[ModeRun], stages, *, bwd: bool):
    import pandas as pd

    prefix = "b_" if bwd else ""
    data = {}
    for run in runs:
        col = {}
        for stage in stages:
            col[stage] = report.get(f"{run.name}::{prefix}{stage}", float("nan"))
        col["stage_sum"] = sum(v for v in col.values() if v == v)
        col["e2e"] = report.get(f"{run.name}::e2e_{'bwd' if bwd else 'fwd'}", float("nan"))
        data[run.label] = col
    return pd.DataFrame(data)[[r.label for r in runs]]


def _summary_table(report: dict, runs: list[ModeRun], *, use_fla: bool, need_bwd: bool):
    import pandas as pd

    rows = {}
    names = [(r.name, r.label, r.tokens) for r in runs]
    if use_fla:
        names += [(n, n, None) for n in FLA_MODES if f"{n}::e2e_fwd" in report]
    for name, label, tokens in names:
        fwd = report.get(f"{name}::e2e_fwd", float("nan"))
        bwd = report.get(f"{name}::e2e_bwd", float("nan")) if need_bwd else float("nan")
        rows[label] = {
            "tokens/card": tokens if tokens is not None else float("nan"),
            "fwd_ms": fwd,
            "bwd_ms": bwd,
            "fwd+bwd_ms": fwd + bwd,
        }
    df = pd.DataFrame(rows).T
    base = rows.get(runs[0].label, {}).get("fwd_ms", float("nan"))
    df["fwd_vs_" + runs[0].label] = df["fwd_ms"] / base
    return df


def report_run(report: dict, runs: list[ModeRun], *, use_fla: bool, need_bwd: bool,
               header: str):
    import pandas as pd

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:9.4f}")
    print(f"\n===== {header} =====")
    print("      NaN = the stage does not exist in that mode. `stage_sum` can exceed "
          "`e2e`: every\n      stage boundary is a CUDA-event pair, so work the "
          "uninstrumented path overlaps gets\n      billed to whichever stage waits "
          "for it. `e2e` is the number to quote.")
    print("\n[Forward] per-stage ms")
    print(_stage_table(report, runs, FWD_STAGES, bwd=False))
    if need_bwd:
        print("\n[Backward] per-stage ms")
        print(_stage_table(report, runs, BWD_STAGES, bwd=True))
    print("\n[Summary] end-to-end ms")
    print(_summary_table(report, runs, use_fla=use_fla, need_bwd=need_bwd))


def report_parity(errs_by_mode: dict[str, dict[str, float]]) -> bool:
    """Print the stage-vs-production diff. Returns True when everything is in bounds."""
    import pandas as pd

    df = pd.DataFrame(errs_by_mode).T
    print(f"\n[Parity] cp_stages.py vs chunk_gated_delta_rule, "
          f"max relative error (rtol={PARITY_RTOL:g})")
    with pd.option_context("display.float_format", lambda x: f"{x:.2e}"):
        print(df)
    bad = {m: e for m, e in errs_by_mode.items()
           if any(v > PARITY_RTOL for v in e.values() if v == v)}
    if bad:
        print(f"[Parity] FAIL: {', '.join(sorted(bad))} -- the hand-written stage "
              f"decomposition no longer matches production")
    else:
        print("[Parity] OK")
    return not bad


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def profile_one(
    *,
    rank: int,
    world_size: int,
    timer: CudaTimer,
    mode_names: list[str],
    protocol: str,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    use_fla: bool,
    need_bwd: bool,
    do_time: bool,
    do_check: bool,
    data_dtype: str = "bfloat16",
    swa_ratio: float = 0.75,
    random_seed: int = 42,
    cu_seqlens: list[int] | None = None,
) -> bool:
    """Profile one configuration under one protocol. Returns the parity verdict."""
    global_tokens = num_tokens if protocol == "strong" else num_tokens * world_size
    if cu_seqlens is not None:
        global_tokens = cu_seqlens[-1]
    assert global_tokens % world_size == 0, (
        f"global T={global_tokens} is not divisible by world_size={world_size}"
    )

    inputs = generate_inputs(
        num_tokens=global_tokens, num_k_heads=num_k_heads, num_v_heads=num_v_heads,
        data_dtype=data_dtype, swa_ratio=swa_ratio, random_seed=random_seed,
        cu_seqlens=cu_seqlens,
    )
    runs = [
        ModeRun(name, inputs, rank=rank, world_size=world_size,
                protocol=protocol, num_v_heads=num_v_heads)
        for name in mode_names
    ]

    header = (
        f"{protocol} | global T={global_tokens} W={world_size} "
        f"Hk={num_k_heads} Hv={num_v_heads} chunk={CHUNK_SIZE} "
        f"cu={inputs['cu_g'].tolist() if len(inputs['cu_g']) <= 6 else len(inputs['cu_g']) - 1}"
    )

    if do_time:
        timer.reset()
        for run in runs:
            time_mode(timer, run, need_bwd=need_bwd)
        if use_fla:
            time_fla(timer, inputs, rank=rank, world_size=world_size,
                     protocol=protocol, need_bwd=need_bwd)
        if rank == 0:
            report_run(timer.report(), runs, use_fla=use_fla, need_bwd=need_bwd,
                       header=header)

    ok = True
    if do_check:
        errs = {run.label: check_parity(run, need_bwd=need_bwd) for run in runs}
        if world_size > 1:
            errs = _all_reduce_errs(errs, inputs["q"].device)
        if rank == 0:
            ok = report_parity(errs)
    return ok


def _all_reduce_errs(errs: dict[str, dict[str, float]], device):
    """Worst error across ranks -- a rank-0-only report would miss a boundary bug on
    the last card. The key order is identical on every rank by construction."""
    keys = [(m, k) for m in sorted(errs) for k in sorted(errs[m])]
    buf = torch.tensor([errs[m][k] for m, k in keys], device=device, dtype=torch.float64)
    buf = torch.nan_to_num(buf, nan=-1.0)
    dist.all_reduce(buf, op=dist.ReduceOp.MAX)
    out = {m: {} for m in errs}
    for (m, k), value in zip(keys, buf.tolist()):
        out[m][k] = float("nan") if value < 0 else value
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Profile FlashQLA CP: none / intra / inter / inter_intra")
    parser.add_argument("--set", type=str, default=None,
                        help="preset name, loads profile/settings/{set}.csv")
    parser.add_argument("--modes", type=str, default=None,
                        help="comma-separated subset of "
                             "none,intra,inter,inter_intra (default: all runnable)")
    parser.add_argument("--scaling", choices=["strong", "overhead", "both"],
                        default="strong",
                        help="strong: --seqlen is global T. overhead: --seqlen is "
                             "per-card T and single-card modes use the same slice.")
    parser.add_argument("--seqlen", "--num-tokens", type=int, default=32768)
    parser.add_argument("--nvh", "--num-v-heads", type=int, default=16)
    parser.add_argument("--nkh", "--num-k-heads", type=int, default=0,
                        help="0 = same as --nvh")
    parser.add_argument("--cu-seqlens", type=str, default=None,
                        help="global cu_seqlens, e.g. 0-8192-32768 (overrides --seqlen)")
    parser.add_argument("--data-dtype", type=str, default="bfloat16")
    parser.add_argument("--swa-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fwd-only", action="store_true",
                        help="skip every backward measurement")
    parser.add_argument("--fla", action="store_true", help="add the FLA baselines")
    parser.add_argument("--nsys", action="store_true",
                        help="NVTX annotation only, no timing (for nsys profile)")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="capture+replay each pass as a CUDA graph (with --nsys)")
    parser.add_argument("--no-check", action="store_true",
                        help="skip the cp_stages-vs-production parity step")
    parser.add_argument("--check-only", action="store_true",
                        help="run only the parity step")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    if args.nkh <= 0:
        args.nkh = args.nvh

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", dist.get_rank())))
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        torch.cuda.set_device(0)
        rank, world_size = 0, 1

    need_bwd = not args.fwd_only and bwd_available()
    if args.fwd_only is False and not bwd_available() and rank == 0:
        print("[warn] no backward kernels on this arch -- forward only")

    if args.modes:
        mode_names = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        mode_names = C.modes_for(world_size)
    skipped = []
    kept = []
    for name in mode_names:
        reason = C.skip_reason(name, world_size, need_bwd=need_bwd)
        (kept if reason is None else skipped).append(name if reason is None else (name, reason))
    if rank == 0:
        for name, reason in skipped:
            print(f"[skip] {name}: {reason}")
    if not kept:
        if rank == 0:
            print("[profile_cp] nothing to run")
        if distributed:
            dist.destroy_process_group()
        return
    mode_names = kept

    protocols = ["strong", "overhead"] if args.scaling == "both" else [args.scaling]
    cu = [int(x) for x in args.cu_seqlens.split("-")] if args.cu_seqlens else None

    if args.nsys:
        inputs = generate_inputs(
            num_tokens=(cu[-1] if cu else args.seqlen), num_k_heads=args.nkh,
            num_v_heads=args.nvh, data_dtype=args.data_dtype,
            swa_ratio=args.swa_ratio, random_seed=args.seed, cu_seqlens=cu,
        )
        runs = [
            ModeRun(name, inputs, rank=rank, world_size=world_size,
                    protocol=protocols[0], num_v_heads=args.nvh)
            for name in mode_names
        ]
        if rank == 0:
            print(f"[nsys] modes={[r.label for r in runs]} W={world_size} "
                  f"warmup={args.warmup} rep={args.rep} graph={args.cuda_graph}")
        nsys_run(runs, warmup=args.warmup, rep=args.rep, need_bwd=need_bwd,
                 use_cuda_graph=args.cuda_graph)
        if rank == 0:
            print("[nsys] done")
        if distributed:
            dist.destroy_process_group()
        return

    timer = CudaTimer(warmup=args.warmup, rep=args.rep)
    configs = _load_configs(args)
    all_ok = True
    for protocol in protocols:
        for cfg in configs:
            torch.cuda.empty_cache()
            if rank == 0:
                print("-" * 100)
            all_ok &= profile_one(
                rank=rank, world_size=world_size, timer=timer,
                mode_names=mode_names, protocol=protocol,
                use_fla=args.fla, need_bwd=need_bwd,
                do_time=not args.check_only, do_check=not args.no_check,
                **cfg,
            )
    if rank == 0:
        print("-" * 100)
    if distributed:
        dist.destroy_process_group()
    if not all_ok:
        sys.exit(1)


def _load_configs(args) -> list[dict]:
    """One dict of ``generate_inputs`` knobs per configuration to profile."""
    base = dict(
        num_tokens=args.seqlen, num_k_heads=args.nkh, num_v_heads=args.nvh,
        data_dtype=args.data_dtype, swa_ratio=args.swa_ratio,
        random_seed=args.seed,
        cu_seqlens=[int(x) for x in args.cu_seqlens.split("-")] if args.cu_seqlens else None,
    )
    if args.set is None:
        return [base]

    import pandas as pd

    preset = pd.read_csv(
        os.path.join(PROJECT_ROOT, "profile", "settings", f"{args.set}.csv"))
    configs = []
    for _, row in preset.iterrows():
        data = row.to_dict()
        cfg = dict(base)
        if isinstance(data.get("cu_seqlens"), str):
            cfg["cu_seqlens"] = [int(x) for x in data["cu_seqlens"].split("-")]
        for key, cast in (("num_tokens", int), ("num_k_heads", int),
                          ("num_v_heads", int), ("data_dtype", str),
                          ("swa_ratio", float), ("random_seed", int)):
            if key in data and data[key] == data[key]:  # not NaN
                cfg[key] = cast(data[key])
        configs.append(cfg)
    return configs


if __name__ == "__main__":
    main()
