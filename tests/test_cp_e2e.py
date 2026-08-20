# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""L3: end-to-end CP correctness through the public API, for every mode and world size.

One pytest test per ``(world_size, mode)``; every case in the shared matrix runs inside
that test, so a world size pays the process-group setup cost once instead of once per
case. Each test prints a per-case table (which path actually ran, and the worst relative
error) and fails with that table attached, so one run tells you *which* configurations
broke rather than just the first one.

The oracle is always the same: one GPU, the whole global sequence, CP disabled.

The world size decides which modes are live, so the grid is diagonal: ``world_size=1``
runs the single-card modes (``none`` / ``intra``) in-process, with no process group; from
``world_size=2`` up, one process per rank is spawned and only the ``inter`` modes run --
a single-card mode there would just repeat identical work on every rank.

The mode-independent intra features (the ``auto_cp`` heuristic, the forward CP cache,
mixed forward/backward CP) live in ``test_cp_features.py``: they exercise the direct-call
wrappers rather than this case matrix.

Debugging a single configuration::

    pytest tests/test_cp_e2e.py --cp-world-sizes=2 --cp-case=offset-tpc2048-Hk8Hv8-g0.0625 -s
"""
import os
import sys
from dataclasses import dataclass

import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cp_common as C

MODES = [
    pytest.param("none", id="none"),
    pytest.param("intra", id="intra"),
    pytest.param("inter", id="inter"),
    pytest.param("inter_intra", id="inter_intra", marks=pytest.mark.cp_inter_intra),
]


def _select_cases(mode_name: str, world_size: int, case_id: str | None):
    cases = C.cases_for_mode(mode_name, world_size)
    if case_id is None:
        return cases
    return [C.find_case(mode_name, world_size, case_id)]


@dataclass
class _Row:
    case_id: str
    is_inter: bool
    is_intra: bool
    expect_intra: bool
    ratios: dict


def _format_table(mode_name: str, world_size: int, rows: list[_Row]) -> str:
    width = max((len(r.case_id) for r in rows), default=10)
    lines = [f"[{mode_name} CP, world_size={world_size}] rtol={C.RTOL:g}",
             f"  {'case'.ljust(width)}  path         worst"]
    for r in rows:
        name, value = C.worst(r.ratios)
        path = f"{'inter' if r.is_inter else '-'}+{'intra' if r.is_intra else '-'}"
        verdict = "OK  " if value <= C.RTOL else "FAIL"
        detail = "n/a" if name is None else f"{name}={value:.2e}"
        lines.append(f"  {r.case_id.ljust(width)}  {path:<12} {verdict} {detail}")
    return "\n".join(lines)


def _run_cases(mode_name, world_size, rank, group, case_id, device) -> list[_Row]:
    """Every selected case, forward and backward, on one rank."""
    mode = C.CP_MODES[mode_name]
    rows: list[_Row] = []
    for case in _select_cases(mode_name, world_size, case_id):
        inp = C.make_inputs(case, world_size, device)
        reference = C.run_reference(inp, need_grad=True)
        local, ctx = C.run_mode(inp, mode_name, rank, group=group, need_grad=True)
        ratios = C.compare(inp, local, reference, rank,
                           num_local_seqs=ctx.num_seqs, is_inter=ctx.is_inter)
        if group is not None:
            # Every rank must reach the same verdict, so the ratios are MAX-reduced.
            ratios = C.all_reduce_ratios(ratios, device)
        rows.append(_Row(
            case_id=case.id,
            is_inter=ctx.is_inter,
            is_intra=ctx.is_intra,
            # `force_intra_cp` bypasses the heuristic outright, so an intra-capable
            # mode must report an active split. Without it either answer is valid --
            # the table records which one happened.
            expect_intra=mode.is_intra and case.force_intra_cp,
            ratios=ratios,
        ))
        del inp, reference, local
        torch.cuda.empty_cache()
    return rows


def _verdict(mode_name: str, world_size: int, rank: int, rows: list[_Row]):
    """Print the table (rank 0 only) and assert on it."""
    is_intra_mode = C.CP_MODES[mode_name].is_intra
    table = _format_table(mode_name, world_size, rows)
    if rank == 0:
        print("\n" + table, flush=True)

    failures = [
        f"{r.case_id}: {C.format_ratios(r.ratios)}"
        for r in rows if C.worst(r.ratios)[1] > C.RTOL
    ]
    path_errors = []
    for r in rows:
        if r.expect_intra and not r.is_intra:
            path_errors.append(
                f"{r.case_id}: force_intra_cp was set but the context reports is_intra=False")
        if not is_intra_mode and r.is_intra:
            path_errors.append(
                f"{r.case_id}: mode {mode_name} unexpectedly enabled intra CP")
    assert not failures and not path_errors, (
        f"{mode_name} CP mismatch on rank {rank} (world_size={world_size})\n"
        + table
        + ("\n" + "\n".join(failures) if failures else "")
        + ("\n" + "\n".join(path_errors) if path_errors else "")
    )


def _cp_worker(rank: int, world_size: int, mode_name: str, case_id: str | None):
    """One spawned rank: join the group, run the cases, decide pass/fail locally."""
    try:
        group = C.init_distributed(rank, world_size, C.CP_MODES[mode_name].port(world_size))
        rows = _run_cases(mode_name, world_size, rank, group, case_id, f"cuda:{rank}")
    finally:
        C.cleanup_distributed()
    _verdict(mode_name, world_size, rank, rows)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.needs_bwd
@pytest.mark.parametrize("mode_name", MODES)
def test_cp_matrix(request, cp_world_size, mode_name):
    if cp_world_size is None:
        pytest.skip("no runnable world size (see --cp-world-sizes)")
    reason = C.skip_reason(mode_name, cp_world_size)
    if reason is None and cp_world_size > 1 and not C.CP_MODES[mode_name].is_inter:
        # A single-card mode ignores the world size, so spawning W ranks would just run
        # the same work W times. It is covered once, at world_size=1.
        reason = f"{mode_name} CP is single-card; covered at world_size=1"
    if reason:
        pytest.skip(reason)

    case_id = request.config.getoption("--cp-case")
    if cp_world_size == 1:
        # No inter split at W=1, so no process group and no spawn: run right here.
        rows = _run_cases(mode_name, 1, 0, None, case_id, "cuda:0")
        _verdict(mode_name, 1, 0, rows)
        return

    mp.start_processes(
        _cp_worker,
        args=(cp_world_size, mode_name, case_id),
        nprocs=cp_world_size,
        join=True,
        start_method="spawn",
    )
