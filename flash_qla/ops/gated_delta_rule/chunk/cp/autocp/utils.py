# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Shared autocp pieces: the structural features and the model form.

Split out of :mod:`.decision` so the online decision and the offline fit read one
definition of each -- ``tools/autocp/`` imports this module rather than restating
it, which is what makes "the fit saw exactly the features the decision sees" true
by construction rather than by an equivalence test.

**Standard library only** -- :mod:`.decision` sits on the launch path and must not
pull in numpy / pandas / torch, so neither may this module.

Two groups:

* structural features -- :func:`seq_chunks`, :func:`depth_runs`,
  :func:`schedule_depth`, :func:`struct_features`. No fitted parameters.
* the model -- :data:`MODEL_TERMS` plus :func:`predict_kernel` and
  :func:`load_coefs`.

Everything only a calibration run touches lives in ``tools/autocp/utils.py``
instead: the ``L_cp`` sweep grid, the shape CSV schema, the measurement cache, and
``save_coefs`` -- the write side of the file this module only ever reads.
"""

from __future__ import annotations

import csv
import heapq
import math
import os
from dataclasses import dataclass

KERNELS = ("prepare_h", "correct_h0", "fused_fwd", "fused_bwd")

#: Kernels only a *CP* shape pays for. ``cp/preprocess.py`` returns before
#: both of them when ``use_intra_cp`` is false, so CP off pays nothing for them
#: -- not even a launch constant. They are the entire price of enabling CP.
#:
#: Used twice, and it has to be the same tuple both times: :func:`.decision.predict`
#: drops them when pricing the no-CP baseline, and ``tools/autocp/fit.py``'s
#: ``fit_rows`` holds the baseline rows out of their fits, because those rows
#: measure a launched-but-trivial kernel the decision never prices.
CP_ONLY_KERNELS = ("prepare_h", "correct_h0")

#: Kernels the forward pass pays for; ``target='all'`` adds ``fused_bwd``.
FWD_KERNELS = tuple(k for k in KERNELS if k != "fused_bwd")

MIN_LCP = 4
#: correct_h0 launches ``ceildiv(DV, 32)`` CTAs along the value dim per (seq, head).
CORR_BV_SPLIT = 4

#: Coefficients ship as a CSV, not as a literal: ``tools/autocp/fit.py`` writes
#: this file and it is the only definition of the model. One file per calibrated
#: arch under ``coefs/``; ``sm100.csv`` is fitted on GB200 (sm_100, P=152) with
#: chunk=64, swa_ratio=0. This is the only data the wheel has to carry
#: (``setup.py``'s ``package_data`` names ``coefs/*.csv``) -- the shape sets that
#: produced it are calibration input and live in ``tools/autocp/cases/``.
COEFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coefs")
DEFAULT_COEFS_PATH = os.path.join(COEFS_DIR, "sm100.csv")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
#: Per kernel, the linear terms as ``(StructFeatures field, coefficient name)``,
#: plus an implicit constant ``c``.
#:
#: This is the *only* place the model form is written down, and it stays in the
#: shipped package for that reason even though three of its four readers are
#: offline: the design matrix (``fit._design``), the coefficient packing
#: (``fit.fit_all``) and the offline prediction (``fit._predict_column``) are all
#: generated from it, as is the online prediction (:func:`.decision.predict`), so
#: the fit and the launch path cannot drift apart -- previously the form was
#: spelled out in four places and their agreement rested on matching column order
#: by hand.
#:
#: ``correct_h0`` has no schedule-depth main term of its own: its cost *is* the
#: serial scan depth, so it is priced ``kappa * d_corr + c`` with no ``tau``.
MODEL_TERMS = {
    "prepare_h": (("d_p", "tau"), ("u", "kappa")),
    "correct_h0": (("d_corr", "kappa"),),
    "fused_fwd": (("d_fb", "tau"), ("u", "kappa")),
    "fused_bwd": (("d_fb", "tau"), ("u", "kappa")),
}

#: Coefficient names in design-matrix column order, constant last.
COEF_NAMES = {k: tuple(c for _, c in terms) + ("c",)
              for k, terms in MODEL_TERMS.items()}


def predict_kernel(coefs: dict, feat: "StructFeatures", kernel: str) -> float:
    """Predicted milliseconds for one kernel on one candidate shape."""
    k = coefs["kernels"][kernel]
    return (sum(k[cname] * getattr(feat, fname)
                for fname, cname in MODEL_TERMS[kernel]) + k["c"])


def load_coefs(path: str | None = None) -> dict:
    path = path or DEFAULT_COEFS_PATH
    kernels: dict[str, dict[str, float]] = {}
    P = chunk = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            kernels[row["kernel"].strip()] = dict(
                tau=float(row["tau"]) if row["tau"] else 0.0,
                kappa=float(row["kappa"]),
                c=float(row["c"]),
            )
            P, chunk = int(row["P"]), int(row["chunk"])
    missing = [k for k in KERNELS if k not in kernels]
    if missing:
        raise ValueError(f"coefficient CSV is missing kernels {missing}: {path}")
    return {"kernels": kernels, "P": P, "chunk": chunk}


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------
def seq_chunks(cu_seqlens, chunk: int) -> list[int]:
    """Per-raw-sequence chunk counts from cumulative token offsets."""
    cu = list(cu_seqlens)
    return [math.ceil((cu[i + 1] - cu[i]) / chunk) for i in range(len(cu) - 1)]


def depth_runs(chunks, S: int, warmup: int):
    fb, pr, co = [], [], []
    w = min(S, warmup)  # internal partitions all have depth S, so this is per-S
    for c in chunks:
        n = max(1, math.ceil(c / S))
        if n > 1:
            fb.append((S, n - 1))
            pr.append((w, n - 1))
        fb.append((c - S * (n - 1), 1))
        pr.append((0, 1))
        co.append((n, 1))
    return fb, pr, co


def schedule_depth(runs, copies: int, P: int) -> float:
    """Depth of the critical path once ``P`` SMs greedily backfill ``runs``.

    ``runs`` is ``(depth, count)`` groups of interchangeable CTAs, each replicated
    ``copies`` times (once per head); the result is in the same unit as ``depth``,
    i.e. chunks, and is what ``tau`` prices. Greedy is exact here because the CTAs
    are independent -- an SM taking the next one as soon as it frees is optimal,
    and a short tail run lands in the gaps the deep runs leave rather than adding
    a wave of its own. That is why tail hiding needs no correction term.
    """
    P, copies = int(P), int(copies)
    if P < 1:
        raise ValueError(f"P (SM count) must be >= 1, got {P}")

    free = {0.0: P}  # free time -> how many SMs come free at it
    times = [0.0]    # min-heap over the distinct free times
    end = 0.0
    for d, count in runs:
        need = count * copies
        while need:
            t = times[0]
            n = free[t]
            k = n if n <= need else need
            if k == n:
                del free[t]
                heapq.heappop(times)
            else:
                free[t] = n - k
            t1 = t + d
            if t1 in free:
                free[t1] += k
            else:
                free[t1] = k
                heapq.heappush(times, t1)
            if t1 > end:
                end = t1
            need -= k
    return end


@dataclass(frozen=True)
class StructFeatures:
    """Fit-parameter-free structural features of one candidate ``S``."""

    S: int
    num_raw: int          # raw sequences
    num_partitions: int   # CP partitions (== cp_batch_size)
    u: float              # G / P, with G = H * num_partitions total CTAs
    d_fb: float           # fused_fwd / fused_bwd schedule depth
    d_p: float            # prepare_h schedule depth
    d_corr: float         # correct_h0 schedule depth

    @property
    def enable_cp(self) -> bool:
        """True when at least one raw sequence was actually partitioned."""
        return self.num_partitions > self.num_raw


def struct_features(chunks, S: int, H: int, P: int,
                    warmup: int | None = None) -> StructFeatures:
    if warmup is None:
        warmup = S
    fb, pr, co = depth_runs(chunks, S, warmup)
    n_part = sum(n for n, _ in co)  # corr depth *is* the sequence's partition count
    return StructFeatures(
        S=S,
        num_raw=len(chunks),
        num_partitions=n_part,
        u=H * n_part / P,
        d_fb=schedule_depth(fb, H, P),
        d_p=schedule_depth(pr, H, P),
        d_corr=schedule_depth(co, CORR_BV_SPLIT * H, P),
    )
