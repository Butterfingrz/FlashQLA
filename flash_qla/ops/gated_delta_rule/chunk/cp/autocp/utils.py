# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import csv
import heapq
import math
import os
import warnings
from collections import Counter
from dataclasses import dataclass

ROWS = ("prepare_h", "correct_h0", "correct_dht", "fused_fwd",
        "prepare_dh", "recompute_h", "fused_bwd")

INFER_ROWS = ("prepare_h", "correct_h0", "fused_fwd")

TRAIN_ROWS = ("prepare_h", "correct_h0", "fused_fwd",
              "prepare_dh", "correct_dht", "recompute_h", "fused_bwd")

CP_ONLY_ROWS = ("prepare_h", "correct_h0", "correct_dht", "prepare_dh")

#: Coefficient row -> measured columns + the depth feature each fills. Offline fit only.
ROW_SOURCES = {
    "prepare_h":   (("prepare_h", "d_p_fwd"), ("prepare_h_bidi", "d_p_bidi")),
    "correct_h0":  (("correct_h0", "d_corr"),),
    "correct_dht": (("correct_dht", "d_corr"),),
    "fused_fwd":   (("fused_fwd", "d_fb"),),
    "prepare_dh":  (("prepare_dh", "d_p_bwd"),),
    "recompute_h": (("recompute_h", "d_fb"),),
    "fused_bwd":   (("fused_bwd", "d_fb"),),
}

# Legacy CSVs named the pre-split shared row ``correct``; read it as ``correct_h0``.
_ROW_ALIASES = {"correct": "correct_h0"}

# Pre-split names kept for legacy imports.
KERNELS = ROWS
FWD_KERNELS = INFER_ROWS
CP_ONLY_KERNELS = CP_ONLY_ROWS

MIN_LCP = 4
CORR_BV_SPLIT = 4

WARMUP_TOL = -10.0

COEFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coefs")
DEFAULT_COEFS_PATH = os.path.join(COEFS_DIR, "sm100.csv")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
# Per kernel, the linear terms as (StructFeatures field, coefficient name), plus
# an implicit constant c. prepare_h's d_p is mode-dependent (fwd-only vs bidi).
MODEL_TERMS = {
    "prepare_h": (("d_p", "tau"), ("u", "kappa")),
    "correct_h0": (("d_corr", "tau"), ("u", "kappa")),
    "correct_dht": (("d_corr", "tau"), ("u", "kappa")),
    "fused_fwd": (("d_fb", "tau"), ("u", "kappa")),
    "prepare_dh": (("d_p_bwd", "tau"), ("u", "kappa")),
    "recompute_h": (("d_fb", "tau"), ("u", "kappa")),
    "fused_bwd": (("d_fb", "tau"), ("u", "kappa")),
}

#: Coefficient names in design-matrix column order, constant last.
COEF_NAMES = {k: tuple(c for _, c in terms) + ("c",)
              for k, terms in MODEL_TERMS.items()}


def predict_kernel(coefs: dict, feat: "StructFeatures", kernel: str) -> float:
    k = coefs["kernels"][kernel]
    return (sum(k[cname] * getattr(feat, fname)
                for fname, cname in MODEL_TERMS[kernel]) + k["c"])


def load_coefs(path: str | None = None) -> dict:
    path = path or DEFAULT_COEFS_PATH
    kernels: dict[str, dict[str, float]] = {}
    P = chunk = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["kernel"].strip()
            canon = _ROW_ALIASES.get(name, name)
            if canon in kernels:
                raise ValueError(
                    f"coefficient CSV defines row {canon!r} twice "
                    f"(second time as {name!r}): {path}")
            entry = dict(
                tau=float(row["tau"]) if row["tau"] else 0.0,
                kappa=float(row["kappa"]) if row["kappa"] else 0.0,
                c=float(row["c"]) if row["c"] else 0.0,
            )
            # Legacy correct_h0 put the depth coefficient in kappa (blank tau);
            # migrate it to the unified form (kappa -> tau, no u term) loudly.
            if canon == "correct_h0" and not row["tau"].strip() and entry["kappa"]:
                warnings.warn(
                    f"autocp: {path} has a legacy 'correct_h0' row (blank tau, "
                    "depth coefficient in kappa); migrating it to the unified form "
                    "(kappa -> tau, u term = 0). Re-fit to silence this.",
                    RuntimeWarning, stacklevel=2,
                )
                entry["tau"], entry["kappa"] = entry["kappa"], 0.0
            kernels[canon] = entry
            P, chunk = int(row["P"]), int(row["chunk"])
    missing = [k for k in INFER_ROWS if k not in kernels]
    if missing:
        raise ValueError(f"coefficient CSV is missing rows {missing}: {path}")
    return {"kernels": kernels, "P": P, "chunk": chunk}


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------
def seq_chunks(cu_seqlens, chunk: int) -> list[int]:
    cu = list(cu_seqlens)
    return [math.ceil((cu[i + 1] - cu[i]) / chunk) for i in range(len(cu) - 1)]


def schedule_depth(runs, copies: int, P: int) -> float:
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

    S: int
    num_raw: int          # raw sequences
    num_partitions: int   # CP partitions (== cp_batch_size)
    u: float              # G / P, with G = H * num_partitions total CTAs
    d_fb: float           # fused_fwd / fused_bwd / recompute_h schedule depth
    d_p: float            # prepare_h schedule depth -- forward-only warmup for
                          # is_train=False, bidirectional (max of both) for True
    d_corr: float         # correct schedule depth
    d_p_bwd: float = 0.0  # prepare_dh schedule depth; 0 unless is_train

    @property
    def enable_cp(self) -> bool:
        return self.num_partitions > self.num_raw


def warmup_from_gate_avg(avg_per_head, chunk: int,
                         tol: float = WARMUP_TOL) -> list[int]:
    out: list[int] = []
    for a in avg_per_head:
        if a >= -1e-12:
            out.append(math.inf)
        else:
            out.append(max(1, math.ceil(tol / (a * chunk))))
    return out


def struct_features(chunks, S: int, H: int, P: int,
                    warmup_per_head, is_train: bool = False) -> StructFeatures:
    if len(warmup_per_head) != H:
        raise ValueError(
            f"warmup_per_head has {len(warmup_per_head)} entries, expected H={H}")

    fb, co = [], []
    for c in chunks:
        n = max(1, math.ceil(c / S))
        if n > 1:
            fb.append((S, n - 1))
        fb.append((c - S * (n - 1), 1))
        co.append((n, 1))
    n_part = sum(n for n, _ in co)  # corr depth *is* the sequence's partition count

    prep, bwd = [], []
    for depth, copies in Counter(min(k, S) for k in warmup_per_head).items():
        w = min(S, depth)
        for c in chunks:
            n = max(1, math.ceil(c / S))
            if not is_train:
                if n > 1:
                    prep.append((w, (n - 1) * copies))
                prep.append((0, copies))
                continue
            if n == 1:
                prep.append((0, copies))
                bwd.append((0, copies))
                continue
            rem = min(c - S * (n - 1), w)  # ragged partition can't warm past its len
            prep.append((w, (n - 1) * copies))
            prep.append((rem, copies))
            if n > 2:
                bwd.append((w, (n - 2) * copies))
            bwd.append((rem, copies))
            bwd.append((0, copies))

    return StructFeatures(
        S=S,
        num_raw=len(chunks),
        num_partitions=n_part,
        u=H * n_part / P,
        d_fb=schedule_depth(fb, H, P),
        d_p=schedule_depth(prep, 1, P),
        d_corr=schedule_depth(co, CORR_BV_SPLIT * H, P),
        d_p_bwd=schedule_depth(bwd, 1, P) if is_train else 0.0,
    )
