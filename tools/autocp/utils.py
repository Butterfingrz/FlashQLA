# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Offline-only autocp helpers: the L_cp grid, the shape CSV, the cache, the writer.

The other half of this pair is ``autocp.utils`` inside the shipped package. The
split is by *who needs it at launch time*:

* **there** -- the structural features, the model form and :func:`load_coefs`.
  ``decision.py`` calls those on the launch path, and the offline side imports
  them from there rather than keeping a copy, which is what makes "the fit saw
  exactly the features the decision sees" true by construction.
* **here** -- everything only a calibration run touches: the ``L_cp`` grid it
  sweeps, the shape CSV it reads, the measurement cache it fills, and
  :func:`save_coefs`, the write side of a file the package only ever reads.

Nothing here ships in the wheel, and nothing in ``flash_qla/`` may import it.
**Standard library only** all the same: ``--from-cache`` refits with no GPU and
must not pull in torch, and :mod:`.fit` is the only module allowed numpy/pandas.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from flash_qla.ops.gated_delta_rule.chunk.cp.autocp.utils import (
    COEF_NAMES,
    KERNELS,
    MIN_LCP,
    seq_chunks,
)


def save_coefs(path: str, coefs: dict) -> None:
    """Write fitted coefficients in the layout ``utils.load_coefs`` reads.

    The read side lives in the package because the launch path needs it; only a
    fit ever writes, so the writer lives out here.

    ``P`` / ``chunk`` are duplicated on every row so the reader needs no separate
    metadata parsing. A coefficient the kernel's model form does not use (``tau``
    for ``correct_h0``) is stored empty rather than as ``0``.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "tau", "kappa", "c", "P", "chunk"])
        for name in KERNELS:
            k = coefs["kernels"][name]
            used = COEF_NAMES[name]
            cell = lambda c: f"{k[c]:.8g}" if c in used else ""
            w.writerow([name, cell("tau"), cell("kappa"), cell("c"),
                        coefs["P"], coefs["chunk"]])


# ---------------------------------------------------------------------------
# The L_cp grid
# ---------------------------------------------------------------------------
#: Geometric step between successive ``L_cp`` samples. 1.5 puts 13 points on a
#: 512-chunk sequence, and deliberately lands off powers of two -- whether the
#: measured optimum sits exactly on a pow2 is one of the things the sweep is
#: meant to answer, so a pow2-only grid could not see it.
DEFAULT_LCP_RATIO = 1.5


def lcp_grid(cmax: int, ratio: float = DEFAULT_LCP_RATIO,
             min_S: int = MIN_LCP) -> list[int]:
    """Geometric ``L_cp`` grid from ``min_S`` to ``cmax``, ending exactly on ``cmax``.

    ``cmax`` is the single baseline point: every ``S >= cmax`` collapses to the
    same no-CP shape, so sampling beyond it would re-measure one shape under
    different labels.

    A sequence too short to enable CP (``cmax <= min_S``) yields just ``[cmax]``
    -- the baseline row alone, no CP points.

    This is the *measurement* grid, not the decision's: ``decision.py`` enumerates
    its own candidates at wave boundaries. The two are deliberately different, so
    that the shapes the model is scored on are not the ones it was handed.
    """
    cmax = int(cmax)
    if cmax <= min_S:
        return [cmax]
    grid, s = [], int(min_S)
    while s < cmax:
        grid.append(s)
        # round() alone stalls at small S, where ratio is under one chunk.
        s = max(s + 1, int(round(s * ratio)))
    grid.append(cmax)
    return grid


# ---------------------------------------------------------------------------
# Shape CSV: what the fit takes as input
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapeSpec:
    """One calibration shape. Carries no timings and no ``L_cp``.

    ``L_cp`` is not an input: it is enumerated per shape by :func:`lcp_grid`,
    because the useful grid depends on the shape's own ``cmax``.
    """

    name: str
    H: int                    # value heads
    Hk: int                   # key heads
    cu: tuple[int, ...]       # cumulative token offsets, cu[0] == 0
    dtype: str = "bfloat16"
    swa_ratio: float = 0.0

    @property
    def seqlens(self) -> tuple[int, ...]:
        return tuple(self.cu[i + 1] - self.cu[i] for i in range(len(self.cu) - 1))

    @property
    def T(self) -> int:
        return int(self.cu[-1])

    @property
    def B_raw(self) -> int:
        return len(self.cu) - 1

    def chunks(self, chunk: int) -> list[int]:
        return seq_chunks(self.cu, chunk)

    def cmax(self, chunk: int) -> int:
        return max(self.chunks(chunk))

    @property
    def cu_str(self) -> str:
        return ":".join(str(int(x)) for x in self.cu)


def _parse_cu(row: dict, name: str) -> tuple[int, ...]:
    """Shape from whichever of the three accepted spellings the row uses."""
    if row.get("cu"):
        cu = [int(x) for x in str(row["cu"]).replace(",", ":").split(":")]
        if cu[0] != 0:
            raise ValueError(f"{name}: 'cu' must start at 0, got {cu[0]}")
        lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    elif row.get("seqlens"):
        lens = [int(x) for x in str(row["seqlens"]).replace(":", ",").split(",")]
    elif row.get("seqlen"):
        lens = [int(row["seqlen"])] * int(row.get("batch") or 1)
    else:
        raise ValueError(
            f"{name}: need one of 'cu' / 'seqlens' / 'seqlen' (+ optional 'batch')")
    if not lens or min(lens) <= 0:
        raise ValueError(f"{name}: non-positive sequence length in {lens}")
    out, acc = [0], 0
    for x in lens:
        acc += x
        out.append(acc)
    return tuple(out)


def load_specs(path: str, *, Hk: int | None = None, dtype: str = "bfloat16",
               swa_ratio: float = 0.0) -> list[ShapeSpec]:
    """Read a shape CSV.

    Required: ``H`` plus one of ``cu`` / ``seqlens`` / ``seqlen`` (+ ``batch``).
    Optional: ``name``, ``Hk`` (defaults to ``H``), ``dtype``, ``swa_ratio`` --
    a missing optional column falls back to the argument here, so the smallest
    usable CSV is two columns wide.

    ``#`` lines are skipped, so a calibration set can say in the file itself why
    its shapes were chosen. The checked-in sets are ``cases/train.csv`` and
    ``cases/eval.csv`` next to this file.
    """
    specs: list[ShapeSpec] = []
    with open(path, newline="") as f:
        rows = (ln for ln in f if not ln.lstrip().startswith("#"))
        for i, row in enumerate(csv.DictReader(rows)):
            row = {(k.strip() if k else k): v for k, v in row.items()}
            if not any((v or "").strip() for v in row.values()):
                continue  # blank line
            if not row.get("H"):
                raise ValueError(
                    f"{path}: data row {i + 1} "
                    f"({row.get('name') or 'unnamed'}) has no 'H'")
            H = int(row["H"])
            fallback = f"row{i + 1}"
            cu = _parse_cu(row, row.get("name") or fallback)
            B = len(cu) - 1
            specs.append(ShapeSpec(
                name=(row.get("name") or "").strip() or f"h{H}_{B}x{cu[-1] // B}",
                H=H,
                Hk=int(row["Hk"]) if row.get("Hk") else (Hk if Hk else H),
                cu=cu,
                dtype=(row.get("dtype") or "").strip() or dtype,
                swa_ratio=(float(row["swa_ratio"])
                           if row.get("swa_ratio") not in (None, "") else swa_ratio),
            ))
    if not specs:
        raise ValueError(f"no shape rows in {path}")
    dupes = sorted({s.name for s in specs
                    if sum(x.name == s.name for x in specs) > 1})
    if dupes:
        raise ValueError(
            f"{path}: duplicate shape names {dupes}. Names are the report's "
            f"grouping key (and the unit of regret / verdict), so they must be "
            f"unique -- add or edit the 'name' column.")
    return specs


# ---------------------------------------------------------------------------
# Measurement cache: what sits between measuring and fitting
# ---------------------------------------------------------------------------
#: Cache columns. The shape / timing block keeps the names the earlier sweep
#: scripts wrote so historical ``debug/*.csv`` stay readable side by side; the
#: leading block is what makes a row addressable.
CACHE_COLUMNS = (
    "name", "device", "P", "chunk", "dtype", "swa_ratio",
    "H", "Hk", "T", "Lc", "L_cp", "N_part", "B_raw", "cmax", "cu",
    "warmup_max", "warmup_mean",
    "prepare_h", "correct_h0", "fused_fwd", "fused_bwd",
    "fwd_total", "all_total",
)

#: What identifies a measurement. ``name`` is deliberately absent: relabelling a
#: shape must not invalidate its timings.
CACHE_KEY_COLUMNS = ("device", "P", "chunk", "dtype", "swa_ratio",
                     "H", "Hk", "cu", "L_cp")

_KEY_FMT = {"P": int, "chunk": int, "H": int, "Hk": int, "L_cp": int,
            "swa_ratio": lambda v: f"{float(v):g}"}


def cache_key(row: dict) -> tuple:
    """Hashable identity of a cache row, tolerant of str/int/float spellings."""
    out = []
    for col in CACHE_KEY_COLUMNS:
        v = row[col]
        f = _KEY_FMT.get(col)
        out.append(str(f(v)) if f else str(v).strip())
    return tuple(out)


def read_cache(path: str) -> dict[tuple, dict]:
    """``cache_key -> row`` for an existing cache (empty dict when absent).

    Later rows win, so re-measuring by appending supersedes the old value
    without having to rewrite the file.
    """
    if not path or not os.path.exists(path):
        return {}
    out: dict[tuple, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            missing = [c for c in CACHE_KEY_COLUMNS if c not in row]
            if missing:
                raise ValueError(
                    f"{path} is not an autocp measurement cache "
                    f"(missing key columns {missing})")
            for col in ("warmup_mean", *KERNELS, "fwd_total", "all_total"):
                if row.get(col) not in (None, ""):
                    row[col] = float(row[col])
            for col in ("H", "Hk", "T", "Lc", "L_cp", "N_part", "B_raw", "cmax",
                        "warmup_max", "P", "chunk"):
                if row.get(col) not in (None, ""):
                    row[col] = int(float(row[col]))
            out[cache_key(row)] = row
    return out


def append_cache(path: str, row: dict) -> None:
    """Append one measured row, writing the header if the file is new.

    Appending per row (rather than once at the end) is what makes an interrupted
    sweep resumable: a full calibration is hours of GPU time.
    """
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CACHE_COLUMNS),
                          extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def device_slug(name: str) -> str:
    """Filesystem-safe device tag for the default cache path."""
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()
                   ).strip("_") or "unknown"


def default_cache_path(device: str, root: str = "debug") -> str:
    return os.path.join(root, f"autocp_cache_{device_slug(device)}.csv")
