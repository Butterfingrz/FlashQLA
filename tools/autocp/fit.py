# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Offline autocp calibration: shape CSVs in, coefficient CSV out.

The whole pipeline lives here -- **measuring included**:

    shape CSV -> L_cp grid -> measure (cached) -> least squares -> coefficient CSV

Taking *shapes* rather than pre-measured timings is what makes a recalibration
reproducible from what is checked in. ``warmup_max`` is the reason it has to work
this way: it is an output of ``get_warmup_chunks`` (it depends on the gate values
and on ``swa_ratio``), so no shape-only CSV can carry it and no fit can proceed
without having measured. The ``L_cp`` grid is likewise generated
(:func:`.utils.lcp_grid`) instead of being written down per shape.

Measurements are cached (see :func:`.utils.read_cache`), so re-fitting is cheap
and interrupted sweeps resume. ``--from-cache`` fits from the cache alone and
never imports torch.

One least-squares solve per kernel over all rows at once. Every shape dependency
(H, batch, varlen, sub-segment length) is carried by the fit-parameter-free
structural features from :mod:`.utils`, so the coefficients are global -- there
are no per-H or per-batch tables. The features come from the very function the
online decision calls, so "the fit saw what the decision sees" holds by
construction.

Generalization is answered by a held-out ``--eval`` shape CSV rather than by
leave-one-out over the training set.

This module lives in ``tools/`` rather than in ``flash_qla/`` because none of it
runs at launch time: it is the offline half of autocp, and it is the only thing
here that wants numpy / pandas (and, with ``--plot``, matplotlib). The shipped
package keeps only what the decision needs -- ``utils`` (the shared model form),
``decision`` and ``coefs/``.

Run it as a module from the repo root, so that ``tools.autocp`` resolves::

    python -m tools.autocp.fit \\
        --train tools/autocp/cases/train.csv \\
        --eval  tools/autocp/cases/eval.csv \\
        --out   flash_qla/ops/gated_delta_rule/chunk/cp/autocp/coefs/sm100.csv \\
        --plot  debug/autocp_fit

``scripts/fit_autocp.sh`` wraps that with the checked-in paths already filled in.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from flash_qla.ops.gated_delta_rule.chunk.cp.autocp.decision import decide
from flash_qla.ops.gated_delta_rule.chunk.cp.autocp.utils import (
    COEF_NAMES,
    KERNELS,
    MODEL_TERMS,
    CP_ONLY_KERNELS,
    struct_features,
)
from .utils import (
    cache_key,
    default_cache_path,
    lcp_grid,
    load_specs,
    read_cache,
    save_coefs,
)

__all__ = ["build_frame", "rows_for_specs", "fit_kernel", "fit_rows", "fit_all",
           "mape", "accuracy_report", "decision_report", "nocp_measured",
           "plot_fit", "main", "CP_ONLY_KERNELS"]


def fit_rows(df: pd.DataFrame, kernel: str) -> pd.DataFrame:
    return df[df["cp"]] if kernel in CP_ONLY_KERNELS else df


# ---------------------------------------------------------------------------
# Cache rows -> feature frame
# ---------------------------------------------------------------------------
def rows_for_specs(cache: dict, specs, *, P: int, chunk: int, ratio: float,
                   device: str, strict: bool = True) -> list[dict]:
    rows, missing = [], []
    for spec in specs:
        for lcp in lcp_grid(spec.cmax(chunk), ratio):
            key = cache_key(dict(device=device, P=P, chunk=chunk,
                                 dtype=spec.dtype, swa_ratio=spec.swa_ratio,
                                 H=spec.H, Hk=spec.Hk, cu=spec.cu_str,
                                 L_cp=lcp))
            row = cache.get(key)
            if row is None:
                missing.append((spec.name, lcp))
            else:
                rows.append({**row, "name": spec.name})
    if missing and strict:
        raise SystemExit(
            f"{len(missing)} measurement(s) missing from the cache, e.g. "
            f"{missing[:6]}. Drop --from-cache to measure them.")
    return rows


def build_frame(rows, P: int, chunk: int = 64, part: str | None = None
                ) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        raise ValueError("no measured rows")
    df["setting"] = df["name"]
    if part is not None:
        df["part"] = part

    feats = [struct_features(
        [int(c) for c in _chunks_of(r["cu"], chunk)],
        int(r["L_cp"]), int(r["H"]), P, warmup=int(r["warmup_max"]))
        for _, r in df.iterrows()]

    df["u"] = [f.u for f in feats]
    df["d_fb"] = [f.d_fb for f in feats]
    df["d_p"] = [f.d_p for f in feats]
    df["d_corr"] = [f.d_corr for f in feats]
    df["cp"] = [f.enable_cp for f in feats]

    if "N_part" in df:
        got = df["N_part"].astype(int).to_numpy()
        want = np.array([f.num_partitions for f in feats])
        bad = np.flatnonzero(got != want)
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"{bad.size} row(s) where the measured partition count differs "
                f"from struct_features, e.g. {df['setting'].iloc[i]} "
                f"L_cp={df['L_cp'].iloc[i]}: measured {got[i]} vs modelled "
                f"{want[i]}. depth_runs and _build_intra_cp_context disagree.")
    return df


def _chunks_of(cu: str, chunk: int) -> list[int]:
    from flash_qla.ops.gated_delta_rule.chunk.cp.autocp.utils import seq_chunks
    return seq_chunks([int(x) for x in str(cu).split(":")], chunk)


# ---------------------------------------------------------------------------
# Least squares
# ---------------------------------------------------------------------------
def _design(df: pd.DataFrame, kernel: str):
    cols = [df[field].to_numpy(float) for field, _ in MODEL_TERMS[kernel]]
    X = np.stack(cols + [np.ones(len(df))], axis=1)
    return X, df[kernel].to_numpy(float)


def fit_kernel(df: pd.DataFrame, kernel: str, settings=None) -> np.ndarray:
    sub = df if settings is None else df[df["setting"].isin(settings)]
    X, y = _design(fit_rows(sub, kernel), kernel)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef[0] = max(coef[0], 0.0)
    return coef


def fit_all(df: pd.DataFrame, P: int, chunk: int = 64) -> dict:
    """Fit every kernel and pack the result in :func:`.utils.load_coefs` layout."""
    kernels = {}
    for k in KERNELS:
        coef = fit_kernel(df, k)
        packed = dict(tau=0.0, kappa=0.0, c=0.0)
        packed.update(zip(COEF_NAMES[k], (float(x) for x in coef)))
        kernels[k] = packed
    return {"kernels": kernels, "P": int(P), "chunk": int(chunk)}


def _predict_column(coefs: dict, df: pd.DataFrame, kernel: str) -> np.ndarray:
    X, _ = _design(df, kernel)
    vec = np.array([coefs["kernels"][kernel][c] for c in COEF_NAMES[kernel]])
    return X @ vec


def mape(y: np.ndarray, yhat: np.ndarray, mask) -> float:
    """Mean absolute percentage error over ``mask`` (NaN when empty)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / np.abs(y[mask])) * 100)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def accuracy_report(df: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    keys = (["part", "setting"] if "part" in df else ["setting"])
    rows = []
    for k in KERNELS:
        y = df[k].to_numpy(float)
        yhat = _predict_column(coefs, df, k)
        priced = df.index.isin(fit_rows(df, k).index)
        for vals, g in df.groupby(keys, sort=False):
            vals = vals if isinstance(vals, tuple) else (vals,)
            sel = df.index.isin(g.index) & priced
            rows.append({**dict(zip(keys, vals)), "kernel": k,
                         "mape": mape(y, yhat, sel), "n_fit": int(sel.sum())})
    return pd.DataFrame(rows).sort_values([*keys, "kernel"], kind="stable")


def nocp_measured(base_rows: pd.DataFrame) -> float:
    return float(base_rows["fused_fwd"].iloc[0] + base_rows["fused_bwd"].iloc[0])


def decision_report(df: pd.DataFrame, coefs: dict, P: int) -> pd.DataFrame:
    out = []
    for label, sub in df.groupby("setting", sort=True):
        H = int(sub["H"].iloc[0])
        cu = [int(x) for x in str(sub["cu"].iloc[0]).split(":")]
        cp = sub[sub["cp"]]
        base = sub[~sub["cp"]]
        kw = dict(cu_seqlens=cu, num_v_heads=H, coefs=coefs, P=P, debug=True)

        rec = dict(setting=label)
        if "part" in sub:
            rec["part"] = sub["part"].iloc[0]
        rec.update(n_cp=len(cp), n_base=len(base))
        r = decide(**kw)
        rec["verdict"] = "CP" if r["use_cp"] else "no-CP"
        rec["lcp"] = r["lcp"]

        if len(cp):
            grid = sorted(int(s) for s in cp["L_cp"].unique())
            on_grid = decide(candidates=grid, margin=0.0,
                             **{k: v for k, v in kw.items()})
            S = on_grid["best_cp_lcp"]
            at_model = float(cp[cp["L_cp"] == S]["all_total"].iloc[0])
            best = float(cp["all_total"].min())
            rec.update(
                lcp_model=S,
                lcp_meas=int(cp.loc[cp["all_total"].idxmin(), "L_cp"]),
                regret_pct=(at_model / best - 1.0) * 100)
        else:
            rec.update(lcp_model=None, lcp_meas=None, regret_pct=float("nan"))

        if len(base):
            t_base = nocp_measured(base)
            rec["base_err_pct"] = (r["nocp_pred_ms"] / t_base - 1.0) * 100
            rec["verdict_meas"] = ("CP" if len(cp)
                                   and float(cp["all_total"].min()) < t_base
                                   else "no-CP")
        else:
            rec["base_err_pct"] = float("nan")
            rec["verdict_meas"] = None
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
#: Panel tint per part, so a held-out shape is recognisable without reading the
#: title.
_PART_BG = {"train": "#ffffff", "eval": "#eef4ee"}
_KERNEL_COLOR = {"prepare_h": "#4C78A8", "correct_h0": "#F58518",
                 "fused_fwd": "#54A24B", "fused_bwd": "#E45756"}


def _pyplot():
    """matplotlib, imported here and nowhere else.

    Plotting is optional and matplotlib is deliberately **not** a dependency of
    ``flash_qla``: nothing at run time plots, and the shipped package does not
    even contain this module. So it is an import check with an install line rather
    than a requirement -- and :func:`main` runs this check up front, before
    measuring, so a missing matplotlib costs no GPU time.
    """
    try:
        import matplotlib
    except ImportError as e:
        raise SystemExit(
            "--plot needs matplotlib, which is not a dependency of flash_qla "
            "(nothing on the launch path plots). Install it for calibration "
            "only:\n    pip install matplotlib") from e
    matplotlib.use("Agg")  # calibration runs on headless GPU boxes
    import matplotlib.pyplot as plt
    return matplotlib, plt


def _parts(df: pd.DataFrame) -> pd.Series:
    return df["part"] if "part" in df else pd.Series("train", index=df.index)


def _totals(coefs: dict, df: pd.DataFrame):
    """``(measured, predicted)`` latency under the *decision's* accounting.

    Two conventions in one pair of columns, because that is what :func:`decide`
    compares: a CP row pays all four kernels, a baseline row only the fused pair.
    ``cp/preprocess.py`` returns before ``prepare_h`` / ``correct_h0`` when CP is
    off, so pricing them there would charge the no-CP branch for launches it never
    makes. The measured columns *do* hold those trivial launches -- which is why
    they have to be dropped by hand here instead of summing all four.
    """
    pred = {k: _predict_column(coefs, df, k) for k in KERNELS}
    fused = [k for k in KERNELS if k not in CP_ONLY_KERNELS]
    cp = df["cp"].to_numpy(bool)
    return (np.where(cp, df[list(KERNELS)].sum(axis=1).to_numpy(float),
                     df[fused].sum(axis=1).to_numpy(float)),
            np.where(cp, sum(pred[k] for k in KERNELS),
                     sum(pred[k] for k in fused)))


def _plot_curves(plt, matplotlib, df: pd.DataFrame, dec: pd.DataFrame,
                 path: str) -> str:
    """Per shape: total latency vs ``L_cp``, measured points against the curve.

    This is the figure the decision is actually read off: the model's ``L_cp`` is
    the curve's arg-min, and CP-on/off is the curve dipping below the no-CP line.
    The fit only ever saw ``train`` rows, so an ``eval`` panel is the curve
    extrapolated onto a shape it was never given -- hence the two tints.

    Arg-mins and regret come from ``dec`` rather than being recomputed, so the
    picture and the printed table cannot disagree.
    """
    rec = {r["setting"]: r for r in dec.to_dict("records")}
    shapes = sorted(df["setting"].unique())
    ncol = min(4, len(shapes))
    nrow = -(-len(shapes) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow),
                             squeeze=False)
    for ax, name in zip(axes.ravel(), shapes):
        g = df[df["setting"] == name].sort_values("L_cp")
        cp, base = g[g["cp"]], g[~g["cp"]]
        part = str(_parts(g).iloc[0])
        r = rec.get(name, {})
        if len(cp):
            ax.plot(cp["L_cp"], cp["meas_total"], "o", ms=4.5, color="#333",
                    label="measured (CP)")
            ax.plot(cp["L_cp"], cp["pred_total"], "-", lw=1.7, color="#E45756",
                    label="model")
        if len(base):
            ax.axhline(float(base["meas_total"].iloc[0]), color="#4C78A8",
                       lw=1.3, label="measured no-CP")
            ax.axhline(float(base["pred_total"].iloc[0]), color="#4C78A8",
                       lw=1.1, ls="--", label="model no-CP")
        for key, color in (("lcp_meas", "#333"), ("lcp_model", "#E45756")):
            if r.get(key):
                ax.axvline(int(r[key]), color=color, ls=":", lw=1.1)
        regret = r.get("regret_pct")
        sub = (f"L* meas {r.get('lcp_meas', '-')} / model "
               f"{r.get('lcp_model', '-')}"
               + (f", regret {regret:.2f}%" if regret == regret else "")
               + f"\nverdict {r.get('verdict', '?')}"
               + (f" (measured {r['verdict_meas']})" if r.get("verdict_meas")
                  else ""))
        ax.set(xscale="log", xlabel="L_cp (chunks)", ylabel="latency (ms)",
               title=f"{name}  [{part}]\n{sub}")
        ax.title.set_fontsize(9)
        # Tick the shape's own grid, thinned to at most six labels and written as
        # plain chunk counts: a short shape spans well under a decade, where the
        # default log labelling reads "4x10^0" and runs its minor labels together.
        grid = sorted(int(x) for x in g["L_cp"].unique())
        ax.set_xticks(grid[::-(-len(grid) // 6)])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_facecolor(_PART_BG.get(part, "#ffffff"))
        ax.grid(alpha=.25, which="both")
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=7)
    for ax in axes.ravel()[len(shapes):]:
        ax.axis("off")
    fig.suptitle("autocp: fitted latency vs L_cp per shape "
                 "(green panels = held-out eval shapes)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_parity(plt, matplotlib, df: pd.DataFrame, coefs: dict,
                 path: str) -> str:
    """Per kernel: predicted vs measured, train and eval on the same identity line.

    Where the curves figure shows whether the *choice* is right, this shows
    whether the *model* is -- and it is where the held-out shapes earn their
    keep: eval markers straying off the diagonal that train markers hug is
    overfitting, visible before any MAPE column is read.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9))
    parts = _parts(df).to_numpy()
    for ax, k in zip(axes.ravel(), KERNELS):
        priced = df.index.isin(fit_rows(df, k).index)
        y, yhat = df[k].to_numpy(float), _predict_column(coefs, df, k)
        tags = []
        # eval is drawn hollow and on top: the two sets overlap heavily, and a
        # filled marker large enough to spot would bury the train points under it.
        for part, style in (
                ("train", dict(marker="o", s=24, alpha=.7,
                               c=_KERNEL_COLOR[k])),
                ("eval", dict(marker="*", s=75, facecolors="none",
                              edgecolors="k", linewidths=.8, zorder=5))):
            m = priced & (parts == part)
            if not m.any():
                continue
            ax.scatter(y[m], yhat[m], label=f"{part} (n={int(m.sum())})", **style)
            tags.append(f"{part} {mape(y, yhat, m):.1f}%")
        lo, hi = y[priced].min() * .7, y[priced].max() * 1.3
        ax.plot([lo, hi], [lo, hi], "k-", lw=.8)
        for f in (1.1, 1 / 1.1):
            ax.plot([lo, hi], [lo * f, hi * f], "k--", lw=.6, alpha=.45)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
               xlabel="measured (ms)", ylabel="predicted (ms)",
               title=f"{k}\nMAPE  {'   '.join(tags)}")
        ax.title.set_fontsize(10)
        for axis in (ax.xaxis, ax.yaxis):  # log minor labels collide otherwise
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=.25, which="both")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("autocp latency model: per-kernel parity, fitted on train only "
                 "(dashed = +/-10%)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_fit(df: pd.DataFrame, coefs: dict, dec: pd.DataFrame,
             outdir: str) -> list[str]:
    """Write ``curves.png`` and ``parity.png`` under ``outdir``."""
    matplotlib, plt = _pyplot()
    os.makedirs(outdir, exist_ok=True)
    df = df.copy()
    df["meas_total"], df["pred_total"] = _totals(coefs, df)
    return [
        _plot_curves(plt, matplotlib, df, dec,
                     os.path.join(outdir, "curves.png")),
        _plot_parity(plt, matplotlib, df, coefs,
                     os.path.join(outdir, "parity.png")),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_env(args, cache: dict):
    """Settle ``(device, P)``: from the flags, else the cache, else the GPU."""
    device, P = args.device, args.sm_count
    if device is None or P is None:
        if args.from_cache:
            devs = {r["device"] for r in cache.values()}
            Ps = {int(r["P"]) for r in cache.values()}
            if device is None:
                if len(devs) != 1:
                    raise SystemExit(
                        f"--from-cache needs --device: cache holds {sorted(devs)}")
                device = devs.pop()
            if P is None:
                if len(Ps) != 1:
                    raise SystemExit(
                        f"--from-cache needs --sm-count: cache holds {sorted(Ps)}")
                P = Ps.pop()
        else:
            import torch  # only the measuring path may need torch
            if device is None:
                device = torch.cuda.get_device_name()
            if P is None:
                P = torch.cuda.get_device_properties().multi_processor_count
    return device, int(P)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="fit the autocp latency model from shape CSVs "
                    "(measuring the kernels as needed)")
    ap.add_argument("--train", required=True, help="training shape CSV")
    ap.add_argument("--eval", dest="eval_path", default=None,
                    help="held-out shape CSV (accuracy + decision quality only)")
    ap.add_argument("--out", default=None, help="output coefficient CSV")
    ap.add_argument("--cache", default=None,
                    help="measurement cache CSV "
                         "(default debug/autocp_cache_<device>.csv)")
    ap.add_argument("--sm-count", type=int, default=None,
                    help="P; defaults to this device's SM count")
    ap.add_argument("--device", default=None,
                    help="device name for the cache key; defaults to this GPU")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lcp-ratio", type=float, default=None,
                    help="geometric step of the L_cp grid (default 1.5)")
    ap.add_argument("--Hk", type=int, default=None,
                    help="key heads when the CSV has no Hk column (default: H)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--swa-ratio", type=float, default=0.0)
    ap.add_argument("--warmup-ms", type=float, default=25)
    ap.add_argument("--rep-ms", type=float, default=100)
    ap.add_argument("--measure-only", action="store_true",
                    help="fill the cache and stop")
    ap.add_argument("--from-cache", action="store_true",
                    help="fit from the cache only; error on a missing row "
                         "(never imports torch)")
    ap.add_argument("--report-csv", default=None,
                    help="also write the accuracy / decision tables here "
                         "(<stem>_accuracy.csv, <stem>_decision.csv)")
    ap.add_argument("--plot", default=None, metavar="DIR",
                    help="write curves.png / parity.png here (needs matplotlib, "
                         "which is not a flash_qla dependency)")
    args = ap.parse_args()

    if args.lcp_ratio is None:
        from .utils import DEFAULT_LCP_RATIO
        args.lcp_ratio = DEFAULT_LCP_RATIO
    if not args.measure_only and not args.out:
        raise SystemExit("--out is required unless --measure-only is given")
    if args.plot:
        _pyplot()  # fail on a missing matplotlib now, not after measuring

    kw = dict(Hk=args.Hk, dtype=args.dtype, swa_ratio=args.swa_ratio)
    groups = [("train", load_specs(args.train, **kw))]
    if args.eval_path:
        groups.append(("eval", load_specs(args.eval_path, **kw)))
    clash = ({s.name for s in groups[0][1]} & {s.name for s in groups[-1][1]}
             if len(groups) > 1 else set())
    if clash:
        raise SystemExit(f"shape names appear in both --train and --eval: "
                         f"{sorted(clash)}")

    cache_path = args.cache
    if cache_path is None and args.device:
        cache_path = default_cache_path(args.device)
    cache = read_cache(cache_path) if cache_path else {}
    device, P = _resolve_env(args, cache)
    if cache_path is None:
        cache_path = default_cache_path(device)
        cache = read_cache(cache_path)

    specs = [s for _, g in groups for s in g]
    if args.from_cache:
        print(f"cache: {cache_path} ({len(cache)} rows, measuring disabled)")
    else:
        from .measure import measure_specs
        cache = measure_specs(specs, cache_path=cache_path, P=P,
                              chunk=args.chunk, ratio=args.lcp_ratio,
                              device=device, warmup_ms=args.warmup_ms,
                              rep_ms=args.rep_ms)
        if args.measure_only:
            print(f"\nmeasure-only: cache is at {cache_path}")
            return

    frames = []
    for part, g in groups:
        rows = rows_for_specs(cache, g, P=P, chunk=args.chunk,
                              ratio=args.lcp_ratio, device=device)
        df = build_frame(rows, P, args.chunk, part=part)
        frames.append(df)
        print(f"{part + ':':6s} {len(g)} shapes, {len(df)} rows "
              f"({int(df['cp'].sum())} CP, "
              f"{int((~df['cp']).sum())} baseline)")
    train = frames[0]
    allrows = pd.concat(frames, ignore_index=True)

    coefs = fit_all(train, P, args.chunk)
    print(f"\n=== coefficients (P={P}, chunk={args.chunk}) ===")
    print(f"fitted on train only; baseline rows held out of "
          f"{', '.join(CP_ONLY_KERNELS)}")
    for k in KERNELS:
        c = coefs["kernels"][k]
        cell = lambda n: (f"{c[n]:.7f}" if n in COEF_NAMES[k] else "     --    ")
        print(f"  {k:11s} tau={cell('tau')}  kappa={cell('kappa')}  "
              f"c={c['c']:.6f}")

    acc = accuracy_report(allrows, coefs)
    print("\n=== MAPE% per setting x kernel (rows that price the kernel) ===")
    print(acc.round(2).to_string(index=False))
    print("\nmean by part x kernel:")
    print(acc.groupby(["part", "kernel"])["mape"].mean().unstack()
          .round(2).to_string())

    dec = decision_report(allrows, coefs, P)
    print("\n=== decision quality (on the measured grid) ===")
    print(dec.round(2).to_string(index=False))
    bad = dec[dec["verdict"] != dec["verdict_meas"]].dropna(subset=["verdict_meas"])
    if len(bad):
        print(f"\n!! {len(bad)} setting(s) where the verdict disagrees with the "
              f"measurement: {sorted(bad['setting'])}")

    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        save_coefs(args.out, coefs)
        print(f"\ncoefficients written: {args.out}")
    if args.report_csv:
        stem = os.path.splitext(args.report_csv)[0]
        acc.to_csv(f"{stem}_accuracy.csv", index=False)
        dec.to_csv(f"{stem}_decision.csv", index=False)
        print(f"reports written: {stem}_accuracy.csv, {stem}_decision.csv")
    # Last, so that a plotting failure cannot cost the coefficients a run paid
    # tens of minutes of GPU time for.
    if args.plot:
        for p in plot_fit(allrows, coefs, dec, args.plot):
            print(f"figure written: {p}")


if __name__ == "__main__":
    main()
