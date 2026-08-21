# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from flash_qla.ops.gated_delta_rule.chunk.cp.autocp.decision import decide
from flash_qla.ops.gated_delta_rule.chunk.cp.autocp.utils import (
    COEF_NAMES,
    CP_ONLY_ROWS,
    INFER_ROWS,
    MODEL_TERMS,
    ROWS,
    ROW_SOURCES,
    TRAIN_ROWS,
    struct_features,
)
from .utils import (
    MEASURED,
    cache_key,
    cache_row_complete,
    default_cache_path,
    lcp_grid,
    load_specs,
    read_cache,
    save_coefs,
)

__all__ = ["build_frame", "rows_for_specs", "fit_kernel", "fit_rows", "fit_all",
           "mape", "accuracy_report", "decision_report", "nocp_measured",
           "plot_fit", "main", "CP_ONLY_ROWS", "STEP_SITES", "SOURCE_OF",
           "MODES"]

# build_frame's feature columns. d_p_fwd / d_p_bidi exist only here: the online
# struct_features carries the mode-appropriate depth in the single field d_p.
FEATURE_COLUMNS = ("u", "d_fb", "d_p_fwd", "d_p_bidi", "d_p_bwd", "d_corr")

# Measured column -> (coefficient row, depth feature column); inverse of ROW_SOURCES.
SOURCE_OF = {col: (row, depth)
             for row, sources in ROW_SOURCES.items()
             for col, depth in sources}

# Per mode, the launches a step makes as (row, measured column, depth column) --
# INFER_ROWS / TRAIN_ROWS resolved down to the timed call sites (_check_taxonomy
# enforces order and multiplicity).
STEP_SITES = {
    "infer": (("prepare_h", "prepare_h", "d_p_fwd"),
              ("correct_h0", "correct_h0", "d_corr"),
              ("fused_fwd", "fused_fwd", "d_fb")),
    "train": (("prepare_h", "prepare_h_bidi", "d_p_bidi"),
              ("correct_h0", "correct_h0", "d_corr"),
              ("fused_fwd", "fused_fwd", "d_fb"),
              ("prepare_dh", "prepare_dh", "d_p_bwd"),
              ("correct_dht", "correct_dht", "d_corr"),
              ("recompute_h", "recompute_h", "d_fb"),
              ("fused_bwd", "fused_bwd", "d_fb")),
}

# The cached total each mode is scored against (written by measure.py).
MODES = {"infer": "fwd_total", "train": "all_total"}


def _fields(row: str, source) -> list[str]:
    depth_field = MODEL_TERMS[row][0][0]
    _, depth_col = source
    return [depth_col if field == depth_field else field
            for field, _ in MODEL_TERMS[row]]


def _check_taxonomy() -> None:
    for row, sources in ROW_SOURCES.items():
        if row not in MODEL_TERMS:
            raise AssertionError(f"ROW_SOURCES row {row!r} has no MODEL_TERMS entry")
        for col, depth in sources:
            if col not in MEASURED:
                raise AssertionError(f"row {row!r} names unmeasured column {col!r}")
            for field in _fields(row, (col, depth)):
                if field not in FEATURE_COLUMNS:
                    raise AssertionError(
                        f"row {row!r} source {col!r} needs feature {field!r}, "
                        f"which build_frame does not produce")
    if set(ROW_SOURCES) != set(ROWS):
        raise AssertionError("ROW_SOURCES and ROWS name different rows")
    if set(SOURCE_OF) != set(MEASURED):
        raise AssertionError(
            f"every measured column must fit exactly one row; "
            f"unclaimed {sorted(set(MEASURED) - set(SOURCE_OF))}")
    for mode, sites in STEP_SITES.items():
        want = {"infer": INFER_ROWS, "train": TRAIN_ROWS}[mode]
        if tuple(r for r, _, _ in sites) != tuple(want):
            raise AssertionError(f"STEP_SITES[{mode!r}] does not match {want}")
        for row, col, depth in sites:
            if (col, depth) not in ROW_SOURCES[row]:
                raise AssertionError(
                    f"STEP_SITES[{mode!r}] uses source ({col!r}, {depth!r}), "
                    f"which is not one of row {row!r}'s")
    if set(MODES) != set(STEP_SITES):
        raise AssertionError("MODES and STEP_SITES name different modes")


_check_taxonomy()


def fit_rows(df: pd.DataFrame, row: str) -> pd.DataFrame:
    return df[df["cp"]] if row in CP_ONLY_ROWS else df


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
            if row is None or not cache_row_complete(row):
                missing.append((spec.name, lcp))
            else:
                rows.append({**row, "name": spec.name})
    if missing and strict:
        raise SystemExit(
            f"{len(missing)} measurement(s) missing or incomplete in the cache "
            f"(a row is incomplete when any of {len(MEASURED)} timed columns is "
            f"absent), e.g. {missing[:6]}. Drop --from-cache to measure them.")
    return rows


def build_frame(rows, P: int, chunk: int = 64, part: str | None = None
                ) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        raise ValueError("no measured rows")
    df["setting"] = df["name"]
    if part is not None:
        df["part"] = part

    # Features twice per row, once per mode: only the warmup depth (d_p) differs
    # -- fwd-only for inference, bidi max for training -- kept under distinct names.
    args = [([int(c) for c in _chunks_of(r["cu"], chunk)],
             int(r["L_cp"]), int(r["H"]), int(r["warmup_max"]))
            for _, r in df.iterrows()]
    feats = [struct_features(c, S, H, P, warmup=w, is_train=False)
             for c, S, H, w in args]
    feats_tr = [struct_features(c, S, H, P, warmup=w, is_train=True)
                for c, S, H, w in args]

    df["u"] = [f.u for f in feats]
    df["d_fb"] = [f.d_fb for f in feats]
    df["d_p_fwd"] = [f.d_p for f in feats]
    df["d_p_bidi"] = [f.d_p for f in feats_tr]
    df["d_p_bwd"] = [f.d_p_bwd for f in feats_tr]
    df["d_corr"] = [f.d_corr for f in feats]
    df["cp"] = [f.enable_cp for f in feats]
    missing = [c for c in FEATURE_COLUMNS if c not in df]
    if missing:
        raise AssertionError(f"build_frame did not produce {missing}")

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
def _design(df: pd.DataFrame, row: str, source):
    X = np.stack([df[f].to_numpy(float) for f in _fields(row, source)]
                 + [np.ones(len(df))], axis=1)
    return X, df[source[0]].to_numpy(float)


def fit_kernel(df: pd.DataFrame, row: str, settings=None) -> np.ndarray:
    sub = df if settings is None else df[df["setting"].isin(settings)]
    sub = fit_rows(sub, row)
    blocks = [_design(sub, row, s) for s in ROW_SOURCES[row]]
    X = np.concatenate([b[0] for b in blocks], axis=0)
    y = np.concatenate([b[1] for b in blocks], axis=0)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef[0] = max(coef[0], 0.0)
    return coef


def fit_all(df: pd.DataFrame, P: int, chunk: int = 64) -> dict:
    kernels = {}
    for k in ROWS:
        coef = fit_kernel(df, k)
        packed = dict(tau=0.0, kappa=0.0, c=0.0)
        packed.update(zip(COEF_NAMES[k], (float(x) for x in coef)))
        kernels[k] = packed
    return {"kernels": kernels, "P": int(P), "chunk": int(chunk)}


def _predict_column(coefs: dict, df: pd.DataFrame, row: str,
                    source) -> np.ndarray:
    X, _ = _design(df, row, source)
    vec = np.array([coefs["kernels"][row][c] for c in COEF_NAMES[row]])
    return X @ vec


def mape(y: np.ndarray, yhat: np.ndarray, mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / np.abs(y[mask])) * 100)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def accuracy_report(df: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    keys = (["part", "setting"] if "part" in df else ["setting"])
    out = []
    for col in MEASURED:
        row, depth = SOURCE_OF[col]
        y = df[col].to_numpy(float)
        yhat = _predict_column(coefs, df, row, (col, depth))
        priced = df.index.isin(fit_rows(df, row).index)
        for vals, g in df.groupby(keys, sort=False):
            vals = vals if isinstance(vals, tuple) else (vals,)
            sel = df.index.isin(g.index) & priced
            out.append({**dict(zip(keys, vals)), "site": col, "row": row,
                        "mape": mape(y, yhat, sel), "n_fit": int(sel.sum())})
    return pd.DataFrame(out).sort_values([*keys, "site"], kind="stable")


def nocp_measured(base_rows: pd.DataFrame, mode: str = "train") -> float:
    cols = [c for r, c, _ in STEP_SITES[mode] if r not in CP_ONLY_ROWS]
    return float(sum(base_rows[c].iloc[0] for c in cols))


def decision_report(df: pd.DataFrame, coefs: dict, P: int,
                    mode: str = "train") -> pd.DataFrame:
    total_col = MODES[mode]
    out = []
    for label, sub in df.groupby("setting", sort=True):
        H = int(sub["H"].iloc[0])
        cu = [int(x) for x in str(sub["cu"].iloc[0]).split(":")]
        cp = sub[sub["cp"]]
        base = sub[~sub["cp"]]
        kw = dict(cu_seqlens=cu, num_v_heads=H, coefs=coefs, P=P, debug=True,
                  is_train=(mode == "train"))

        rec = dict(setting=label, mode=mode)
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
            at_model = float(cp[cp["L_cp"] == S][total_col].iloc[0])
            best = float(cp[total_col].min())
            rec.update(
                lcp_model=S,
                lcp_meas=int(cp.loc[cp[total_col].idxmin(), "L_cp"]),
                regret_pct=(at_model / best - 1.0) * 100)
        else:
            rec.update(lcp_model=None, lcp_meas=None, regret_pct=float("nan"))

        if len(base):
            t_base = nocp_measured(base, mode)
            rec["base_err_pct"] = (r["nocp_pred_ms"] / t_base - 1.0) * 100
            rec["verdict_meas"] = ("CP" if len(cp)
                                   and float(cp[total_col].min()) < t_base
                                   else "no-CP")
        else:
            rec["base_err_pct"] = float("nan")
            rec["verdict_meas"] = None
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# Panel tint per part (held-out eval shapes read differently from train).
_PART_BG = {"train": "#ffffff", "eval": "#eef4ee"}

# One colour per call site; two sources of one row get two tints of one hue.
_SITE_COLOR = {"prepare_h": "#4C78A8", "prepare_h_bidi": "#8AB4D8",
               "correct_h0": "#F58518", "correct_dht": "#FBBE79",
               "fused_fwd": "#54A24B", "fused_bwd": "#E45756",
               "prepare_dh": "#B279A2", "recompute_h": "#9D755D"}


def _pyplot():
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


def _totals(coefs: dict, df: pd.DataFrame, mode: str = "train"):
    sites = STEP_SITES[mode]
    base_sites = tuple(s for s in sites if s[0] not in CP_ONLY_ROWS)
    meas = lambda ss: df[[c for _, c, _ in ss]].sum(axis=1).to_numpy(float)
    pred = lambda ss: sum(_predict_column(coefs, df, r, (c, d))
                          for r, c, d in ss)
    cp = df["cp"].to_numpy(bool)
    return (np.where(cp, meas(sites), meas(base_sites)),
            np.where(cp, pred(sites), pred(base_sites)))


def _plot_curves(plt, matplotlib, df: pd.DataFrame, dec: pd.DataFrame,
                 path: str, mode: str = "train") -> str:
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
        # tick the shape's own grid as plain chunk counts, thinned to <=6 labels
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
    fig.suptitle(f"autocp [{mode}]: fitted latency vs L_cp per shape "
                 f"(green panels = held-out eval shapes)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_parity(plt, matplotlib, df: pd.DataFrame, coefs: dict,
                 path: str) -> str:
    ncol = min(3, len(MEASURED))
    nrow = -(-len(MEASURED) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.75 * ncol, 4.5 * nrow),
                            squeeze=False)
    parts = _parts(df).to_numpy()
    for ax, k in zip(axes.ravel(), MEASURED):
        row, depth = SOURCE_OF[k]
        priced = df.index.isin(fit_rows(df, row).index)
        y = df[k].to_numpy(float)
        yhat = _predict_column(coefs, df, row, (k, depth))
        tags = []
        # eval drawn hollow and on top so it isn't buried under the train points
        for part, style in (
                ("train", dict(marker="o", s=24, alpha=.7,
                               c=_SITE_COLOR[k])),
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
               title=f"{k}   [row {row}, d={depth}]\n"
                     f"MAPE  {'   '.join(tags)}")
        ax.title.set_fontsize(10)
        for axis in (ax.xaxis, ax.yaxis):  # log minor labels collide otherwise
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=.25, which="both")
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes.ravel()[len(MEASURED):]:
        ax.axis("off")
    fig.suptitle("autocp latency model: per-call-site parity, fitted on train only "
                 "(dashed = +/-10%; sources sharing a row share a hue)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_fit(df: pd.DataFrame, coefs: dict, decs: dict,
             outdir: str) -> list[str]:
    matplotlib, plt = _pyplot()
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for mode, dec in decs.items():
        d = df.copy()
        d["meas_total"], d["pred_total"] = _totals(coefs, d, mode)
        paths.append(_plot_curves(plt, matplotlib, d, dec,
                                  os.path.join(outdir, f"curves_{mode}.png"),
                                  mode))
    paths.append(_plot_parity(plt, matplotlib, df, coefs,
                              os.path.join(outdir, "parity.png")))
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_env(args, cache: dict):
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
                    help="write curves_<mode>.png / parity.png here (needs "
                         "matplotlib, which is not a flash_qla dependency)")
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
          f"{', '.join(CP_ONLY_ROWS)}")
    for k in ROWS:
        c = coefs["kernels"][k]
        cell = lambda n: (f"{c[n]:.7f}" if n in COEF_NAMES[k] else "     --    ")
        srcs = ", ".join(col for col, _ in ROW_SOURCES[k])
        print(f"  {k:11s} tau={cell('tau')}  kappa={cell('kappa')}  "
              f"c={c['c']:.6f}   <- {srcs}")

    acc = accuracy_report(allrows, coefs)
    print("\n=== MAPE% per setting x call site (rows that price it) ===")
    print(acc.round(2).to_string(index=False))
    print("\nmean by part x call site:")
    print(acc.groupby(["part", "site"])["mape"].mean().unstack()
          .round(2).to_string())

    decs = {}
    for mode in MODES:
        dec = decision_report(allrows, coefs, P, mode)
        decs[mode] = dec
        print(f"\n=== decision quality, {mode} "
              f"(model vs the measured {MODES[mode]} grid) ===")
        print(dec.round(2).to_string(index=False))
        bad = dec[dec["verdict"] != dec["verdict_meas"]].dropna(
            subset=["verdict_meas"])
        if len(bad):
            print(f"!! {len(bad)} setting(s) where the {mode} verdict disagrees "
                  f"with the measurement: {sorted(bad['setting'])}")

    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        save_coefs(args.out, coefs)
        print(f"\ncoefficients written: {args.out}")
    if args.report_csv:
        stem = os.path.splitext(args.report_csv)[0]
        acc.to_csv(f"{stem}_accuracy.csv", index=False)
        # both modes in one file, distinguished by the mode column
        pd.concat(decs.values(), ignore_index=True).to_csv(
            f"{stem}_decision.csv", index=False)
        print(f"reports written: {stem}_accuracy.csv, {stem}_decision.csv")
    # last, so a plotting failure can't cost a coefficient run its GPU time
    if args.plot:
        for p in plot_fit(allrows, coefs, decs, args.plot):
            print(f"figure written: {p}")


if __name__ == "__main__":
    main()
