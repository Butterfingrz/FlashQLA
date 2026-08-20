# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""The online intra-card CP decision: pick ``L_cp`` and decide CP on/off.

Called from ``cp/context.py`` on the launch path, so this module is standard
library only. The structural features, the model form and the coefficient CSV
live in :mod:`.utils` -- shared verbatim with the offline fit, which is what
makes "the fit saw exactly the features the decision sees" true by construction
rather than by an equivalence test.
"""

from __future__ import annotations

import functools
import math

from .utils import (
    FWD_KERNELS,
    KERNELS,
    MIN_LCP,
    CP_ONLY_KERNELS,
    StructFeatures,
    load_coefs,
    predict_kernel,
    seq_chunks,
    struct_features,
)

MAX_EVALS = 3


@functools.lru_cache(maxsize=4)
def _default_coefs(path: str | None) -> dict:
    return load_coefs(path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(coefs: dict, feat: StructFeatures, target: str = "all",
            baseline: bool = False) -> float:
    """Total predicted milliseconds for one candidate shape.

    ``baseline=True`` prices *CP switched off*, not "CP on with one partition per
    sequence": the two differ by the whole of :data:`CP_ONLY_KERNELS`, which
    the CP-off path never launches. At ``S = cmax`` the fused kernels see a
    byte-identical shape either way, so ``d_fb`` / ``u`` carry over unchanged and
    no separate baseline coefficients are needed.
    """
    if target not in ("fwd", "all"):
        raise ValueError(f"target must be 'fwd' or 'all', got {target!r}")
    priced = FWD_KERNELS if target == "fwd" else KERNELS
    return sum(predict_kernel(coefs, feat, k) for k in priced
               if not (baseline and k in CP_ONLY_KERNELS))


def wave_candidates(chunks, H: int, P: int, min_S: int = MIN_LCP):
    cmax, total = max(chunks), sum(chunks)
    if min_S > cmax:
        return
    last = None  # consecutive waves can share an S once the steps get small
    for w in range(1, max(1, math.ceil(total * H / (min_S * P))) + 1):
        s = min(cmax, max(min_S, math.ceil(total * H / (w * P))))
        if last is None or s < last:
            last = s
            yield s
        if s <= min_S:
            return


# def latency_floor(coefs: dict, target: str, chunks, H: int, P: int,
#                   S: int, u: float) -> float:
#     K = coefs["kernels"]
#     p, co, fw = K["prepare_h"], K["correct_h0"], K["fused_fwd"]
#     tau_fb, kappa, const = fw["tau"], p["kappa"] + fw["kappa"], p["c"] + fw["c"]
#     if target == "all":
#         bw = K["fused_bwd"]
#         tau_fb += bw["tau"]
#         kappa += bw["kappa"]
#         const += bw["c"]
#     total = sum(chunks)
#     return (tau_fb * H * total / P
#             + p["tau"] * H * max(0, total - len(chunks) * S) / P
#             + kappa * (math.ceil(u) - 1)
#             + const + co["c"])

def decide(cu_seqlens=None, seq_lens=None, num_chunks=None, num_v_heads=None,
           coefs=None, coefs_path=None, P=None, chunk=None, margin=0.05,
           target="all", min_S=MIN_LCP, max_evals=MAX_EVALS, candidates=None,
           prune=True, debug=False):
    if coefs is None:
        coefs = _default_coefs(coefs_path)
    if P is None:
        P = coefs["P"]
    if chunk is None:
        chunk = coefs["chunk"]
    if num_v_heads is None:
        raise ValueError("num_v_heads is required")

    if num_chunks is not None:
        chunks = [int(c) for c in num_chunks]
    elif cu_seqlens is not None:
        chunks = seq_chunks(cu_seqlens, chunk)
    elif seq_lens is not None:
        chunks = [math.ceil(int(x) / chunk) for x in seq_lens]
    else:
        raise ValueError("provide one of cu_seqlens / seq_lens / num_chunks")
    if not chunks or min(chunks) <= 0:
        raise ValueError(f"invalid sequence lengths: {chunks}")

    H, cmax = int(num_v_heads), max(chunks)
    base_feat = struct_features(chunks, cmax, H, P)
    base_t = predict(coefs, base_feat, target, baseline=True)

    if candidates is not None:
        cands, budget = sorted(set(candidates)), len(candidates)
    else:
        cands, budget = wave_candidates(chunks, H, P, min_S), max(1, max_evals)

    sweep, best_feat, best_t = [], None, float("inf")
    for S in cands:
        feat = struct_features(chunks, S, H, P)
        t = predict(coefs, feat, target, baseline=not feat.enable_cp)
        sweep.append((S, t))
        if feat.enable_cp and t < best_t:
            best_feat, best_t = feat, t
        if best_feat is None:
            continue
        # if len(sweep) >= budget or (
        #         prune and latency_floor(coefs, target, chunks, H, P, S,
        #                                 feat.u) > best_t):
        if len(sweep) >= budget:
            break
    sweep.sort()

    if best_feat is None:
        use_cp, best_lcp = False, None
    else:
        use_cp, best_lcp = best_t < base_t * (1.0 - margin), int(best_feat.S)
    lcp = best_lcp if use_cp else None

    if not debug:
        return use_cp, lcp
    return {
        "use_cp": use_cp,
        "lcp": lcp,
        "best_cp_lcp": best_lcp,
        "best_cp_ms": best_t,  # inf when no CP point exists
        "nocp_pred_ms": base_t,
        "sweep": sweep,
    }
