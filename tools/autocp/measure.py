# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Isolated timing of the four intra-CP kernels, one row per ``(shape, L_cp)``.

This is the measurement half of :mod:`.fit`: it turns a ``ShapeSpec`` plus an
``L_cp`` grid into cache rows, and knows nothing about least squares. Needs torch
/ tilelang -- which is half the reason it lives out here in ``tools/`` rather than
in the shipped package: nothing on the launch path may import it.

Two things it deliberately does *not* do itself:

* **It does not build the CP context.** ``context._build_intra_cp_context`` does,
  which is the same call the launch path makes -- so what gets timed is the
  partitioning that actually ships, not a copy of it that can drift. Reaching a
  private name across the package boundary is the price of that guarantee; the
  alternative is a second partitioner here that can silently disagree.
* **It does not re-derive the structural features.** The fit does that from the
  cached shape columns via ``autocp.utils.struct_features``.
"""

from __future__ import annotations

import math
from typing import Iterator

import tilelang
import torch
import torch.nn.functional as F

from flash_qla.ops.utils import chunk_local_cumsum
from flash_qla.utils import l2norm

from flash_qla.ops.gated_delta_rule.chunk import (  # noqa: F401 -- re-exports
    CHUNK_SIZE,
    correct_initial_states,
    fused_gdr_bwd,
    fused_gdr_fwd,
    fused_gdr_h,
    get_warmup_chunks,
    kkt_solve,
)
from flash_qla.ops.gated_delta_rule.chunk.cp.context import (
    _build_intra_cp_context,
)
from flash_qla.ops.gated_delta_rule.chunk.cp.autocp.utils import KERNELS
from .utils import (
    ShapeSpec, lcp_grid, append_cache, cache_key, read_cache,
)

#: Head dims the calibration runs at (the shipped kernels' configuration).
K_DIM = V_DIM = 128

_DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
           "float16": torch.float16, "fp16": torch.float16, "half": torch.float16}


def torch_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[str(name).strip().lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}; "
                         f"expected one of {sorted(_DTYPES)}") from None


def bench(fn, warmup_ms: float = 25, rep_ms: float = 100) -> float:
    """Milliseconds per call.

    ``do_bench``'s ``warmup`` / ``rep`` are *target milliseconds*, not iteration
    counts: it estimates one call, sizes the loop from that, flushes L2 between
    reps and averages CUDA-event pairs. The first call triggers tilelang JIT and
    can take minutes.
    """
    return tilelang.profiler.do_bench(fn, warmup=warmup_ms, rep=rep_ms)


def generate_inputs(T: int, Hk: int, Hv: int, dtype: torch.dtype,
                    swa_ratio: float, seed: int = 42) -> dict:
    """One shape's inputs. Matches ``profile_inter_cp.generate_inputs``."""
    dev = "cuda"
    torch.manual_seed(seed)
    q = l2norm(torch.randn(1, T, Hk, K_DIM, device=dev, dtype=dtype))
    k = l2norm(torch.randn(1, T, Hk, K_DIM, device=dev, dtype=dtype))
    v = torch.randn(1, T, Hv, V_DIM, device=dev, dtype=dtype)
    do = torch.randn(1, T, Hv, V_DIM, device=dev, dtype=dtype)
    g = F.logsigmoid(torch.randn(1, T, Hv, device=dev, dtype=torch.float32)) / 16
    beta = torch.randn(1, T, Hv, device=dev, dtype=torch.float32).sigmoid()
    # SWA heads get gate 0 (no decay), the rest keep theirs -- this is what sets
    # the warmup-length distribution, and hence ``warmup_max``.
    swa = torch.zeros(Hv, dtype=torch.bool, device=dev)
    swa[: math.ceil(swa_ratio * Hv)] = 1
    swa = swa[torch.randperm(Hv, device=dev)]
    g[:, :, ~swa] = 0.0
    return dict(q=q, k=k, v=v, do=do, g=g, beta=beta, scale=K_DIM ** -0.5)


def measure_shape(spec: ShapeSpec, lcps, *, P: int, chunk: int = CHUNK_SIZE,
                  device: str | None = None, warmup_ms: float = 25,
                  rep_ms: float = 100, seed: int = 42) -> Iterator[dict]:
    """Time the four kernels at each ``L_cp`` in ``lcps``. Yields cache rows.

    The inputs, the gate cumsum and the KKT solve are built **once per shape**:
    all three use the *raw* ``cu_seqlens`` and so are independent of ``L_cp``
    (this mirrors ``chunk_gated_delta_rule_fwd``, which also computes them before
    any CP context). Only the CP context, the warmup scan and the two untimed
    ``fused_gdr_h`` setup calls are per-``L_cp``.
    """
    dev = "cuda"
    if device is None:
        device = torch.cuda.get_device_name()
    dtype = torch_dtype(spec.dtype)
    cu_raw = torch.tensor(spec.cu, device=dev, dtype=torch.int32)
    chunks = spec.chunks(chunk)

    inp = generate_inputs(spec.T, spec.Hk, spec.H, dtype, spec.swa_ratio, seed)
    g_c = chunk_local_cumsum(inp["g"], cu_seqlens=cu_raw, chunk_size=chunk)
    A = kkt_solve(inp["k"], inp["beta"], cu_seqlens=cu_raw, chunk_size=chunk)

    for lcp in lcps:
        ctx = _build_intra_cp_context(cu_raw, chunk, chunks, int(lcp))
        cp_cu = ctx.intra_cp_cu_seqlens
        n_part = cp_cu.numel() - 1

        nw, fb_mask = get_warmup_chunks(
            g=g_c, cu_seqlens=cp_cu, ht_mask=ctx.ht_mask,
            chunk_size=chunk, threshold=-10.0,
        )

        # Untimed setup: prepare_h's outputs feed correct_h0, whose output feeds
        # the fused kernels, and fused_bwd needs the recomputed h states.
        _, ht, mt = fused_gdr_h(
            k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
            initial_state=None, output_final_state=True, output_h=False,
            cu_seqlens=cp_cu, num_warmup_chunks=nw,
        )
        cp_h0 = correct_initial_states(
            raw_h0=None, ht_buffer=ht, mt_buffer=mt,
            fallback_mask=fb_mask, seq_map_r2c=ctx.seq_map_r2c,
        )
        h_states, _, _ = fused_gdr_h(
            k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
            initial_state=cp_h0, output_final_state=False, output_h=True,
            cu_seqlens=cp_cu,
        )

        def run_prepare_h():
            return fused_gdr_h(
                k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
                initial_state=None, output_final_state=True, output_h=False,
                cu_seqlens=cp_cu, num_warmup_chunks=nw,
            )

        def run_correct_h0():
            return correct_initial_states(
                raw_h0=None, ht_buffer=ht, mt_buffer=mt,
                fallback_mask=fb_mask, seq_map_r2c=ctx.seq_map_r2c,
            )

        def run_fused_fwd():
            return fused_gdr_fwd(
                q=inp["q"], k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
                scale=inp["scale"], initial_state=cp_h0,
                output_final_state=True, output_h=False, output_o=True,
                cu_seqlens=cp_cu, cp_seq_map=ctx.seq_map_c2r,
                raw_cu_seqlens=cu_raw,
            )

        def run_fused_bwd():
            return fused_gdr_bwd(
                q=inp["q"], k=inp["k"], v=inp["v"], a=A, g=g_c, b=inp["beta"],
                do=inp["do"], dht=None, h=h_states, scale=inp["scale"],
                cu_seqlens=cp_cu,
            )

        t = {name: bench(fn, warmup_ms, rep_ms) for name, fn in (
            ("prepare_h", run_prepare_h), ("correct_h0", run_correct_h0),
            ("fused_fwd", run_fused_fwd), ("fused_bwd", run_fused_bwd))}

        yield dict(
            name=spec.name, device=device, P=int(P), chunk=int(chunk),
            dtype=spec.dtype, swa_ratio=f"{spec.swa_ratio:g}",
            H=spec.H, Hk=spec.Hk, T=spec.T, Lc=sum(chunks), L_cp=int(lcp),
            N_part=n_part, B_raw=spec.B_raw, cmax=max(chunks), cu=spec.cu_str,
            warmup_max=int(nw.max().item()),
            warmup_mean=float(nw.float().mean().item()),
            **t,
            fwd_total=t["prepare_h"] + t["correct_h0"] + t["fused_fwd"],
            all_total=sum(t[k] for k in KERNELS),
        )


def plan_cases(specs, *, P: int, chunk: int, ratio: float, device: str,
               cache: dict) -> tuple[list[tuple], int]:
    """``[(spec, [lcp, ...])]`` still to measure, plus the number of cache hits.

    Splitting planning from measuring is what lets ``fit`` report "N hit, M to
    measure" before touching the GPU -- and lets ``--from-cache`` name exactly
    which rows are missing without importing torch.
    """
    todo, hits = [], 0
    for spec in specs:
        want = []
        for lcp in lcp_grid(spec.cmax(chunk), ratio):
            key = cache_key(dict(device=device, P=P, chunk=chunk,
                                 dtype=spec.dtype, swa_ratio=spec.swa_ratio,
                                 H=spec.H, Hk=spec.Hk, cu=spec.cu_str,
                                 L_cp=lcp))
            if key in cache:
                hits += 1
            else:
                want.append(lcp)
        if want:
            todo.append((spec, want))
    return todo, hits


def measure_specs(specs, *, cache_path: str, P: int, chunk: int = CHUNK_SIZE,
                  ratio: float | None = None, device: str | None = None,
                  warmup_ms: float = 25, rep_ms: float = 100, seed: int = 42,
                  verbose: bool = True) -> dict[tuple, dict]:
    """Measure whatever the cache is missing and return the full cache.

    Each row is appended and flushed as soon as it is measured. With tilelang's
    cache warm the 141-row reference calibration takes 4.5 minutes; the first run
    on a fresh cache is dominated by JIT and is an order of magnitude longer, so
    an interrupted sweep has to resume rather than restart.
    """
    from .utils import DEFAULT_LCP_RATIO
    if ratio is None:
        ratio = DEFAULT_LCP_RATIO
    if device is None:
        device = torch.cuda.get_device_name()
    cache = read_cache(cache_path)
    todo, hits = plan_cases(specs, P=P, chunk=chunk, ratio=ratio,
                            device=device, cache=cache)
    n_todo = sum(len(l) for _, l in todo)
    if verbose:
        print(f"cache: {cache_path} ({hits} hit, {n_todo} to measure)")
    done = 0
    for spec, lcps in todo:
        for row in measure_shape(spec, lcps, P=P, chunk=chunk, device=device,
                                 warmup_ms=warmup_ms, rep_ms=rep_ms, seed=seed):
            append_cache(cache_path, row)
            cache[cache_key(row)] = row
            done += 1
            if verbose:
                print(f"  [{done}/{n_todo}] {spec.name:12s} L_cp={row['L_cp']:<5d} "
                      f"N_part={row['N_part']:<5d} all_total={row['all_total']:.4f} ms")
    return cache
