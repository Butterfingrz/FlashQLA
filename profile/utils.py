# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Profiling helpers for the ``profile/`` scripts.

* ``profile``    -- torch.profiler run with a per-kernel time breakdown.
* ``CudaTimer``  -- CUDA-event timer that averages user-marked intervals over reps.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextlib import contextmanager

import torch
import tilelang


def profile(func, inputs, wait: int = 50, warmup: int = 50, rep: int = 100):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=rep),
    ) as prof:
        for idx in range(wait + warmup + rep):
            func(*inputs)
            prof.step()

    cuda_events = [
        evt for evt in prof.events()
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.device_time > 0
    ]

    kernels_per_iter = None
    for n_kernels in range(1, len(cuda_events) + 1):
        if len(cuda_events) % n_kernels == 0:
            chunk = [e.name for e in cuda_events[:n_kernels]]
            ok = True
            for i in range(n_kernels, len(cuda_events), n_kernels):
                if [e.name for e in cuda_events[i:i + n_kernels]] != chunk:
                    ok = False
                    break
            if ok:
                kernels_per_iter = n_kernels
                break

    if kernels_per_iter is None:
        kernels_per_iter = len(cuda_events) // rep if rep > 0 else len(cuda_events)

    if kernels_per_iter == 0:
        result = {}
        result["total"] = tilelang.profiler.do_bench(
            lambda: func(*inputs), warmup=warmup, rep=rep
        )
        return result

    num_iters = len(cuda_events) // kernels_per_iter
    sums = {}
    order = []
    for i in range(kernels_per_iter):
        name = cuda_events[i].name
        count = sum(1 for j in range(i) if cuda_events[j].name == name)
        key = f"{name}#{count}" if count > 0 else name
        order.append(key)
        sums[key] = 0.0

    for it in range(num_iters):
        base = it * kernels_per_iter
        for i, key in enumerate(order):
            sums[key] += cuda_events[base + i].device_time * 1e-3

    result = {k: sums[k] / num_iters for k in order}
    result["total"] = tilelang.profiler.do_bench(
        lambda: func(*inputs), warmup=warmup, rep=rep
    )
    return result


class CudaTimer:
    """CUDA event profiler.

    ``mark()`` records cuda event pairs inside a function;
    ``bench()`` runs that function warmup+rep times and averages the marks.

    Usage::

        timer = CudaTimer()

        def my_pipeline(timer, x):
            with timer.mark("step_a"):
                y = step_a(x)
            with timer.mark("step_b"):
                z = step_b(y)

        timer.bench(my_pipeline, timer, x)
        timer.print_report()
    """

    def __init__(self, warmup: int = 25, rep: int = 100):
        self.warmup = warmup
        self.rep = rep
        self._recording = False
        self._pending: dict[str, list[tuple]] = defaultdict(list)
        self._results: OrderedDict[str, float] = OrderedDict()

    @contextmanager
    def mark(self, tag: str):
        """Record a CUDA-event interval. Only accumulates when inside ``bench()``."""
        if not self._recording:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self._pending[tag].append((start, end))

    def bench(self, fn, *args, **kwargs):
        """Run *fn* warmup+rep times; average all ``mark()`` intervals inside."""
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        self._recording = True
        self._pending.clear()
        for _ in range(self.rep):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        self._recording = False
        for tag, pairs in self._pending.items():
            self._results[tag] = sum(s.elapsed_time(e) for s, e in pairs) / len(pairs)
        self._pending.clear()

    def report(self) -> OrderedDict[str, float]:
        return OrderedDict(self._results)

    def reset(self):
        self._results.clear()

    def print_report(self, title: str = ""):
        import pandas as pd
        header = title or "ms"
        print(pd.DataFrame({header: self._results}, dtype=float).round(4))
