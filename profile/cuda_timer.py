# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextlib import contextmanager

import torch


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
