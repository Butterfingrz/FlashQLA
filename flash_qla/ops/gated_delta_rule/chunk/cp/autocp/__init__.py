# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Automatic intra-card CP configuration -- the launch-time half.

* :mod:`.decision` -- the online decision ``cp/context.py`` calls: pick ``L_cp``
  and decide CP on/off from a latency model.
* :mod:`.utils` -- the structural features and the model form it prices with.
* ``coefs/`` -- the fitted coefficients, one CSV per calibrated arch.

Standard library only, both modules: this sits on the launch path and must not
pull in numpy / pandas / torch.

Calibration -- measuring, fitting, the shape sets -- lives in ``tools/autocp/``,
outside the package, because none of it runs at launch time and all of it wants
numpy / pandas / torch. It imports :mod:`.utils` for the model form rather than
restating it, so the fit and the decision cannot drift apart.

This package has no test module of its own: the one that covered the fit and
``decision`` was retired from the live test set. ``docs/autocp_decision.md`` is
the derivation and says where it went.
"""

from .decision import decide
from .utils import (
    DEFAULT_COEFS_PATH,
    MIN_LCP,
    load_coefs,
)

__all__ = [
    "DEFAULT_COEFS_PATH",
    "MIN_LCP",
    "decide",
    "load_coefs",
]
