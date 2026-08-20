# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Pytest wiring: arch/capability gating and CP test selection.

Capability gating reads the single table in :mod:`cp_arch` instead of spelling arch
names out per test, so a test says *what it needs* (``cp_inter_intra``, ``needs_bwd``)
rather than *where it runs*.

CP selection options::

    pytest tests/test_cp_e2e.py --cp-world-sizes=2,4   # only these world sizes
    pytest tests/test_cp_e2e.py --cp-case=single-tpc2048-Hk8Hv8-g0.0625

Tests opt into the sweep by taking a ``cp_world_size`` argument (see
:func:`pytest_generate_tests`).
"""
import pytest
import torch

from cp_arch import ARCH, GPU_AVAILABLE, supports

requires_gpu = pytest.mark.gpu
requires_hopper = pytest.mark.hopper
requires_blackwell = pytest.mark.blackwell
requires_sm120 = pytest.mark.sm120

#: World sizes the CP tests sweep when no explicit ``--cp-world-sizes`` is given.
#: Anything above the visible device count is dropped. ``1`` is part of the sweep on
#: purpose: it is the single-card path (no process group), and it always runs.
DEFAULT_CP_WORLD_SIZES = (1, 2, 3, 4)


def pytest_addoption(parser):
    group = parser.getgroup("flash_qla CP")
    group.addoption(
        "--cp-world-sizes",
        action="store",
        default=None,
        help=(
            "Comma-separated world sizes for the end-to-end CP tests "
            f"(default: {','.join(map(str, DEFAULT_CP_WORLD_SIZES))} capped at the "
            "visible device count; 1 is the single-card path)."
        ),
    )
    group.addoption(
        "--cp-case",
        action="store",
        default=None,
        help=(
            "Run only the CP case with this id (see cp_common.CPCase.id). Intended for "
            "debugging a single failing configuration."
        ),
    )


def cp_world_sizes(config) -> list[int]:
    """Resolve ``--cp-world-sizes`` against the visible device count."""
    raw = config.getoption("--cp-world-sizes")
    if raw:
        wanted = [int(x) for x in str(raw).replace(" ", "").split(",") if x]
    else:
        wanted = list(DEFAULT_CP_WORLD_SIZES)
    available = torch.cuda.device_count()
    return [w for w in wanted if w <= available]


def cp_case_filter(config) -> str | None:
    return config.getoption("--cp-case")


def pytest_generate_tests(metafunc):
    """Any test taking a ``cp_world_size`` argument gets the resolved sweep.

    The name is deliberately distinct from a plain ``world_size`` parameter so tests
    that want to pin their own world sizes (e.g. the topology unit tests) still can.
    When nothing is runnable the sweep is a single ``None``, which keeps the test
    collected so it reports a skip instead of silently vanishing.

    ``multigpu`` is applied per parameter rather than per test, so ``-m multigpu`` selects
    only the spawning world sizes while ``W1`` -- the single-card path -- still runs on a
    one-GPU box.
    """
    if "cp_world_size" not in metafunc.fixturenames:
        return
    sizes = cp_world_sizes(metafunc.config)
    if not sizes:
        metafunc.parametrize("cp_world_size", [None], ids=["W-none"])
        return
    metafunc.parametrize("cp_world_size", [
        pytest.param(w, id=f"W{w}", marks=[pytest.mark.multigpu] if w > 1 else [])
        for w in sizes
    ])


def pytest_collection_modifyitems(config, items):
    device_count = torch.cuda.device_count()
    for item in items:
        if "gpu" in item.keywords and not GPU_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="CUDA GPU not available"))
        if "hopper" in item.keywords and ARCH != "SM90":
            item.add_marker(pytest.mark.skip(reason="Hopper (SM90) GPU required"))
        if "blackwell" in item.keywords and ARCH not in ["SM100", "SM103"]:
            item.add_marker(pytest.mark.skip(reason="Blackwell (SM100 or SM103) GPU required"))
        if "sm120" in item.keywords and ARCH != "SM120":
            item.add_marker(pytest.mark.skip(reason="Blackwell SM120 GPU required"))
        # Capability-based gating (see cp_arch.ARCH_CAPS).
        if "cp_inter_intra" in item.keywords and not supports("inter_intra"):
            item.add_marker(pytest.mark.skip(
                reason=f"combined inter+intra CP is not implemented on {ARCH}"))
        if "needs_bwd" in item.keywords and not supports("bwd"):
            item.add_marker(pytest.mark.skip(
                reason=f"backward kernels are not implemented on {ARCH}"))
        if "multigpu" in item.keywords and device_count < 2:
            item.add_marker(pytest.mark.skip(
                reason=f"multi-GPU test requires >= 2 GPUs, found {device_count}"))
