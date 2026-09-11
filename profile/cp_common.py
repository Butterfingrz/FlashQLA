# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Thin CP scaffolding for the profiling scripts: the mode registry (the four
``(is_inter, is_intra)`` modes, each knowing its context builder, capability gate and
minimum world size) plus the process-group helpers. The tests roll their own cases inline
(``test_cp_e2e.py``); this drives the config sweeps in ``profile/``. Kept free of pytest
imports so the profile scripts can use it standalone.

The capability table (``arch_utils``) still lives under ``tests/`` -- conftest reads it during
collection -- so add that directory to the path before importing it.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "tests"))

from arch_utils import ARCH, caps, supports  # noqa: F401  (re-exported for convenience)

from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE
from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context

# Distinct rendezvous ports per (mode, world_size) so consecutive spawns in one
# pytest session never collide on a port still in TIME_WAIT.
_BASE_PORT = 29540


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CPMode:
    name: str
    is_inter: bool
    is_intra: bool
    #: ``arch_utils`` capability key that must be true for this mode to run at all.
    capability: str | None
    index: int

    @property
    def needs_dist(self) -> bool:
        return self.is_inter

    @property
    def min_world_size(self) -> int:
        return 2 if self.is_inter else 1

    def port(self, world_size: int) -> str:
        return str(_BASE_PORT + self.index * 8 + world_size)

    def supported(self) -> bool:
        return self.capability is None or supports(self.capability)

    def make_ctx(
        self,
        cu_seqlens: torch.Tensor,
        *,
        num_v_heads: int,
        group=None,
        force_intra_cp: bool = False,
        is_train: bool = False,
    ):
        """Build this mode's context over the **global** varlen offsets; the inter
        split derives each card's local view from it."""
        if self.is_inter:
            assert group is not None, f"mode {self.name} needs a process group"
        return build_cp_context(
            cu_seqlens,
            enable_inter=self.is_inter,
            enable_intra=self.is_intra,
            group=group,
            num_v_heads=num_v_heads,
            chunk_size=CHUNK_SIZE,
            is_train=is_train,
            force_intra_cp=force_intra_cp,
        )


CP_MODES: dict[str, CPMode] = {
    m.name: m
    for m in [
        CPMode("none", is_inter=False, is_intra=False, capability=None, index=0),
        CPMode("intra", is_inter=False, is_intra=True, capability="intra", index=1),
        CPMode("inter", is_inter=True, is_intra=False, capability="inter", index=2),
        CPMode("inter_intra", is_inter=True, is_intra=True, capability="inter_intra", index=3),
    ]
}

ALL_MODES = list(CP_MODES)


def modes_for(world_size: int, *, supported_only: bool = True) -> list[str]:
    """Mode names runnable at ``world_size`` on this arch."""
    out = []
    for name, mode in CP_MODES.items():
        if world_size < mode.min_world_size:
            continue
        if supported_only and not mode.supported():
            continue
        out.append(name)
    return out


def skip_reason(mode_name: str, world_size: int, *, need_bwd: bool = True) -> str | None:
    """Human-readable reason this (mode, world_size) cannot run here, or ``None``."""
    mode = CP_MODES[mode_name]
    if not mode.supported():
        return f"{mode_name} CP is not supported on {ARCH}"
    if need_bwd and not supports("bwd"):
        return f"backward kernels are not implemented on {ARCH}"
    if world_size < mode.min_world_size:
        return f"{mode_name} CP needs world_size >= {mode.min_world_size}"
    if torch.cuda.device_count() < world_size:
        return f"needs >= {world_size} GPUs, found {torch.cuda.device_count()}"
    return None


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def init_distributed(rank: int, world_size: int, port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return dist.group.WORLD


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
