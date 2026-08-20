#!/bin/bash
# nsys-trace the CP modes with NVTX annotations.
# Output: ${OUTDIR}/nsys_cp.nsys-rep
#
# Usage:
#   bash scripts/nsys_cp.sh                                 # default: all modes, 4 cards
#   MODES=inter_intra bash scripts/nsys_cp.sh               # one mode only (cleaner trace)
#   CUDA_GRAPH=1 bash scripts/nsys_cp.sh                    # capture+replay each pass
#   NGPU=2 SEQLEN=16384 bash scripts/nsys_cp.sh             # custom config
set -euo pipefail

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

NGPU=${NGPU:-4}
SEQLEN=${SEQLEN:-32768}
NVH=${NVH:-32}
NKH=${NKH:-${NVH}}
CU_SEQLENS="${CU_SEQLENS:-0-${SEQLEN}}"
WARMUP=${WARMUP:-5}
REP=${REP:-3}
OUTDIR=${OUTDIR:-.}
CUDA_GRAPH=${CUDA_GRAPH:-0}
PORT=${PORT:-29501}

ARGS=(
    --nsys
    --seqlen "${SEQLEN}"
    --nvh "${NVH}"
    --nkh "${NKH}"
    --cu-seqlens "${CU_SEQLENS}"
    --warmup "${WARMUP}"
    --rep "${REP}"
)
[ -n "${MODES:-}" ] && ARGS+=(--modes "${MODES}")
[ "${CUDA_GRAPH}" = "1" ] && ARGS+=(--cuda-graph)

mkdir -p "${OUTDIR}"

if [ "${NGPU}" = "1" ]; then
    LAUNCH=(python profile/profile_cp.py)
else
    LAUNCH=(torchrun --nproc_per_node="${NGPU}" --master_port="${PORT}" profile/profile_cp.py)
fi

nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --output="${OUTDIR}/nsys_cp" \
    --force-overwrite=true \
    --trace-fork-before-exec=true \
    "${LAUNCH[@]}" "${ARGS[@]}"

echo ""
echo "Output: ${OUTDIR}/nsys_cp.nsys-rep"
