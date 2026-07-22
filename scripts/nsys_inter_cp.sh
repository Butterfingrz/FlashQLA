#!/bin/bash
# nsys profile inter-card CP with NVTX annotations.
# Output: nsys_inter_cp.nsys-rep in OUTDIR.
#
# Usage:
#   bash scripts/nsys_inter_cp.sh                      # default
#   CUDA_GRAPH=1 bash scripts/nsys_inter_cp.sh          # with CUDA graph
#   NGPU=2 SEQLEN=16384 bash scripts/nsys_inter_cp.sh   # custom config
set -euo pipefail

export NCCL_DEBUG=WARN

NGPU=${NGPU:-4}
SEQLEN=${SEQLEN:-32768}
NVH=${NVH:-128}
NKH=${NKH:-128}
CU_SEQLENS="${CU_SEQLENS:-0-${SEQLEN}}"
WARMUP=${WARMUP:-5}
REP=${REP:-3}
OUTDIR=${OUTDIR:-.}
CUDA_GRAPH=${CUDA_GRAPH:-0}

EXTRA_FLAGS=""
if [ "${CUDA_GRAPH}" = "1" ]; then
    EXTRA_FLAGS="--cuda-graph"
fi

mkdir -p "${OUTDIR}"

nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --output="${OUTDIR}/nsys_inter_cp" \
    --force-overwrite=true \
    --trace-fork-before-exec=true \
    torchrun --nproc_per_node="${NGPU}" --master_port=29501 \
        profile/profile_inter_cp.py \
        --seqlen "${SEQLEN}" \
        --nvh "${NVH}" \
        --nkh "${NKH}" \
        --cu-seqlens "${CU_SEQLENS}" \
        --warmup "${WARMUP}" \
        --rep "${REP}" \
        --nsys \
        ${EXTRA_FLAGS}

echo ""
echo "Output: ${OUTDIR}/nsys_inter_cp.nsys-rep"
