#!/bin/bash
# Profile every CP mode (none / intra / inter / inter_intra) in one run.
#
# Usage:
#   bash scripts/profile_cp.sh                              # 4 cards, strong scaling
#   NGPU=2 bash scripts/profile_cp.sh                       # 2 cards
#   SCALING=both bash scripts/profile_cp.sh                 # both baseline views
#   MODES=inter,inter_intra bash scripts/profile_cp.sh      # a subset
#   FLA=1 bash scripts/profile_cp.sh                        # add the FLA baselines
#   SET=cp bash scripts/profile_cp.sh                       # sweep profile/settings/cp.csv
#   EXTRA="--no-check" bash scripts/profile_cp.sh           # skip the parity step
#
# Every run ends with a parity check of profile/cp_stages.py against the production
# path; a mismatch exits non-zero.
set -euo pipefail

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

NGPU=${NGPU:-4}
SEQLEN=${SEQLEN:-32768}
NVH=${NVH:-32}
NKH=${NKH:-${NVH}}
SCALING=${SCALING:-strong}
WARMUP=${WARMUP:-25}
REP=${REP:-100}
PORT=${PORT:-29500}

ARGS=(
    --seqlen "${SEQLEN}"
    --nvh "${NVH}"
    --nkh "${NKH}"
    --scaling "${SCALING}"
    --warmup "${WARMUP}"
    --rep "${REP}"
)
[ -n "${CU_SEQLENS:-}" ] && ARGS+=(--cu-seqlens "${CU_SEQLENS}")
[ -n "${MODES:-}" ] && ARGS+=(--modes "${MODES}")
[ -n "${SET:-}" ] && ARGS+=(--set "${SET}")
[ "${FLA:-0}" = "1" ] && ARGS+=(--fla)

if [ "${NGPU}" = "1" ]; then
    python profile/profile_cp.py "${ARGS[@]}" ${EXTRA:-}
else
    torchrun --nproc_per_node="${NGPU}" --master_port="${PORT}" \
        profile/profile_cp.py "${ARGS[@]}" ${EXTRA:-}
fi
