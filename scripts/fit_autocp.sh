#!/bin/bash
# Calibrate the autocp latency model: shape CSVs in, coefficient CSV out.
# Wraps `python -m tools.autocp.fit`: every run reads/writes the per-device
# measurement cache, so it only times what is missing, then least squares. A
# cold sweep takes tens of minutes; a warm refit seconds.
#
#   bash scripts/fit_autocp.sh                      # fit, measuring what's missing
#   bash scripts/fit_autocp.sh --refresh-cache      # re-time every case first
#   OUT=<shipped coefs> bash scripts/fit_autocp.sh  # ship the result
#
# Env overrides: ARCH TRAIN EVAL CACHE REPORT PLOT OUT. Trailing args go to
# fit.py and win over the defaults below. Omit ARCH to fit for the GPU you're on
# (fit.py probes it; name it only to override). Coefficients are never installed
# implicitly: they land in tmp/coefs_<arch>.csv by default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASES="${ROOT}/tools/autocp/cases"

ARCH=${ARCH:-}                            # empty -> fit.py detects this GPU's arch
TRAIN=${TRAIN:-${CASES}/train.csv}
EVAL=${EVAL-${CASES}/eval.csv}             # `-` not `:-`, so "" really disables
OUT=${OUT:-}                              # empty -> tmp/coefs_<arch>.csv
REPORT=${REPORT-${ROOT}/tmp/autocp_report.csv}
CACHE=${CACHE:-}
PLOT=${PLOT-}

cd "${ROOT}"

ARGS=(--train "${TRAIN}")
if [ -n "${ARCH}" ]; then ARGS+=(--arch "${ARCH}"); fi
if [ -n "${OUT}" ]; then ARGS+=(--out "${OUT}"); fi
if [ -n "${EVAL}" ]; then ARGS+=(--eval "${EVAL}"); fi
if [ -n "${REPORT}" ]; then ARGS+=(--report-csv "${REPORT}"); fi
if [ -n "${CACHE}" ]; then ARGS+=(--cache "${CACHE}"); fi
if [ -n "${PLOT}" ]; then ARGS+=(--plot "${PLOT}"); fi

echo "arch:   ${ARCH:-(detect on this GPU)}"
echo "train:  ${TRAIN}"
echo "eval:   ${EVAL:-(none)}"
echo "out:    ${OUT:-(tmp/coefs_<arch>.csv)}"
echo "cache:  ${CACHE:-(fit.py default: tmp/autocp_cache_<device>.csv)}"
echo "plot:   ${PLOT:-(none)}"
echo ""

python -m tools.autocp.fit "${ARGS[@]}" "$@"
