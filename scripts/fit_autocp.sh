#!/bin/bash
# Calibrate the autocp latency model: shape CSVs in, coefficient CSV out.
#
# Wraps `python -m tools.autocp.fit`, which measures the four kernels itself
# (cached, so an interrupted sweep resumes) and then solves one least squares
# per kernel. A full sweep is ~145 measured cases -- tens of minutes, plus
# tilelang JIT on the first run.
#
# Usage:
#   bash scripts/fit_autocp.sh                       # measure (resuming) + fit
#   bash scripts/fit_autocp.sh --from-cache          # refit from the cache, no GPU
#   CACHE=<path>/autocp_cache_nvidia_gb200.csv bash scripts/fit_autocp.sh --from-cache
#   bash scripts/fit_autocp.sh --measure-only        # only fill the cache
#   bash scripts/fit_autocp.sh --lcp-ratio 3.0       # coarser L_cp grid (quick check)
#   PLOT=debug/autocp_fit bash scripts/fit_autocp.sh # also draw the fitted curves
#   OUT=<shipped coefs> bash scripts/fit_autocp.sh   # ship the result (see below)
#
# PLOT is off by default because it wants matplotlib, which flash_qla does not
# depend on (nothing on the launch path plots).
#
# Anything after the script name is forwarded to fit.py and wins over the
# defaults here, so `--cache`, `--device`, `--sm-count`, `--warmup-ms` etc. all
# work unchanged.
#
# By default the coefficients land in debug/ and are NOT installed. Compare them
# against the shipped ones first, then re-run with OUT pointing at the shipped
# path -- absolute kernel times on GB200 drift 15-25% between sessions, so a
# recalibration should be a deliberate act, not a side effect.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Calibration input lives outside the package (it is not shipped); only the
# fitted coefficients land inside it.
CASES="${ROOT}/tools/autocp/cases"
ARCH=${ARCH:-sm100}
SHIPPED="${ROOT}/flash_qla/ops/gated_delta_rule/chunk/cp/autocp/coefs/${ARCH}.csv"

TRAIN=${TRAIN:-${CASES}/train.csv}
EVAL=${EVAL-${CASES}/eval.csv}             # empty string disables the held-out set
OUT=${OUT:-${ROOT}/debug/coefs_${ARCH}.csv}
REPORT=${REPORT-${ROOT}/debug/autocp_report.csv}
CACHE=${CACHE:-}
PLOT=${PLOT-}

cd "${ROOT}"
mkdir -p "$(dirname "${OUT}")"

# --from-cache needs to know which cache; fit.py's default path is derived from
# the live GPU's name, which that path deliberately never queries. Auto-pick when
# debug/ holds exactly one. Not done for the measuring path: there the device is
# known, and reusing another GPU's cache would silently blend two machines.
if [ -z "${CACHE}" ] && [[ " $* " == *" --from-cache "* ]]; then
    shopt -s nullglob
    found=(debug/autocp_cache_*.csv)
    if [ ${#found[@]} -eq 1 ]; then
        CACHE="${found[0]}"
    elif [ ${#found[@]} -gt 1 ]; then
        echo "several caches in debug/; pick one with CACHE=<path>:" >&2
        printf '  %s\n' "${found[@]}" >&2
        exit 2
    fi
fi

ARGS=(--train "${TRAIN}" --out "${OUT}" --arch "${ARCH}")
if [ -n "${EVAL}" ]; then ARGS+=(--eval "${EVAL}"); fi
if [ -n "${REPORT}" ]; then ARGS+=(--report-csv "${REPORT}"); fi
if [ -n "${CACHE}" ]; then ARGS+=(--cache "${CACHE}"); fi
if [ -n "${PLOT}" ]; then ARGS+=(--plot "${PLOT}"); fi

echo "arch:   ${ARCH}"
echo "train:  ${TRAIN}"
echo "eval:   ${EVAL:-(none)}"
echo "out:    ${OUT}"
echo "cache:  ${CACHE:-(fit.py default: debug/autocp_cache_<device>.csv)}"
echo "plot:   ${PLOT:-(none)}"
echo ""

python -m tools.autocp.fit "${ARGS[@]}" "$@"

if [ -f "${OUT}" ] && [ "$(readlink -f "${OUT}")" != "$(readlink -f "${SHIPPED}")" ]; then
    echo ""
    echo "not installed. compare, then ship:"
    echo "  diff ${OUT} ${SHIPPED}"
    echo "  cp   ${OUT} ${SHIPPED}"
fi
