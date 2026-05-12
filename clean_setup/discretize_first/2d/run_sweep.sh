#!/usr/bin/env bash
# run_sweep.sh  --  Sweep over grid/eps/gamma combos on this machine's GPU.
#
# Usage (run directly on the GPU machine):
#   ./run_sweep.sh
#
# Edit GRIDS and STEPS below.  All combinations are run as a cross-product.
# Jobs run sequentially so only one GPU is used at a time.

set -e

# --- Edit these to match your setup ---
MATLAB_BIN="matlab"
GPU_DEVICE=1          # 1-based GPU index; check with gpuDeviceTable in MATLAB
PROJ="proj_fokker_planck_spike2"
# --------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG_DIR="${SCRIPT_DIR}/config"
EXP_DIR="${SCRIPT_DIR}/experiments"
RES_DIR="${SCRIPT_DIR}/results"

# =============================================================================
# SWEEP PARAMETERS
#
# GRIDS: one entry per line  ->  nt  nx  ny  eps
# STEPS: one entry per line  ->  gamma  c      (tau = c * gamma)
#                                               convergence needs tau > gamma * ||A||^2 <= gamma
#                                               so c > 1 is the safe range
#
# All combinations of GRIDS x STEPS are run.
# =============================================================================
GRIDS=(
    "32   64    64   1e-4"
    "32   64    64   1e-2"
    "32   64    64   1e-1"
    "32   64    64   1"
    "64   128   128  1e-4"
    "64   128   128  1e-2"
    "64   128   128  1e-1"
    "64   128   128  1"
    "128  256   256  1e-4"
    "128  256   256  1e-2"
    "128  256   256  1e-1"
    "128  256   256  1"
)

STEPS=(
#  gamma   c
   "0.01   1.1"
   "0.05   1.1"
   "0.1    1.1"
   "0.5    1.1"
   "1.0    1.1"
)
# =============================================================================

mkdir -p "${RES_DIR}/figures"

LOG="${RES_DIR}/sweep_log.csv"
if [ ! -f "${LOG}" ]; then
    echo "job,nt,nx,ny,eps,gamma,tau,status,bash_wall_s,matlab_wall_s,iters,converged,error" > "${LOG}"
fi

TOTAL=$(( ${#GRIDS[@]} * ${#STEPS[@]} ))
IDX=0

for GRID in "${GRIDS[@]}"; do
    read -r NT NX NY EPS <<< "$GRID"

    for STEP in "${STEPS[@]}"; do
        read -r GAMMA C <<< "$STEP"
        TAU=$(awk "BEGIN {printf \"%.6g\", ${GAMMA} * ${C}}")

        IDX=$((IDX + 1))
        echo ""
        echo "==> Job ${IDX}/${TOTAL}: nt=${NT} nx=${NX} ny=${NY} eps=${EPS} gamma=${GAMMA} tau=${TAU}"

        cat > "${CFG_DIR}/cfg_ladmm_gaussian_run.m" << MEOF
function cfg = cfg_ladmm_gaussian_run()
    cfg            = cfg_ladmm_gaussian();
    cfg.nt         = ${NT};
    cfg.nx         = ${NX};
    cfg.ny         = ${NY};
    cfg.vareps     = ${EPS};
    cfg.gamma      = ${GAMMA};
    cfg.tau        = ${TAU};
    cfg.projection = @${PROJ};
    cfg.use_gpu    = true;
    cfg.gpu_device = ${GPU_DEVICE};
end
MEOF

        T_START=$(date +%s)
        MATLAB_OUT=$("${MATLAB_BIN}" -nodisplay -nosplash -batch \
            "cd('${EXP_DIR}'); test_sb_gaussian;" 2>&1) \
            && STATUS="OK" || STATUS="FAILED"
        T_END=$(date +%s)
        BASH_WALL=$((T_END - T_START))

        echo "${MATLAB_OUT}"

        MATLAB_WALL=$(echo "${MATLAB_OUT}" | grep -oP 'wall=\K[0-9.]+'     | head -1 || echo "NA")
        ITERS=$(echo "${MATLAB_OUT}"       | grep -oP 'iters=\K[0-9]+'      | head -1 || echo "NA")
        CONVERGED=$(echo "${MATLAB_OUT}"   | grep -oP 'converged=\K[0-9]'   | head -1 || echo "NA")
        ERROR=$(echo "${MATLAB_OUT}"       | grep -oP 'error=\K[0-9.eE+-]+' | head -1 || echo "NA")

        echo "   Job ${IDX}/${TOTAL} ${STATUS} — bash wall: ${BASH_WALL}s  matlab: ${MATLAB_WALL}s  iters: ${ITERS}  converged: ${CONVERGED}  gamma: ${GAMMA}  tau: ${TAU}"
        echo "${IDX},${NT},${NX},${NY},${EPS},${GAMMA},${TAU},${STATUS},${BASH_WALL},${MATLAB_WALL},${ITERS},${CONVERGED},${ERROR}" >> "${LOG}"
    done
done

echo ""
echo "==> Sweep complete. Results in: ${RES_DIR}"
echo ""
echo "==> Timing summary:"
column -t -s',' "${LOG}"
