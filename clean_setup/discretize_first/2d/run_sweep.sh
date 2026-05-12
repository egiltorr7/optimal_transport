#!/usr/bin/env bash
# run_sweep.sh  --  Sweep over eps / grid combos on this machine's GPU.
#
# Usage (run directly on the GPU machine):
#   ./run_sweep.sh
#
# Edit the SWEEP PARAMETERS section below to set the combos you want.
# Jobs run sequentially so only one GPU is used at a time.

set -e

# --- Edit these to match your setup ---
MATLAB_BIN="matlab"
GPU_DEVICE=1          # 1-based GPU index; check with gpuDeviceTable in MATLAB
# --------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG_DIR="${SCRIPT_DIR}/config"
EXP_DIR="${SCRIPT_DIR}/experiments"
RES_DIR="${SCRIPT_DIR}/results"

# =============================================================================
# SWEEP PARAMETERS
# Each row is one job: nt  nx  ny  eps
# Add/remove rows as needed.
# =============================================================================
JOBS=(
#  nt   nx    ny    eps    gamma  tau    proj
   "32   64    64   1e-4   0.1    0.11   proj_fokker_planck_spike2"
   "32   64    64   1e-3   0.1    0.11   proj_fokker_planck_spike2"
   "32   64    64   1e-2   0.1    0.11   proj_fokker_planck_spike2"
   "32   64    64   1e-1   0.1    0.11   proj_fokker_planck_spike2"
   "32   64    64   1      0.1    0.11   proj_fokker_planck_spike2"
   "64   128   128  1e-4   0.1    0.11   proj_fokker_planck_spike2"
   "64   128   128  1e-3   0.1    0.11   proj_fokker_planck_spike2"
   "64   128   128  1e-2   0.1    0.11   proj_fokker_planck_spike2"
   "64   128   128  1e-1   0.1    0.11   proj_fokker_planck_spike2"
   "64   128   128  1      0.1    0.11   proj_fokker_planck_spike2"
   "128  256   256  1e-4   0.1    0.11   proj_fokker_planck_spike2"
   "128  256   256  1e-3   0.1    0.11   proj_fokker_planck_spike2"
   "128  256   256  1e-2   0.1    0.11   proj_fokker_planck_spike2"
   "128  256   256  1e-1   0.1    0.11   proj_fokker_planck_spike2"
   "128  256   256  1      0.1    0.11   proj_fokker_planck_spike2"
   "128  128   128  1e-4   0.1    0.11   proj_fokker_planck_spike2"
   "128  128   128  1e-3   0.1    0.11   proj_fokker_planck_spike2"
   "128  128   128  1e-2   0.1    0.11   proj_fokker_planck_spike2"
   "128  128   128  1e-1   0.1    0.11   proj_fokker_planck_spike2"
   "128  128   128  1      0.1    0.11   proj_fokker_planck_spike2"
   "256  256   256  1e-4   0.1    0.11   proj_fokker_planck_spike2"
   "256  256   256  1e-3   0.1    0.11   proj_fokker_planck_spike2"
   "256  256   256  1e-2   0.1    0.11   proj_fokker_planck_spike2"
   "256  256   256  1e-1   0.1    0.11   proj_fokker_planck_spike2"
   "256  256   256  1      0.1    0.11   proj_fokker_planck_spike2"
)
# =============================================================================

mkdir -p "${RES_DIR}/figures"

# CSV summary log (appended after each job so partial sweeps are recorded)
LOG="${RES_DIR}/sweep_log.csv"
if [ ! -f "${LOG}" ]; then
    echo "job,nt,nx,ny,eps,gamma,tau,status,bash_wall_s,matlab_wall_s,iters,converged,error" > "${LOG}"
fi

TOTAL=${#JOBS[@]}
IDX=0

for JOB in "${JOBS[@]}"; do
    IDX=$((IDX + 1))
    read -r NT NX NY EPS GAMMA TAU PROJ <<< "$JOB"

    echo ""
    echo "==> Job ${IDX}/${TOTAL}: nt=${NT} nx=${NX} ny=${NY} eps=${EPS} proj=${PROJ}"

    # Write cfg_ladmm_gaussian_run.m for this combo
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

    # Run MATLAB, capturing output and timing at the bash level
    T_START=$(date +%s)
    MATLAB_OUT=$("${MATLAB_BIN}" -nodisplay -nosplash -batch "cd('${EXP_DIR}'); test_sb_gaussian;" 2>&1) && STATUS="OK" || STATUS="FAILED"
    T_END=$(date +%s)
    BASH_WALL=$((T_END - T_START))

    echo "${MATLAB_OUT}"

    # Parse key stats from MATLAB stdout
    MATLAB_WALL=$(echo "${MATLAB_OUT}" | grep -oP 'wall=\K[0-9.]+' | head -1 || echo "NA")
    ITERS=$(echo "${MATLAB_OUT}"       | grep -oP 'iters=\K[0-9]+'  | head -1 || echo "NA")
    CONVERGED=$(echo "${MATLAB_OUT}"   | grep -oP 'converged=\K[0-9]' | head -1 || echo "NA")
    ERROR=$(echo "${MATLAB_OUT}"       | grep -oP 'error=\K[0-9.eE+-]+' | head -1 || echo "NA")

    echo "   Job ${IDX}/${TOTAL} ${STATUS} — bash wall: ${BASH_WALL}s  matlab wall: ${MATLAB_WALL}s  iters: ${ITERS}  converged: ${CONVERGED}  gamma: ${GAMMA}"
    echo "${IDX},${NT},${NX},${NY},${EPS},${GAMMA},${TAU},${STATUS},${BASH_WALL},${MATLAB_WALL},${ITERS},${CONVERGED},${ERROR}" >> "${LOG}"
done

echo ""
echo "==> Sweep complete. Results in: ${RES_DIR}"
echo ""
echo "==> Timing summary:"
column -t -s',' "${LOG}"
