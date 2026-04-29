#!/usr/bin/env bash
# run_sweep.sh  --  Sweep over eps / grid combos on a single remote GPU.
#
# Usage (from LOCAL machine):
#   ./run_sweep.sh <user@remote>
#
# Edit the SWEEP PARAMETERS section below to set the combos you want.
# Jobs run sequentially so only one GPU is used at a time.
# Results are pulled locally after each run (so partial results survive failures).

set -e

REMOTE=${1:?"Usage: $0 user@host"}

# --- Edit these to match your setup ---
REMOTE_REPO_DIR="~/git_repos/optimal_transport/clean_setup"
MATLAB_BIN="matlab"
LOCAL_RESULTS_DIR="$(dirname "$0")/results"
GPU_DEVICE=1          # 1-based GPU index on the remote machine
# --------------------------------------

REMOTE_CFG_DIR="${REMOTE_REPO_DIR}/discretize_first/2d/config"
REMOTE_EXP_DIR="${REMOTE_REPO_DIR}/discretize_first/2d/experiments"
REMOTE_RES_DIR="${REMOTE_REPO_DIR}/discretize_first/2d/results"

# =============================================================================
# SWEEP PARAMETERS
# Each row is one job: nt  nx  ny  eps
# Add/remove rows as needed.
# =============================================================================
JOBS=(
#  nt   nx    ny    eps
   "32   64    64   1e-4"
   "32   64    64   1e-3"
   "32   64    64   1e-2"
   "32   64    64   1e-1"
   "32   64    64   1"
   "64   128   128  1e-4"
   "64   128   128  1e-3"
   "64   128   128  1e-2"
   "64   128   128  1e-1"
   "64   128   128  1"
   "128  256   256  1e-4"
   "128  256   256  1e-3"
   "128  256   256  1e-2"
   "128  256   256  1e-1"
   "128  256   256  1"
   "128  128   128  1e-4"
   "128  128   128  1e-3"
   "128  128   128  1e-2"
   "128  128   128  1e-1"
   "128  128   128  1"
   "256  256   256  1e-4"
   "256  256   256  1e-3"
   "256  256   256  1e-2"
   "256  256   256  1e-1"
   "256  256   256  1"
)
# =============================================================================

# Sync code once (exclude results so we don't push stale outputs back)
echo "==> Syncing code to ${REMOTE}:${REMOTE_REPO_DIR} ..."
rsync -az --exclude='results/' --exclude='*.asv' --exclude='cfg_ladmm_gaussian_run.m' \
    "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/" \
    "${REMOTE}:${REMOTE_REPO_DIR}/"

mkdir -p "${LOCAL_RESULTS_DIR}/figures"

# CSV summary log (appended after each job)
LOG="${LOCAL_RESULTS_DIR}/sweep_log.csv"
if [ ! -f "${LOG}" ]; then
    echo "job,nt,nx,ny,eps,status,bash_wall_s,matlab_wall_s,iters,converged,error" > "${LOG}"
fi

TOTAL=${#JOBS[@]}
IDX=0

for JOB in "${JOBS[@]}"; do
    IDX=$((IDX + 1))
    read -r NT NX NY EPS <<< "$JOB"

    echo ""
    echo "==> Job ${IDX}/${TOTAL}: nt=${NT} nx=${NX} ny=${NY} eps=${EPS}"

    # Write a fresh cfg_ladmm_gaussian_run.m on the remote for this combo
    ssh "${REMOTE}" bash -l -c "cat > ${REMOTE_CFG_DIR}/cfg_ladmm_gaussian_run.m << 'MEOF'
function cfg = cfg_ladmm_gaussian_run()
    cfg         = cfg_ladmm_gaussian();
    cfg.nt      = ${NT};
    cfg.nx      = ${NX};
    cfg.ny      = ${NY};
    cfg.vareps  = ${EPS};
    cfg.use_gpu    = true;
    cfg.gpu_device = ${GPU_DEVICE};
end
MEOF"

    # Run MATLAB, timing at the bash level (includes startup overhead)
    T_START=$(date +%s)
    MATLAB_OUT=$(ssh "${REMOTE}" bash -l -c "
        cd ${REMOTE_EXP_DIR} && \
        ${MATLAB_BIN} -nodisplay -nosplash -batch \"
            cd('${REMOTE_EXP_DIR}');
            test_sb_gaussian;
            exit;
        \"
    " 2>&1) && STATUS="OK" || STATUS="FAILED"
    T_END=$(date +%s)
    BASH_WALL=$((T_END - T_START))

    echo "${MATLAB_OUT}"

    # Parse timing/convergence from MATLAB stdout
    MATLAB_WALL=$(echo "${MATLAB_OUT}" | grep -oP 'wall=\K[0-9.]+' | head -1 || echo "NA")
    ITERS=$(echo "${MATLAB_OUT}"       | grep -oP 'iters=\K[0-9]+'  | head -1 || echo "NA")
    CONVERGED=$(echo "${MATLAB_OUT}"   | grep -oP 'converged=\K[0-9]' | head -1 || echo "NA")
    ERROR=$(echo "${MATLAB_OUT}"       | grep -oP 'error=\K[0-9.eE+-]+' | head -1 || echo "NA")

    echo "   Job ${IDX}/${TOTAL} ${STATUS} — bash wall: ${BASH_WALL}s  matlab wall: ${MATLAB_WALL}s  iters: ${ITERS}  converged: ${CONVERGED}"
    echo "${IDX},${NT},${NX},${NY},${EPS},${STATUS},${BASH_WALL},${MATLAB_WALL},${ITERS},${CONVERGED},${ERROR}" >> "${LOG}"

    # Pull results after each run so partial outputs survive failures
    rsync -az "${REMOTE}:${REMOTE_RES_DIR}/" "${LOCAL_RESULTS_DIR}/"
    echo "   Results synced locally."
done

echo ""
echo "==> Sweep complete. Results in: ${LOCAL_RESULTS_DIR}"
echo ""
echo "==> Timing summary:"
column -t -s',' "${LOG}"
