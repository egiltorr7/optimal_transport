#!/usr/bin/env bash
# run_remote.sh  --  Run a 2D experiment on a remote GPU machine.
#
# Usage (from LOCAL machine):
#   ./run_remote.sh <user@remote> [experiment]
#
# Examples:
#   ./run_remote.sh egil@gpu-server.university.edu
#   ./run_remote.sh egil@gpu-server.university.edu test_sb_gaussian
#
# After the job finishes the script pulls figures and results back locally.
#
# Requirements on the remote machine:
#   - MATLAB on PATH  (or set MATLAB_BIN below)
#   - The repo checked out at REMOTE_REPO_DIR

set -e

REMOTE=${1:?"Usage: $0 user@host [experiment]"}
EXPERIMENT=${2:-test_sb_gaussian}

# --- Edit these to match your setup ---
REMOTE_REPO_DIR="~/git_repos/optimal_transport/clean_setup"
MATLAB_BIN="matlab"          # or full path, e.g. /usr/local/MATLAB/R2024a/bin/matlab
LOCAL_RESULTS_DIR="$(dirname "$0")/results"
# --------------------------------------

REMOTE_EXP_DIR="${REMOTE_REPO_DIR}/discretize_first/2d/experiments"
REMOTE_RES_DIR="${REMOTE_REPO_DIR}/discretize_first/2d/results"

echo "==> Syncing code to ${REMOTE}:${REMOTE_REPO_DIR} ..."
rsync -az --exclude='results/' --exclude='*.asv' \
    "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/" \
    "${REMOTE}:${REMOTE_REPO_DIR}/"

echo "==> Launching MATLAB on ${REMOTE} (experiment: ${EXPERIMENT}) ..."
ssh "${REMOTE}" bash -l -c "
    cd ${REMOTE_EXP_DIR} && \
    ${MATLAB_BIN} -nodisplay -nosplash -batch \"
        cd('${REMOTE_EXP_DIR}');
        ${EXPERIMENT};
        exit;
    \"
"

echo "==> Pulling results and figures back to local ..."
mkdir -p "${LOCAL_RESULTS_DIR}/figures"
rsync -az "${REMOTE}:${REMOTE_RES_DIR}/" "${LOCAL_RESULTS_DIR}/"

echo "==> Done. Results in: ${LOCAL_RESULTS_DIR}"
ls "${LOCAL_RESULTS_DIR}/figures/"
