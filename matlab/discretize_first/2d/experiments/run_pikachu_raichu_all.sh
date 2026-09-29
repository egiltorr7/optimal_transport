#!/bin/bash
# Pikachu -> Raichu transport: both vareps values x both nt values, on the GPU.
#
#   bash run_pikachu_raichu_all.sh [matlab_binary]
#
# 4 runs: vareps in {0, 1e-2} x nt in {64, 128}, all at nx = ny = 128.
#
# ONE MATLAB PROCESS PER RUN, on purpose: MATLAB does not release GPU memory
# between solves inside a process, and a fresh process also avoids allocator
# fragmentation across back-to-back runs. `|| true` keeps one failure from
# abandoning the rest.
#
# Ordered nt=128 FIRST within each vareps: it is the bigger allocation, and
# running the largest case in a fresh process is the standard way to avoid a
# fragmentation-induced OOM that a smaller-first order can cause.
set -u
MATLAB=${1:-matlab}
HERE=$(cd "$(dirname "$0")" && pwd)
D2D=$(cd "$HERE/.." && pwd)

for EPS in 0 1e-2; do
  for NT in 128 64; do
    echo "############ vareps = $EPS   nt = $NT ############"
    "$MATLAB" -batch "cd('$D2D'); setup_paths; run_pikachu_raichu($EPS, $NT);" || true
  done
done

echo "############ done ############"
ls -la "$D2D/results/pikachu"/*.mat 2>/dev/null || echo "(no output found)"
