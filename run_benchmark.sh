#!/bin/bash
# ==============================================
# Cardiac Electrophysiology — Full Benchmark Suite
# Target: AMD Ryzen 9 AI 365 (10C/20T)
# Grid: 2048x2048 | Steps: 5000 | ~30-60 min total
# ==============================================
set -e

GRID=2048
STEPS=5000
SNAP=2500
EXEC=./cardiac_sim

echo "=============================================="
echo "  Cardiac Electrophysiology Benchmark"
echo "  Fenton-Karma 3-Variable Ionic Model"
echo "  Grid: ${GRID}x${GRID} = $((GRID * GRID)) cells"
echo "  Steps: ${STEPS} (500 ms simulated)"
echo "  CPU: AMD Ryzen 9 AI 365 (10C/20T)"
echo "=============================================="
echo ""
echo "Total: 13 runs. Estimated ~30-60 min."
echo "Starting in 5s... (Ctrl+C to abort)"
sleep 5

mkdir -p results
rm -f results/benchmark.csv results/*.bin
echo "strategy,grid_size,nprocs,steps,time_max,time_min,time_avg" > results/benchmark.csv

echo -e "\n>>> [1/13] Sequential baseline..."
mpirun --oversubscribe -np 1 $EXEC 1 $GRID $STEPS $SNAP

echo -e "\n>>> [2/13] Row-block | 2P..."
mpirun --oversubscribe -np 2 $EXEC 1 $GRID $STEPS $SNAP

echo -e "\n>>> [3/13] Row-block | 4P..."
mpirun --oversubscribe -np 4 $EXEC 1 $GRID $STEPS $SNAP

echo -e "\n>>> [4/13] Row-block | 8P..."
mpirun --oversubscribe -np 8 $EXEC 1 $GRID $STEPS $SNAP

echo -e "\n>>> [5/13] Row-block | 16P..."
mpirun --oversubscribe -np 16 $EXEC 1 $GRID $STEPS $SNAP

echo -e "\n>>> [6/13] Column-block | 2P..."
mpirun --oversubscribe -np 2 $EXEC 2 $GRID $STEPS $SNAP

echo -e "\n>>> [7/13] Column-block | 4P..."
mpirun --oversubscribe -np 4 $EXEC 2 $GRID $STEPS $SNAP

echo -e "\n>>> [8/13] Column-block | 8P..."
mpirun --oversubscribe -np 8 $EXEC 2 $GRID $STEPS $SNAP

echo -e "\n>>> [9/13] Column-block | 16P..."
mpirun --oversubscribe -np 16 $EXEC 2 $GRID $STEPS $SNAP

echo -e "\n>>> [10/13] 2D-block | 2P..."
mpirun --oversubscribe -np 2 $EXEC 3 $GRID $STEPS $SNAP

echo -e "\n>>> [11/13] 2D-block | 4P..."
mpirun --oversubscribe -np 4 $EXEC 3 $GRID $STEPS $SNAP

echo -e "\n>>> [12/13] 2D-block | 8P..."
mpirun --oversubscribe -np 8 $EXEC 3 $GRID $STEPS $SNAP

echo -e "\n>>> [13/13] 2D-block | 16P..."
mpirun --oversubscribe -np 16 $EXEC 3 $GRID $STEPS $SNAP

echo ""
echo "=============================================="
echo "  ALL 13 RUNS COMPLETE!"
echo "  Results:  results/benchmark.csv"
echo "  Next:     python3 scripts/visualize.py"
echo "=============================================="
