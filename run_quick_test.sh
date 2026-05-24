#!/bin/bash
# Quick sanity test — small grid, fast run
# Use this FIRST to verify everything compiles and works
set -e

EXEC=./cardiac_sim
mkdir -p results
rm -f results/benchmark.csv results/*.bin
echo "strategy,grid_size,nprocs,steps,time_max,time_min,time_avg" > results/benchmark.csv

echo "Quick test: 512x512 grid, 1000 steps..."
echo ""

echo ">>> Sequential..."
mpirun --oversubscribe -np 1 $EXEC 1 512 1000 500

echo -e "\n>>> Row-block 4P..."
mpirun --oversubscribe -np 4 $EXEC 1 512 1000 500

echo -e "\n>>> Column-block 4P..."
mpirun --oversubscribe -np 4 $EXEC 2 512 1000 500

echo -e "\n>>> 2D-block 4P..."
mpirun --oversubscribe -np 4 $EXEC 3 512 1000 500

echo ""
echo "Quick test passed! Run: python3 scripts/visualize.py"
echo "Then run ./run_benchmark.sh for full results."
