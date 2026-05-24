#!/bin/bash
# ==============================================
# Correctness Validation: Serial vs Parallel
# Compares final-state snapshots across all
# decomposition strategies against the serial
# reference to verify numerical equivalence.
# ==============================================
set -e

GRID=256
STEPS=500
SNAP=0           # no intermediate snapshots (only final)
EXEC=./cardiac_sim

echo "=============================================="
echo "  Correctness Validation"
echo "  Grid: ${GRID}x${GRID} | Steps: ${STEPS}"
echo "=============================================="
echo ""

mkdir -p results
rm -f results/snapshot_*.bin

# Reference: serial run
echo ">>> Serial reference (1P, row-block)..."
mpirun --oversubscribe -np 1 $EXEC 1 $GRID $STEPS $SNAP
cp results/snapshot_$(printf "%06d" $STEPS).bin results/ref_serial.bin

# Row-block 4P
echo -e "\n>>> Row-block 4P..."
rm -f results/snapshot_*.bin
mpirun --oversubscribe -np 4 $EXEC 1 $GRID $STEPS $SNAP
cp results/snapshot_$(printf "%06d" $STEPS).bin results/test_row4.bin

# Column-block 4P
echo -e "\n>>> Column-block 4P..."
rm -f results/snapshot_*.bin
mpirun --oversubscribe -np 4 $EXEC 2 $GRID $STEPS $SNAP
cp results/snapshot_$(printf "%06d" $STEPS).bin results/test_col4.bin

# 2D-block 4P
echo -e "\n>>> 2D-block 4P..."
rm -f results/snapshot_*.bin
mpirun --oversubscribe -np 4 $EXEC 3 $GRID $STEPS $SNAP
cp results/snapshot_$(printf "%06d" $STEPS).bin results/test_2d4.bin

# Compare
echo ""
echo "=============================================="
echo "  Comparing against serial reference..."
echo "=============================================="
python3 scripts/validate.py \
    results/ref_serial.bin \
    results/test_row4.bin \
    results/test_col4.bin \
    results/test_2d4.bin

# Cleanup
rm -f results/ref_serial.bin results/test_row4.bin results/test_col4.bin results/test_2d4.bin
