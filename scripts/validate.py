#!/usr/bin/env python3
"""
Correctness Validation — compare parallel snapshots against serial reference.
Reports max absolute error, L2 norm error, and pass/fail per decomposition.
"""

import sys
import os
import numpy as np

TOLERANCE = 1e-10  # floating-point accumulation tolerance


def load_snapshot(path):
    with open(path, "rb") as f:
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float64, count=N * N).reshape(N, N)
    return N, data


def compare(ref_path, test_path, label):
    N_ref, ref = load_snapshot(ref_path)
    N_test, test = load_snapshot(test_path)

    if N_ref != N_test:
        print(f"  {label:20s}  FAIL  grid size mismatch ({N_ref} vs {N_test})")
        return False

    diff = np.abs(ref - test)
    max_err = np.max(diff)
    l2_err = np.linalg.norm(diff)
    checksum_ref = np.sum(ref)
    checksum_test = np.sum(test)

    passed = max_err < TOLERANCE
    status = "PASS" if passed else "FAIL"

    print(f"  {label:20s}  {status}  "
          f"max_err={max_err:.2e}  L2_err={l2_err:.2e}  "
          f"checksum_ref={checksum_ref:.6f}  checksum_test={checksum_test:.6f}")

    return passed


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <reference.bin> <test1.bin> [test2.bin ...]")
        sys.exit(1)

    ref_path = sys.argv[1]
    test_paths = sys.argv[2:]
    labels = ["Row-block 4P", "Column-block 4P", "2D-block 4P"]

    print(f"\n  {'Decomposition':20s}  {'Result':6s}  {'Max Abs Error':13s}  "
          f"{'L2 Error':13s}  {'Checksum Ref':16s}  {'Checksum Test':16s}")
    print("  " + "-" * 100)

    all_pass = True
    for i, test_path in enumerate(test_paths):
        label = labels[i] if i < len(labels) else os.path.basename(test_path)
        if not compare(ref_path, test_path, label):
            all_pass = False

    print()
    if all_pass:
        print("  All decompositions match serial reference within tolerance.")
    else:
        print("  WARNING: some decompositions differ from serial reference.")
    print()


if __name__ == "__main__":
    main()
