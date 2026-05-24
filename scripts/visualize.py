#!/usr/bin/env python3
"""
Cardiac Electrophysiology Simulation — Visualization Suite
Generates: voltage heatmaps, speedup/efficiency curves, summary table
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, glob, csv

RESULTS = "results"

def load_snapshot(path):
    with open(path, "rb") as f:
        N = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float64, count=N*N).reshape(N, N)
    return N, data

def plot_snapshots():
    files = sorted(glob.glob(os.path.join(RESULTS, "snapshot_*.bin")))
    if not files:
        print("No snapshots found."); return

    colors = ["#0a0a2e", "#0d47a1", "#00bcd4", "#ffeb3b", "#ff5722", "#b71c1c"]
    cmap = mcolors.LinearSegmentedColormap.from_list("cardiac", colors, N=256)

    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5))
    if n == 1: axes = [axes]

    for ax, fp in zip(axes, files):
        N, data = load_snapshot(fp)
        step = int(os.path.basename(fp).split("_")[1].split(".")[0])
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1,
                       origin="lower", interpolation="bilinear")
        ax.set_title(f"t = {step*0.1:.0f} ms", fontsize=14, fontweight="bold")
        ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")

    fig.suptitle("Membrane Voltage — Fenton-Karma Cardiac Model",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.colorbar(im, ax=axes, label="Normalized Voltage (u)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "voltage_snapshots.png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {RESULTS}/voltage_snapshots.png")
    plt.close()

def plot_benchmarks():
    csv_path = os.path.join(RESULTS, "benchmark.csv")
    if not os.path.exists(csv_path):
        print("No benchmark.csv found."); return

    data = {}; baseline = None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            s, np_, t = int(row["strategy"]), int(row["nprocs"]), float(row["time_max"])
            if np_ == 1:
                baseline = t
            else:
                data.setdefault(s, {"np": [], "time": []})
                data[s]["np"].append(np_)
                data[s]["time"].append(t)

    if baseline is None:
        print("No sequential baseline."); return

    names  = {1: "Row-block", 2: "Column-block", 3: "2D-block"}
    colors = {1: "#e74c3c", 2: "#2ecc71", 3: "#3498db"}
    marks  = {1: "o", 2: "s", 3: "D"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Execution time
    ax = axes[0]
    ax.axhline(baseline, color="gray", ls="--", lw=1.5, label=f"Sequential ({baseline:.2f}s)")
    for s in sorted(data):
        ax.plot(data[s]["np"], data[s]["time"], marker=marks[s], color=colors[s],
                label=names[s], lw=2, ms=8)
    ax.set_xlabel("MPI Processes"); ax.set_ylabel("Time (s)")
    ax.set_title("Execution Time", fontweight="bold")
    ax.set_xticks([2,4,8,16]); ax.legend(); ax.grid(alpha=0.3)

    # Speedup
    ax = axes[1]
    procs = [1,2,4,8,16]
    ax.plot(procs, procs, "k--", lw=1.5, label="Ideal")
    for s in sorted(data):
        sp = [baseline/t for t in data[s]["time"]]
        ax.plot(data[s]["np"], sp, marker=marks[s], color=colors[s],
                label=names[s], lw=2, ms=8)
    ax.set_xlabel("MPI Processes"); ax.set_ylabel("Speedup (T₁/Tₚ)")
    ax.set_title("Speedup", fontweight="bold")
    ax.set_xticks([2,4,8,16]); ax.legend(); ax.grid(alpha=0.3)

    # Efficiency
    ax = axes[2]
    ax.axhline(100, color="gray", ls="--", lw=1.5, label="Ideal (100%)")
    for s in sorted(data):
        eff = [100*baseline/(t*p) for t,p in zip(data[s]["time"], data[s]["np"])]
        ax.plot(data[s]["np"], eff, marker=marks[s], color=colors[s],
                label=names[s], lw=2, ms=8)
    ax.set_xlabel("MPI Processes"); ax.set_ylabel("Efficiency (%)")
    ax.set_title("Parallel Efficiency", fontweight="bold")
    ax.set_xticks([2,4,8,16]); ax.set_ylim(0,110); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Fenton-Karma Cardiac Simulation — MPI Benchmarks",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "benchmark_plots.png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {RESULTS}/benchmark_plots.png")
    plt.close()

def print_summary():
    csv_path = os.path.join(RESULTS, "benchmark.csv")
    if not os.path.exists(csv_path): return

    baseline = None; rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if int(row["nprocs"]) == 1: baseline = float(row["time_max"])
            rows.append(row)
    if baseline is None: return

    names = {1: "Row-block", 2: "Column-block", 3: "2D-block"}
    print("\n" + "="*70)
    print(f"{'Strategy':<14} {'Procs':>6} {'Time(s)':>10} {'Speedup':>10} {'Efficiency':>12}")
    print("="*70)
    print(f"{'Sequential':<14} {'1':>6} {baseline:>10.3f} {'1.00x':>10} {'100.0%':>12}")
    print("-"*70)
    for row in rows:
        s, np_, t = int(row["strategy"]), int(row["nprocs"]), float(row["time_max"])
        if np_ == 1: continue
        sp = baseline / t
        print(f"{names[s]:<14} {np_:>6} {t:>10.3f} {sp:>9.2f}x {100*sp/np_:>11.1f}%")
    print("="*70)

if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    plot_snapshots()
    plot_benchmarks()
    print_summary()
