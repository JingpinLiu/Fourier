"""Reproduce the multi-fish results in the manuscript (Figures 4 and 10): aggregate
spatial maps of stimulus-locked neurons and Fourier-vs-regression accuracy across
amplitude cutoffs, computed across all 10 zebrafish in the dataset.

`notebooks/Fourier_Analysis_Demo.ipynb` walks through this exact analysis pipeline in
detail for a single example fish (Fish 6), whose data is included in `data/`. This
script extends the same pipeline across all 10 fish used in the manuscript, which
requires downloading each fish's own imaging data separately (this is too large to
bundle in the repository).

Expected data layout
---------------------
--data-root should contain one subfolder per fish, named `subject_<ID>/`, each with:
    TimeSeries.h5      # datasets 'CellRespZ' (neurons x frames) and 'absIX' (1-indexed)
    CellXYZnorm.xlsx    # neuron coordinates normalized to the common brain template
    stimulus.xlsx        # per-frame stimulus code (0 = baseline, 1 = right, 2 = left)
    timelist.xlsx         # 1-indexed frame reordering into canonical trial order

This matches the layout of the public dataset from Chen, Mu, Hu et al. (Neuron, 2018),
available at https://doi.org/10.25378/janelia.7272617.

Usage
-----
    python demo/multi_fish_accuracy.py --data-root /path/to/dataset --out-dir results

By default this processes the 10 fish used in the manuscript (IDs 6, 7, 10, 12-18).
"""
import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo_functions import check_wavelength, check_amplitude, check_phase, find_interval

DEFAULT_FISH_IDS = [6, 7, 10, 12, 13, 14, 15, 16, 17, 18]
CUTOFFS = np.arange(0.01, 0.26, 0.01)
# GCaMP6f calcium kernel: single-exponential decay, half-time = 0.4 s, discretized at the
# dataset's nominal ~2 Hz imaging rate. Verified against the formula in
# notebooks/Fourier_Analysis_Demo.ipynb (Step 5.1) to within 0.1% relative error.
KERNEL = [0.0, 0.0, 0.0, 0.0,
          0.5875820411424945, 0.24674332653696773, 0.10373173354329597,
          0.043610339093595935, 0.018332559683645826]
REGRESSION_THRESHOLD = 0.5


def process_fish(subject_dir: Path, cutoffs=CUTOFFS):
    """Run the Fourier + regression pipeline for one fish. Returns a dict with the
    per-cutoff accuracy curve and the coordinates of neurons selected at the 10% cutoff
    (for the aggregate spatial map)."""

    with h5py.File(subject_dir / "TimeSeries.h5", "r") as f:
        cell_resp_z = f["CellRespZ"][()].T
        absIX = f["absIX"][()].astype(int) - 1

    timelist = pd.read_excel(subject_dir / "timelist.xlsx", header=None).to_numpy()[0].astype(int) - 1
    cell_resp_z = np.array([trace[timelist] for trace in cell_resp_z])

    coord_data = pd.read_excel(subject_dir / "CellXYZnorm.xlsx", header=None).to_numpy()
    coordinate = np.array([coord_data[i] for i in absIX])[0]

    stimulus = pd.read_excel(subject_dir / "stimulus.xlsx", header=None).to_numpy()[0]
    stimulus = stimulus[timelist]
    stimulus_right = np.array([1 if x == 1 else 0 for x in stimulus], dtype=float)
    stimulus_left = np.array([1 if x == 2 else 0 for x in stimulus], dtype=float)

    stim_wavelength = check_wavelength(stimulus_left, 1)[0]
    stim_right_window = find_interval(stimulus_right[:int(stim_wavelength)])
    stim_left_window = find_interval(stimulus_left[:int(stim_wavelength)])

    # --- Fourier analysis: extract amplitude/phase at the stimulus frequency for every neuron
    n_neurons = len(cell_resp_z)
    stim_locked_ids, stim_locked_amp, stim_locked_phase = [], [], []
    for i in range(n_neurons):
        wavelengths = check_wavelength(cell_resp_z[i])
        match = np.where(np.array(wavelengths) == stim_wavelength)[0]
        if len(match) > 0:
            k = int(match[0])
            stim_locked_ids.append(i)
            stim_locked_amp.append(check_amplitude(cell_resp_z[i])[k])
            stim_locked_phase.append(check_phase(cell_resp_z[i])[k])
        if (i + 1) % 20000 == 0:
            print(f"  [{subject_dir.name}] Fourier: {i + 1}/{n_neurons}")

    stim_locked = np.column_stack((stim_locked_ids, stim_locked_amp, stim_locked_phase))
    order = np.argsort(stim_locked[:, 1])[::-1]
    stim_sorted = stim_locked[order]
    n_total = len(stim_sorted)

    # --- Regression analysis (ground truth for validation)
    stimulus_left_convolved = np.convolve(stimulus_left, KERNEL, mode="same")
    stimulus_right_convolved = np.convolve(stimulus_right, KERNEL, mode="same")

    reg_left_ids, reg_right_ids = [], []
    for i in range(n_neurons):
        if np.corrcoef(cell_resp_z[i], stimulus_left_convolved)[0, 1] >= REGRESSION_THRESHOLD:
            reg_left_ids.append(i)
        if np.corrcoef(cell_resp_z[i], stimulus_right_convolved)[0, 1] >= REGRESSION_THRESHOLD:
            reg_right_ids.append(i)
        if (i + 1) % 20000 == 0:
            print(f"  [{subject_dir.name}] Regression: {i + 1}/{n_neurons}")
    regression_all = set(reg_left_ids) | set(reg_right_ids)

    # --- Accuracy across amplitude cutoffs (Figure 10)
    # Uses this fish's own detected stimulus windows (stim_right_window/stim_left_window),
    # since other fish may have slightly different trial alignment than Fish 6.
    accuracies = []
    for c in cutoffs:
        n_c = int(n_total * c)
        top_c = stim_sorted[:n_c]
        phases_c = top_c[:, 2]
        ids_c = top_c[:, 0].astype(int)
        in_right = (phases_c >= stim_right_window[0]) & (phases_c <= stim_right_window[1])
        in_left = (phases_c >= stim_left_window[0]) & (phases_c <= stim_left_window[1])
        selected = set(ids_c[in_right | in_left].tolist())
        accuracies.append(100 * len(selected & regression_all) / max(len(regression_all), 1))

    # --- Spatial map at the 10% cutoff (Figure 4)
    n_top = int(n_total * 0.10)
    top10 = stim_sorted[:n_top]
    phases_top = top10[:, 2]
    ids_top = top10[:, 0].astype(int)
    right_ids = ids_top[(phases_top >= stim_right_window[0]) & (phases_top <= stim_right_window[1])]
    left_ids = ids_top[(phases_top >= stim_left_window[0]) & (phases_top <= stim_left_window[1])]

    return {
        "accuracies": accuracies,
        "right_coords": coordinate[right_ids],
        "left_coords": coordinate[left_ids],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True,
                         help="Folder containing subject_<ID>/ subfolders (see module docstring).")
    parser.add_argument("--fish", type=int, nargs="+", default=DEFAULT_FISH_IDS,
                         help=f"Fish IDs to process (default: {DEFAULT_FISH_IDS}).")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results",
                         help="Where to save plots and aggregated arrays.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_accuracies = []
    all_right_coords, all_left_coords = [], []

    for fish_id in args.fish:
        subject_dir = args.data_root / f"subject_{fish_id}"
        if not subject_dir.exists():
            print(f"Skipping fish {fish_id}: {subject_dir} not found.")
            continue
        print(f"Processing fish {fish_id}...")
        result = process_fish(subject_dir)
        all_accuracies.append(result["accuracies"])
        all_right_coords.append(result["right_coords"])
        all_left_coords.append(result["left_coords"])
        print(f"  Done. Accuracy at 10% cutoff: {result['accuracies'][9]:.1f}%")

    if not all_accuracies:
        print("No fish were processed - check --data-root.")
        return

    all_accuracies = np.array(all_accuracies)
    right_coords = np.concatenate(all_right_coords)
    left_coords = np.concatenate(all_left_coords)

    np.save(args.out_dir / "accuracy_by_cutoff.npy", all_accuracies)
    np.save(args.out_dir / "right_stimulus_coords.npy", right_coords)
    np.save(args.out_dir / "left_stimulus_coords.npy", left_coords)

    # --- Figure 10-style plot: accuracy vs. cutoff, averaged across fish
    plt.figure(figsize=(8, 5))
    for row in all_accuracies:
        plt.plot(CUTOFFS * 100, row, alpha=0.3, color="grey")
    plt.plot(CUTOFFS * 100, all_accuracies.mean(axis=0), color="red", marker="o", label="Average")
    plt.axvline(10, color="black", linestyle="--", label="cutoff used in the manuscript (10%)")
    plt.xlabel("Amplitude cutoff (top %)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy of Fourier analysis vs. amplitude cutoff (n={len(all_accuracies)} fish)")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.ylim(0, 100)
    plt.savefig(args.out_dir / "accuracy_by_cutoff.png", dpi=150)
    print(f"Average accuracy at 10% cutoff: {all_accuracies[:, 9].mean():.1f}%")

    # --- Figure 4-style plot: aggregate spatial map (top view)
    outline_xy = pd.read_excel(PROJECT_ROOT / "data" / "outline_xy.xlsx", header=None).to_numpy()
    xy_lines = np.argwhere(outline_xy)

    plt.figure(figsize=(20, 12))
    ax = plt.axes()
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.scatter(left_coords[:, 0], left_coords[:, 1], s=2, alpha=0.5, c="red", label="Left stimulus")
    plt.scatter(right_coords[:, 0], right_coords[:, 1], s=2, alpha=0.5, c="blue", label="Right stimulus")
    plt.scatter(xy_lines[:, 0], xy_lines[:, 1], s=3, c="white")
    plt.title(f"Aggregate spatial map of stimulus-locked neurons (n={len(args.fish)} fish)", color="white")
    plt.savefig(args.out_dir / "aggregate_spatial_map.png", dpi=150, facecolor="black")

    print(f"Saved results to {args.out_dir}")


if __name__ == "__main__":
    main()
