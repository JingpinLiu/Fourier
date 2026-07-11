"""Headless, end-to-end version of notebooks/Fourier_Analysis_Demo.ipynb.

Runs the full Fourier-based analysis pipeline on the example fish (Fish 6) and saves
every figure from the manuscript automatically -- no Jupyter, no running cells by hand.
This mirrors the notebook's logic and narrative exactly; see the notebook for a cell-by-cell
walkthrough and explanation of each step.

Usage
-----
    python demo/run_fish6_pipeline.py [--data-dir data] [--out-dir results/fish6]

On first run this downloads the ~1.3 GB neuron activity file (data/CellRespZ.h5) if it
is not already present, exactly like the notebook does.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a display window
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo_functions import check_phase, check_frequency, check_amplitude, check_wavelength

WIN_RIGHT = (30, 70)
WIN_LEFT = (100, 140)
STIMULUS_WAVELENGTH = 140
STIMULUS_FREQUENCY = 1 / STIMULUS_WAVELENGTH
AMPLITUDE_CUTOFF = 0.10
REGRESSION_THRESHOLD = 0.5
# GCaMP6f calcium kernel: single-exponential decay, half-time = 0.4 s, discretized at the
# dataset's nominal ~2 Hz imaging rate. Verified against the formula in
# notebooks/Fourier_Analysis_Demo.ipynb (Step 5.1) to within 0.1% relative error.
KERNEL = [0.0, 0.0, 0.0, 0.0,
          0.5875820411424945, 0.24674332653696773, 0.10373173354329597,
          0.043610339093595935, 0.018332559683645826]
CUTOFFS = np.arange(0.01, 0.26, 0.01)


def save(name, out_dir):
    plt.savefig(out_dir / name, dpi=150)
    plt.close()
    print(f"  saved {name}")


def plot_spatial(coords_left, coords_right, outline, axes_labels, title, fname, out_dir,
                  colors_left="red", colors_right="blue", figsize=(20, 12), legend=True,
                  point_size=2, outline_point_size=3):
    """Shared plotting routine for the top/side/front spatial map views used repeatedly
    throughout Step 4 (raw spatial maps and phase-colored spatial maps alike)."""
    plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("black")

    alpha_left = 0.5 if isinstance(colors_left, str) else 1
    alpha_right = 0.5 if isinstance(colors_right, str) else 1
    left_plot = plt.scatter(*coords_left, s=point_size, alpha=alpha_left,
                             c=colors_left, label="Left stimulus")
    right_plot = plt.scatter(*coords_right, s=point_size, alpha=alpha_right,
                              c=colors_right, label="Right stimulus")
    plt.scatter(outline[:, 0], outline[:, 1], s=outline_point_size, c="white", alpha=1)

    if legend:
        leg = plt.legend(handles=[left_plot, right_plot], loc="upper right", frameon=False,
                          fontsize=16, labelcolor="white", scatterpoints=1, markerscale=8)
        for lh in leg.legend_handles:
            lh.set_alpha(0.9)

    plt.title(title, fontsize=18)
    save(fname, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "fish6")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(0)

    summary = {}
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Stimulus periodicity (Figure 1A)
    # ------------------------------------------------------------------
    print("\n=== Step 1: Stimulus periodicity ===")
    stimulus = pd.read_excel(DATA_DIR / "stimulus.xlsx", header=None).to_numpy()[0]
    stimulus_right = list([1 if x == 1 else 0 for x in stimulus])
    stimulus_left = list([1 if x == 2 else 0 for x in stimulus])

    plt.figure(figsize=(10, 2))
    plt.plot(stimulus_left[:500], c="red")
    plt.plot(stimulus_right[:500], c="blue")
    plt.xlabel("Frame")
    plt.grid(alpha=0.4)
    plt.title("Stimulus (500 Frames)")
    plt.legend(["left", "right"])
    save("step1_stimulus_signal.png", OUT_DIR)

    # ------------------------------------------------------------------
    # Step 2: Load neuron activity, download CellRespZ.h5 if needed
    # ------------------------------------------------------------------
    print("\n=== Step 2: Loading neuron activity data ===")
    dest_file = DATA_DIR / "CellRespZ.h5"
    if dest_file.exists():
        print(f"Found dataset at {dest_file}, skipping download.")
    else:
        import gdown
        url = "https://drive.google.com/uc?export=download&id=11j_hQXRwY7URa9UnZYo6vAYR5TWLgHGj"
        print(f"Downloading dataset to {dest_file} ...")
        gdown.download(url, str(dest_file), quiet=False)

    with h5py.File(dest_file, "r") as f:
        cell_resp_z = f["CellRespZ"][()]

    absIX = pd.read_excel(DATA_DIR / "absIX.xlsx")["neuron_index"].to_numpy().T
    xyz_data = pd.read_excel(DATA_DIR / "CellXYZnorm.xlsx", header=None).to_numpy()
    coordinate = [xyz_data[i] for i in absIX]

    print(f"Loaded activity matrix: shape={cell_resp_z.shape} (expected (92538, 3780))")
    cell_resp_z = [[i, x] for i, x in enumerate(cell_resp_z)]
    coordinate = [[i, x] for i, x in enumerate(coordinate)]

    index = random.randint(0, len(cell_resp_z) - 1)
    plt.figure(figsize=(10, 2))
    plt.title(f"Neuron No.{index}")
    plt.grid(alpha=0.4)
    plt.plot(cell_resp_z[index][1][:500], c="blue")
    plt.xlabel("Frame")
    plt.ylabel("deltaF/F(z)")
    save("step2_random_neuron_example.png", OUT_DIR)

    plt.figure(figsize=(10, 2))
    plt.title("Neuron No.25881")
    plt.plot(cell_resp_z[25881][1][:500], c="blue")
    plt.xlabel("Frame")
    plt.ylabel("deltaF/F(z)")
    plt.grid(alpha=0.4)
    save("step2_example_neuron_25881.png", OUT_DIR)

    activity = cell_resp_z[25881][1]
    freqs = check_frequency(activity)
    stim_locked_idx = np.where(np.isclose(freqs, STIMULUS_FREQUENCY, rtol=1e-3))[0]
    if stim_locked_idx.size > 0:
        idx = stim_locked_idx[0].item()
        wavelength = check_wavelength(activity)[idx]
        phase = check_phase(activity)[idx]
        amplitude = check_amplitude(activity)[idx]
        print(f"Example neuron 25881: frequency={freqs[idx]:.6f}, wavelength={wavelength}, "
              f"phase={phase:.2f}, amplitude={amplitude:.3f}")

        frames = np.arange(500)
        sim_signal = amplitude * np.cos(2 * np.pi * freqs[idx] * (frames - phase))
        plt.figure(figsize=(10, 4))
        plt.plot(activity[:500], color="blue", label="Neuron activity")
        plt.plot(frames, sim_signal, color="red", label="Simulated signal")
        plt.title("Simulated Stimulus-Locked Component (500 frames)")
        plt.xlabel("Frame")
        plt.ylabel("Signal amplitude / ΔF/F (z)")
        plt.grid(alpha=0.4)
        plt.legend(loc="upper right")
        save("step2_example_neuron_25881_fourier_fit.png", OUT_DIR)

    # ------------------------------------------------------------------
    # Step 2.3: Identify stimulus-locked neurons (Figure 3B)
    # ------------------------------------------------------------------
    print("\n=== Step 2.3: Identifying stimulus-locked neurons (this is the slow step) ===")
    n_neurons = len(cell_resp_z)
    stim_locked_neuronId, stim_locked_amplitude, stim_locked_phase = [], [], []

    for idx, (neuron_id, act) in enumerate(cell_resp_z):
        freqs = check_frequency(act)
        match = np.where(np.isclose(freqs, STIMULUS_FREQUENCY, rtol=1e-3))[0]
        if match.size > 0:
            k = match[0].item()
            stim_locked_neuronId.append(neuron_id)
            stim_locked_amplitude.append(check_amplitude(act)[k])
            stim_locked_phase.append(check_phase(act)[k])
        if (idx + 1) % 20000 == 0 or (idx + 1) == n_neurons:
            print(f"  Processed {idx + 1}/{n_neurons} neurons ({100 * (idx + 1) / n_neurons:.1f}%)")

    stim_locked_neuron = np.column_stack((
        np.array(stim_locked_neuronId, dtype=int),
        np.array(stim_locked_amplitude, dtype=float),
        np.array(stim_locked_phase, dtype=float),
    ))
    print(f"Found {len(stim_locked_neuron)} stimulus-locked neurons "
          f"({100 * len(stim_locked_neuron) / n_neurons:.1f}% of all neurons).")
    summary["n_stimulus_locked_neurons"] = len(stim_locked_neuron)

    plt.figure(figsize=(10, 6))
    plt.grid(alpha=0.4)
    plt.scatter(stim_locked_phase, stim_locked_amplitude, s=1)
    plt.xlabel("Phase (frames)")
    plt.ylabel("Amplitude")
    plt.title(f"n = {len(stim_locked_neuron)}")
    save("step3_all_stimlocked_amplitude_phase.png", OUT_DIR)

    plt.figure(figsize=(10, 2))
    plt.title("Distribution of Amplitude")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")
    plt.hist(stim_locked_amplitude, bins=100)
    plt.grid(alpha=0.4)
    save("step3_amplitude_distribution.png", OUT_DIR)

    plt.figure(figsize=(10, 2))
    plt.title("Distribution of Phase")
    plt.xlabel("Phase (frames)")
    plt.ylabel("Frequency")
    plt.hist(stim_locked_phase, bins=140)
    plt.grid(alpha=0.4)
    save("step3_phase_distribution.png", OUT_DIR)

    # ------------------------------------------------------------------
    # Step 3: Amplitude + phase selection (Figure 3)
    # ------------------------------------------------------------------
    print("\n=== Step 3: Amplitude/phase selection ===")
    stim_sorted = stim_locked_neuron[np.argsort(stim_locked_neuron[:, 1])[::-1]]
    n_total = len(stim_sorted)
    n_top = int(n_total * AMPLITUDE_CUTOFF)
    sorted_phase = stim_sorted[:, 2]
    sorted_amplitude = stim_sorted[:, 1]

    selection_mask = [1] * n_top + [0] * (n_total - n_top)
    colors = ["tab:blue" if x == 1 else "grey" for x in selection_mask]
    plt.figure(figsize=(10, 6))
    plt.grid(alpha=0.4)
    plt.scatter(sorted_phase, sorted_amplitude, s=1, color=colors)
    plt.xlabel("Phase (frames)")
    plt.ylabel("Amplitude")
    plt.title(f"n = {n_top}")
    save("step3_amplitude_cutoff_selection.png", OUT_DIR)

    phase = stim_sorted[:, 2]
    phase_mask = []
    for i in range(n_top):
        if WIN_RIGHT[0] <= phase[i] <= WIN_RIGHT[1]:
            phase_mask.append(1)
        elif WIN_LEFT[0] <= phase[i] <= WIN_LEFT[1]:
            phase_mask.append(2)
        else:
            phase_mask.append(0)
    phase_mask += (n_total - n_top) * [0]
    phase_mask = np.array(phase_mask)

    colors = ["tab:blue" if x == 1 else "red" if x == 2 else "grey" for x in phase_mask]
    plt.figure(figsize=(10, 6))
    plt.grid(alpha=0.4)
    plt.scatter(sorted_phase, sorted_amplitude, s=1, color=colors)
    plt.xlabel("Phase (frames)")
    plt.ylabel("Amplitude")
    plt.title(f"n = {n_top}")
    save("fig3_amplitude_phase_selection.png", OUT_DIR)

    right_stimulus_neuron = stim_sorted[phase_mask == 1][:, 0].astype(int)
    left_stimulus_neuron = stim_sorted[phase_mask == 2][:, 0].astype(int)
    print(f"Selected {len(right_stimulus_neuron)} right-stimulus and "
          f"{len(left_stimulus_neuron)} left-stimulus neurons.")
    summary["n_fourier_right_selected"] = int(len(right_stimulus_neuron))
    summary["n_fourier_left_selected"] = int(len(left_stimulus_neuron))

    # ------------------------------------------------------------------
    # Step 4: Spatial and phase mapping (Figures 4-7)
    # ------------------------------------------------------------------
    print("\n=== Step 4: Spatial and phase mapping ===")
    outline_xy = pd.read_excel(DATA_DIR / "outline_xy.xlsx", header=None).to_numpy()
    outline_yz = pd.read_excel(DATA_DIR / "outline_yz.xlsx", header=None).to_numpy()
    outline_zx = pd.read_excel(DATA_DIR / "outline_zx.xlsx", header=None).to_numpy()
    xy_lines = np.argwhere(outline_xy)
    yz_lines = np.argwhere(outline_yz)
    zx_lines = np.argwhere(outline_zx)

    def extract_xyz(indices):
        coords = np.array([coordinate[i][1] for i in indices])
        return coords[:, 0], coords[:, 1], coords[:, 2]

    left_x, left_y, left_z = extract_xyz(left_stimulus_neuron)
    right_x, right_y, right_z = extract_xyz(right_stimulus_neuron)

    plot_spatial((left_x, left_y), (right_x, right_y), xy_lines, ("x", "y"),
                 "Spatial distribution of stimulus-locked neurons (Top View)",
                 "fig4_spatial_map_top.png", OUT_DIR, figsize=(20, 12))
    plot_spatial((left_x, left_z), (right_x, right_z), yz_lines, ("x", "z"),
                 "Spatial distribution of stimulus-locked neurons (Side View)",
                 "fig4_spatial_map_side.png", OUT_DIR, figsize=(20, 6), point_size=2, outline_point_size=4)
    plot_spatial((left_y, left_z), (right_y, right_z), zx_lines[:, ::-1], ("y", "z"),
                 "Spatial distribution of stimulus-locked neurons (Front View)",
                 "fig4_spatial_map_front.png", OUT_DIR, figsize=(12, 8), point_size=1, outline_point_size=5)

    right_stimulus_phase = stim_sorted[phase_mask == 1][:, 2].astype(float) - WIN_RIGHT[0]
    left_stimulus_phase = stim_sorted[phase_mask == 2][:, 2].astype(float) - WIN_LEFT[0]

    cmap = plt.get_cmap("hsv")
    norm = plt.Normalize(vmin=0, vmax=39)
    left_phase_color = cmap(norm(left_stimulus_phase))
    right_phase_color = cmap(norm(right_stimulus_phase))

    fig, ax = plt.subplots(figsize=(12, 2))
    fig.subplots_adjust(bottom=0.5)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap="hsv", extent=[0, 40, 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("frames")
    ax.set_xticks([0, 10, 20, 30, 40])
    plt.title("Phase")
    save("fig5_phase_colorbar.png", OUT_DIR)

    plot_spatial((left_x, left_y), (right_x, right_y), xy_lines, ("x", "y"),
                 "Phase map of stimulus-locked neurons (Top View)", "fig5_phase_map_top.png", OUT_DIR,
                 colors_left=left_phase_color, colors_right=right_phase_color, figsize=(20, 12), legend=False)
    plot_spatial((left_x, left_z), (right_x, right_z), yz_lines, ("x", "z"),
                 "Phase map of stimulus-locked neurons (Side View)", "fig5_phase_map_side.png", OUT_DIR,
                 colors_left=left_phase_color, colors_right=right_phase_color, figsize=(20, 6), legend=False,
                 outline_point_size=4)
    plot_spatial((left_y, left_z), (right_y, right_z), zx_lines[:, ::-1], ("y", "z"),
                 "Phase map of stimulus-locked neurons (Front View)", "fig5_phase_map_front.png", OUT_DIR,
                 colors_left=left_phase_color, colors_right=right_phase_color, figsize=(12, 8), legend=False,
                 point_size=1, outline_point_size=5)

    # Figures 6-7: neurons grouped into four sequential phase windows
    phase_window = {"p1": (0, 10), "p2": (10, 20), "p3": (20, 30), "p4": (30, 40)}

    def build_phase_bins(phases, neuron_ids, colors_):
        phases = np.asarray(phases, dtype=float)
        neuron_ids = np.asarray(neuron_ids)
        colors_ = np.asarray(colors_)
        phase_dict = {k: [] for k in phase_window}
        color_dict = {k: [] for k in phase_window}
        for i in range(len(phases)):
            p = phases[i]
            for j, (key, (low, high)) in enumerate(phase_window.items()):
                in_bin = (p >= low) and (p <= high) if j == 0 else (p > low) and (p <= high)
                if in_bin:
                    phase_dict[key].append(neuron_ids[i])
                    color_dict[key].append(colors_[i])
                    break
        return phase_dict, color_dict

    left_phase_bins, left_color_bins = build_phase_bins(left_stimulus_phase, left_stimulus_neuron, left_phase_color)
    right_phase_bins, right_color_bins = build_phase_bins(right_stimulus_phase, right_stimulus_neuron, right_phase_color)

    def ordered_phase_subplot(left_bins, right_bins, left_colors, right_colors, title, fname, right_bins_present=True):
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        fig.patch.set_facecolor("black")
        for idx, key in enumerate(phase_window):
            ax = axes[idx]
            low, high = phase_window[key]
            left_idx = left_bins[key]
            lx = [coordinate[i][1][0] for i in left_idx]
            ly = [coordinate[i][1][1] for i in left_idx]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor("black")
            ax.scatter(xy_lines[:, 0], xy_lines[:, 1], s=2, c="white", alpha=0.5)
            if right_bins_present:
                right_idx = right_bins[key]
                rx = [coordinate[i][1][0] for i in right_idx]
                ry = [coordinate[i][1][1] for i in right_idx]
                ax.scatter(rx, ry, s=2, alpha=1, c=right_colors[key])
            ax.scatter(lx, ly, s=2, alpha=1, c=left_colors[key])
            ax.set_title(f"Frames {low}–{high}", fontsize=18, pad=10, color="white")
        fig.suptitle(title, fontsize=22, y=0.95, color="white")
        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
        ax_spec = fig.add_axes([0.125, 0.02, 0.775, 0.02])
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax_spec.imshow(gradient, aspect="auto", cmap="hsv", extent=[0, 40, 0, 1])
        ax_spec.set_yticks([]); ax_spec.set_xticks([0, 10, 20, 30, 40])
        ax_spec.set_xlabel("Frames", fontsize=14, color="white")
        ax_spec.tick_params(axis="x", colors="white")
        for spine in ax_spec.spines.values():
            spine.set_visible(False)
        ax_spec.set_facecolor("black")
        save(fname, OUT_DIR)

    ordered_phase_subplot(left_phase_bins, right_phase_bins, left_color_bins, right_color_bins,
                           "Top-View Phase Maps of Stimulus-Locked Neurons", "fig6_ordered_phase_map.png")
    ordered_phase_subplot(left_phase_bins, None, left_color_bins, None,
                           "Top-View Phase Maps of Neurons Associated with Left-sided Stimulus",
                           "fig7_left_stimulus_ordered_phase_map.png", right_bins_present=False)

    # ------------------------------------------------------------------
    # Step 5: Regression analysis (Figure 8)
    # ------------------------------------------------------------------
    print("\n=== Step 5: Regression analysis ===")
    stimulus_left_arr = np.array(stimulus_left, dtype=float)
    stimulus_right_arr = np.array(stimulus_right, dtype=float)
    stimulus_left_convolved = np.convolve(stimulus_left_arr, KERNEL, mode="same")
    stimulus_right_convolved = np.convolve(stimulus_right_arr, KERNEL, mode="same")

    plt.figure(figsize=(10, 2))
    plt.plot(stimulus_left_convolved[:500], c="red", label="left regressor")
    plt.plot(stimulus_right_convolved[:500], c="blue", label="right regressor")
    plt.xlabel("Frame")
    plt.title("Stimulus regressor after convolution with the calcium kernel (first 500 frames)")
    plt.legend()
    plt.grid(alpha=0.4)
    save("step5_stimulus_regressor.png", OUT_DIR)

    regression_left, regression_right = [], []
    for idx, (neuron_id, act) in enumerate(cell_resp_z):
        regression_left.append([neuron_id, np.corrcoef(act, stimulus_left_convolved)[0, 1]])
        regression_right.append([neuron_id, np.corrcoef(act, stimulus_right_convolved)[0, 1]])
        if (idx + 1) % 20000 == 0 or (idx + 1) == n_neurons:
            print(f"  Processed {idx + 1}/{n_neurons} neurons ({100 * (idx + 1) / n_neurons:.1f}%)")
    regression_left = np.array(regression_left)
    regression_right = np.array(regression_right)

    regression_left_cleaned = regression_left[regression_left[:, 1] >= REGRESSION_THRESHOLD, 0].astype(int)
    regression_right_cleaned = regression_right[regression_right[:, 1] >= REGRESSION_THRESHOLD, 0].astype(int)
    print(f"Left-stimulus neurons (r > {REGRESSION_THRESHOLD}): {len(regression_left_cleaned)}")
    print(f"Right-stimulus neurons (r > {REGRESSION_THRESHOLD}): {len(regression_right_cleaned)}")
    summary["n_regression_left"] = int(len(regression_left_cleaned))
    summary["n_regression_right"] = int(len(regression_right_cleaned))

    stim_locked_ids = stim_locked_neuron[:, 0].astype(int)
    is_reg_right = np.isin(stim_locked_ids, regression_right_cleaned)
    is_reg_left = np.isin(stim_locked_ids, regression_left_cleaned)
    reg_color = np.where(is_reg_right, "purple", np.where(is_reg_left, "green", "lightgrey"))
    amplitude_cutoff_value = stim_sorted[n_top - 1, 1]
    order = np.argsort(reg_color == "lightgrey")[::-1]

    plt.figure(figsize=(10, 6))
    plt.grid(alpha=0.4)
    plt.scatter(stim_locked_neuron[order, 2], stim_locked_neuron[order, 1], s=2, c=reg_color[order])
    plt.axhline(amplitude_cutoff_value, color="black", linestyle="--", linewidth=1)
    for edge in (*WIN_RIGHT, *WIN_LEFT):
        plt.axvline(edge, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Phase (frames)")
    plt.ylabel("Amplitude")
    plt.title(f"Fourier amplitude/phase of regression-identified neurons "
              f"(purple: right, n={is_reg_right.sum()}; green: left, n={is_reg_left.sum()})")
    save("fig8_regression_fourier_scatter.png", OUT_DIR)

    # Supplementary: spatial map of regression-identified neurons, same style as the
    # Fourier spatial map above (not one of the manuscript's numbered figures).
    reg_left_x, reg_left_y, reg_left_z = extract_xyz(regression_left_cleaned)
    reg_right_x, reg_right_y, reg_right_z = extract_xyz(regression_right_cleaned)

    plot_spatial((reg_left_x, reg_left_y), (reg_right_x, reg_right_y), xy_lines, ("x", "y"),
                 "Spatial distribution of regression-identified neurons (Top View)",
                 "fig8_spatial_map_top.png", OUT_DIR, figsize=(20, 12))
    plot_spatial((reg_left_x, reg_left_z), (reg_right_x, reg_right_z), yz_lines, ("x", "z"),
                 "Spatial distribution of regression-identified neurons (Side View)",
                 "fig8_spatial_map_side.png", OUT_DIR, figsize=(20, 6), point_size=2, outline_point_size=4)
    plot_spatial((reg_left_y, reg_left_z), (reg_right_y, reg_right_z), zx_lines[:, ::-1], ("y", "z"),
                 "Spatial distribution of regression-identified neurons (Front View)",
                 "fig8_spatial_map_front.png", OUT_DIR, figsize=(12, 8), point_size=1, outline_point_size=5)

    # ------------------------------------------------------------------
    # Step 6: Fourier vs. regression comparison (Figure 9)
    # ------------------------------------------------------------------
    print("\n=== Step 6: Comparing Fourier and regression populations ===")
    fourier_right_amp = stim_sorted[phase_mask == 1][:, 1]
    fourier_left_amp = stim_sorted[phase_mask == 2][:, 1]
    fourier_amplitude_all = np.concatenate([fourier_right_amp, fourier_left_amp])
    fourier_phase_all = np.concatenate([right_stimulus_phase, left_stimulus_phase])

    reg_right_amp = stim_locked_neuron[is_reg_right, 1]
    reg_right_phase = stim_locked_neuron[is_reg_right, 2] - WIN_RIGHT[0]
    reg_left_amp = stim_locked_neuron[is_reg_left, 1]
    reg_left_phase = stim_locked_neuron[is_reg_left, 2] - WIN_LEFT[0]
    reg_amplitude_all = np.concatenate([reg_right_amp, reg_left_amp])
    reg_phase_all = np.concatenate([reg_right_phase, reg_left_phase])

    plt.figure(figsize=(10, 3))
    plt.hist(fourier_phase_all, bins=40, range=(0, 40), color="red", alpha=0.6, label="Fourier-identified")
    plt.hist(reg_phase_all, bins=40, range=(0, 40), color="blue", alpha=0.6, label="Regression-identified")
    plt.xlabel("Phase relative to stimulus onset (frames)")
    plt.ylabel("Frequency")
    plt.title("Phase distribution: Fourier vs. Regression")
    plt.legend(); plt.grid(alpha=0.4)
    save("fig9a_phase_distribution_comparison.png", OUT_DIR)

    plt.figure(figsize=(10, 3))
    plt.hist(fourier_amplitude_all, bins=50, color="red", alpha=0.6, label="Fourier-identified")
    plt.hist(reg_amplitude_all, bins=50, color="blue", alpha=0.6, label="Regression-identified")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")
    plt.title("Amplitude distribution: Fourier vs. Regression")
    plt.legend(); plt.grid(alpha=0.4)
    save("fig9b_amplitude_distribution_comparison.png", OUT_DIR)

    reg_right_lookup = dict(zip(regression_right[:, 0].astype(int), regression_right[:, 1]))
    reg_left_lookup = dict(zip(regression_left[:, 0].astype(int), regression_left[:, 1]))
    fourier_selected_r = np.concatenate([
        [reg_right_lookup[i] for i in right_stimulus_neuron],
        [reg_left_lookup[i] for i in left_stimulus_neuron],
    ])
    slope, intercept, r_value, p_value, std_err = stats.linregress(fourier_amplitude_all, fourier_selected_r)

    plt.figure(figsize=(8, 6))
    plt.scatter(fourier_amplitude_all, fourier_selected_r, s=2, alpha=0.5)
    x_fit = np.array([fourier_amplitude_all.min(), fourier_amplitude_all.max()])
    plt.plot(x_fit, intercept + slope * x_fit, color="red", label=f"Fit: y = {slope:.4f}x + {intercept:.4f}")
    plt.axhline(REGRESSION_THRESHOLD, color="black", linestyle="--",
                label=f"regression threshold (r={REGRESSION_THRESHOLD})")
    plt.xlabel("Fourier amplitude")
    plt.ylabel("Correlation with stimulus regressor (r)")
    plt.title(f"Amplitude vs. regression correlation (R² = {r_value**2:.4f}, p = {p_value:.2e})")
    plt.legend(); plt.grid(alpha=0.4)
    save("fig9c_amplitude_vs_correlation.png", OUT_DIR)
    print(f"Amplitude-vs-correlation fit: R²={r_value**2:.4f}, p={p_value:.4g}")
    summary["amplitude_vs_correlation_r2"] = round(r_value**2, 4)

    above_thresh = fourier_selected_r >= REGRESSION_THRESHOLD
    plt.figure(figsize=(8, 4))
    plt.hist(fourier_selected_r[~above_thresh], bins=50, color="grey", label=f"r < {REGRESSION_THRESHOLD}")
    plt.hist(fourier_selected_r[above_thresh], bins=50, color="tab:blue", label=f"r ≥ {REGRESSION_THRESHOLD}")
    plt.xlabel("Correlation with stimulus regressor (r)")
    plt.ylabel("Frequency")
    plt.title(f"{100 * above_thresh.mean():.1f}% of Fourier-identified neurons pass the regression threshold")
    plt.legend(); plt.grid(alpha=0.4)
    save("fig9d_regression_value_distribution.png", OUT_DIR)
    summary["pct_fourier_above_regression_threshold"] = round(100 * above_thresh.mean(), 1)

    # ------------------------------------------------------------------
    # Step 7: Accuracy across amplitude cutoffs (Figure 10)
    # ------------------------------------------------------------------
    print("\n=== Step 7: Accuracy across amplitude cutoffs ===")
    regression_all_cleaned = set(regression_left_cleaned.tolist()) | set(regression_right_cleaned.tolist())
    accuracies = []
    for c in CUTOFFS:
        n_c = int(n_total * c)
        top_c = stim_sorted[:n_c]
        phases_c = top_c[:, 2]
        ids_c = top_c[:, 0].astype(int)
        in_right = (phases_c >= WIN_RIGHT[0]) & (phases_c <= WIN_RIGHT[1])
        in_left = (phases_c >= WIN_LEFT[0]) & (phases_c <= WIN_LEFT[1])
        selected_ids = set(ids_c[in_right | in_left].tolist())
        accuracies.append(100 * len(selected_ids & regression_all_cleaned) / len(regression_all_cleaned))

    plt.figure(figsize=(8, 5))
    plt.plot(CUTOFFS * 100, accuracies, marker="o")
    plt.axvline(10, color="black", linestyle="--", label="cutoff used in this pipeline (10%)")
    plt.xlabel("Amplitude cutoff (top %)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy of Fourier analysis vs. amplitude cutoff (example fish)")
    plt.legend(); plt.grid(alpha=0.4); plt.ylim(0, 100)
    save("fig10_accuracy_vs_cutoff.png", OUT_DIR)
    print(f"Accuracy at the 10% cutoff: {accuracies[9]:.1f}%")
    summary["accuracy_at_10pct_cutoff"] = round(accuracies[9], 1)

    # ------------------------------------------------------------------
    summary["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {summary['runtime_seconds']:.1f}s. All figures and summary.json saved to {OUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
