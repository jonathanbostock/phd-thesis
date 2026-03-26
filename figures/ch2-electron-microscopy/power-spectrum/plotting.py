"""FFT power spectrum analysis — ridgeline plots by distance from membrane.

Uses the same analysis approach as dna-brush-paper-2025/figures/fig-2:
  - Full FFT normalised by mean power, positive+negative frequencies combined
  - Angular profiles normalised to mean=0, std=1 before FFT
  - Per-particle data structure; bootstrap resamples over particles
  - Joblib parallelisation for the bootstrap
  - Logspace spacing grid
  - Pickle cache
"""

import math
import pickle
from pathlib import Path
from typing import Any

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm

from utils.plotting import format_axes, save_plot, setup_plot_style

setup_plot_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent
LABELLING_BASE = Path("/home/jonathandbostock/Documents/cryo-em-labelling/output")

H5_FILES: dict[str, Path] = {
    "38bp-dense": LABELLING_BASE / "38bp-dense_20251219_151625/rectified_data.h5",
    "68bp-dense": LABELLING_BASE / "68bp-dense_20251219_152258/rectified_data.h5",
    "68bp-sparse": LABELLING_BASE / "68bp-sparse_20251219_152802/rectified_data.h5",
    "star-dense": LABELLING_BASE / "star-dense_20251219_153213/rectified_data.h5",
}

DATASET_NAMES = list(H5_FILES.keys())
DATASET_TITLES = {
    "38bp-dense": "38 bp Dense",
    "68bp-dense": "68 bp Dense",
    "68bp-sparse": "68 bp Sparse",
    "star-dense": "Star Dense",
}
MARKERS = ["o", "s", "^", "D"]

# Explicit colour mapping (colorblind-safe)
_CB = sns.color_palette("colorblind")
COLORS: dict[str, Any] = {
    "38bp-dense": _CB[0],  # blue
    "68bp-dense": _CB[2],  # teal / green
    "68bp-sparse": _CB[1], # orangey-yellow
    "star-dense": _CB[3],  # vermillion / red
}

Y_LABEL = "Fourier intensity (normalized)"
Y_LIM = (-0.05, 1.05)
Y_TICKS = np.arange(0.0, 1.1, 0.1)

# Spacing range for all analysis
SPACING_MIN = 0.5   # nm
SPACING_MAX = 10.0  # nm
N_SPACING = 200
SPACING_FIT = np.logspace(np.log10(SPACING_MIN), np.log10(SPACING_MAX), N_SPACING)

# Bootstrap / peak-finding parameters
N_BOOTSTRAP = 200
RADIAL_BAND_MIN = 2.0   # nm — used for Figure 5
RADIAL_BAND_MAX = 5.0   # nm — used for Figure 5
PEAK_SEARCH_MIN = 2.0   # nm
PEAK_SEARCH_MAX = 7.0   # nm

ANALYSIS_CACHE = DATA_DIR / "analysis_cache.pkl"   # per-particle data & LOWESS
BOOTSTRAP_CACHE = DATA_DIR / "bootstrap_cache.pkl"  # bootstrap results


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_power_spectrum(
    row_data: np.ndarray,
    curve_length_px: float,
    pixel_scale_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum with mean normalisation and ±frequency combination."""
    row_data = np.asarray(row_data, dtype=float)

    fft_result = np.fft.fft(row_data)
    power_full = np.abs(fft_result)
    power_full = power_full / np.mean(power_full)

    n = len(row_data)
    frequencies = np.fft.fftfreq(n)
    curve_length_nm = curve_length_px * pixel_scale_nm
    spacing_per_sample = curve_length_nm / n

    with np.errstate(divide="ignore", invalid="ignore"):
        spacing_nm_full = np.where(
            frequencies != 0, spacing_per_sample / np.abs(frequencies), np.inf
        )

    positive_mask = frequencies > 0
    pos_indices = np.where(positive_mask)[0]

    power_combined = power_full[pos_indices].copy()
    for i, pos_idx in enumerate(pos_indices):
        pos_freq = frequencies[pos_idx]
        neg_idx = np.where(np.abs(frequencies + pos_freq) < 1e-10)[0]
        if len(neg_idx) > 0:
            power_combined[i] += power_full[neg_idx[0]]

    return spacing_nm_full[positive_mask], power_combined


def process_h5_file(
    h5_path: Path,
    spacing_range: tuple[float, float] = (SPACING_MIN, SPACING_MAX),
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]]:
    """Load H5 file and return radial distances, LOWESS curves, and per-particle data.

    Returns
    -------
    radial_distances : (n_radial,)
    lowess_curves : list of (spacing_fit, power_fit) per radial distance
    per_particle_data : per_particle_data[r][p] = (spacings, powers) for particle p at radial r
    """
    with h5py.File(h5_path, "r") as f:
        images: np.ndarray = f["images"][:]  # type: ignore[index]
        curve_lengths_px: np.ndarray = f["curve_lengths_px"][:]  # type: ignore[index]
        pixel_scale_nm = float(f.attrs["pixel_scale_nm"])  # type: ignore[arg-type]
        radial_min = float(f.attrs["radial_range_nm_min"])  # type: ignore[arg-type]
        radial_max = float(f.attrs["radial_range_nm_max"])  # type: ignore[arg-type]
        radial_samples = int(f.attrs["radial_samples"])  # type: ignore[arg-type]

    radial_distances = np.linspace(radial_min, radial_max, radial_samples)
    n_particles = images.shape[0]

    per_particle_data: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in range(radial_samples)
    ]
    lowess_curves: list[tuple[np.ndarray, np.ndarray]] = []

    for r_idx in tqdm(range(radial_samples), desc="  radial bands", leave=False):
        all_spacings: list[np.ndarray] = []
        all_powers: list[np.ndarray] = []

        for p in range(n_particles):
            profile = images[p, r_idx, :]
            # Normalise to mean=0, std=1
            std = np.std(profile)
            if std > 1e-10:
                profile = (profile - np.mean(profile)) / std
            else:
                profile = profile - np.mean(profile)

            spacing, power = compute_power_spectrum(
                profile, float(curve_lengths_px[p]), pixel_scale_nm
            )

            mask = (spacing >= spacing_range[0]) & (spacing <= spacing_range[1])
            sp = np.sort(spacing[mask])
            pw = power[mask][np.argsort(spacing[mask])]

            per_particle_data[r_idx].append((sp, pw))
            all_spacings.append(sp)
            all_powers.append(pw)

        combined_sp = np.concatenate(all_spacings)
        combined_pw = np.concatenate(all_powers)
        sort_idx = np.argsort(combined_sp)
        combined_sp = combined_sp[sort_idx]
        combined_pw = combined_pw[sort_idx]

        lowess_result = lowess(combined_pw, combined_sp, frac=0.1)
        lx, uniq = np.unique(lowess_result[:, 0], return_index=True)
        ly = lowess_result[uniq, 1]
        interp_fn = interp1d(lx, ly, kind="linear", bounds_error=False,
                             fill_value=(ly[0], ly[-1]))
        lowess_curves.append((SPACING_FIT, interp_fn(SPACING_FIT)))

    return radial_distances, lowess_curves, per_particle_data


# ---------------------------------------------------------------------------
# Bootstrap helpers (parallelised with joblib)
# ---------------------------------------------------------------------------

def _bootstrap_spectrum_iter(
    boot_idx: int,
    per_particle_data: list[list[tuple[np.ndarray, np.ndarray]]],
    selected_radial_indices: np.ndarray,
    n_particles: int,
) -> np.ndarray:
    """One bootstrap iteration for the comparison spectrum (Figure 5)."""
    np.random.seed(boot_idx)
    resampled = np.random.choice(n_particles, size=n_particles, replace=True)

    all_spacings: list[np.ndarray] = []
    all_powers: list[np.ndarray] = []
    for r_idx in selected_radial_indices:
        for p in resampled:
            sp, pw = per_particle_data[r_idx][p]
            all_spacings.append(sp)
            all_powers.append(pw)

    combined_sp = np.concatenate(all_spacings)
    combined_pw = np.concatenate(all_powers)
    sort_idx = np.argsort(combined_sp)
    lowess_result = lowess(combined_pw[sort_idx], combined_sp[sort_idx], frac=0.1)

    lx, uniq = np.unique(lowess_result[:, 0], return_index=True)
    ly = lowess_result[uniq, 1]
    interp_fn = interp1d(lx, ly, kind="linear", bounds_error=False,
                         fill_value=(ly[0], ly[-1]))
    return interp_fn(SPACING_FIT)


def _bootstrap_amplitude_iter(
    boot_idx: int,
    per_particle_data: list[list[tuple[np.ndarray, np.ndarray]]],
    n_particles: int,
    n_radial: int,
    target_spacing: float,
) -> np.ndarray:
    """One bootstrap iteration for amplitude-vs-distance (Figure 6)."""
    np.random.seed(boot_idx)
    resampled = np.random.choice(n_particles, size=n_particles, replace=True)
    local_window = 4.0  # nm

    values = np.zeros(n_radial)
    for r_idx in range(n_radial):
        all_sp: list[np.ndarray] = []
        all_pw: list[np.ndarray] = []
        for p in resampled:
            sp, pw = per_particle_data[r_idx][p]
            all_sp.append(sp)
            all_pw.append(pw)

        combined_sp = np.concatenate(all_sp)
        combined_pw = np.concatenate(all_pw)

        mask = (combined_sp >= target_spacing - local_window) & (
            combined_sp <= target_spacing + local_window
        )
        local_sp = combined_sp[mask]
        local_pw = combined_pw[mask]

        if len(local_sp) < 4:
            values[r_idx] = np.nan
            continue

        sort_idx = np.argsort(local_sp)
        lowess_result = lowess(local_pw[sort_idx], local_sp[sort_idx], frac=0.1)

        lx, uniq = np.unique(lowess_result[:, 0], return_index=True)
        ly = lowess_result[uniq, 1]
        interp_fn = interp1d(lx, ly, kind="linear", bounds_error=False,
                             fill_value=(ly[0], ly[-1]))
        values[r_idx] = float(interp_fn(target_spacing))

    return values


def compute_or_load_bootstrap(
    all_per_particle: dict[str, tuple[np.ndarray, list[list[tuple[np.ndarray, np.ndarray]]]]],
    all_raw_data: dict[str, tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]],
) -> dict[str, Any]:
    """Return bootstrap results from cache, or compute and cache them."""
    if BOOTSTRAP_CACHE.exists():
        print("Loading bootstrap cache...")
        with open(BOOTSTRAP_CACHE, "rb") as f:
            return pickle.load(f)  # type: ignore[no-any-return]

    print("Computing bootstrap (parallelised)...")
    peak_spacings: dict[str, float] = {}
    peak_powers: dict[str, float] = {}
    spectrum_boot: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    amplitude_boot: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    x_coords = np.arange(0.5, 25.0, 1.0)

    # --- Spectrum bootstrap (Figure 5) ---
    for name in DATASET_NAMES:
        radial_distances, raw_data_list = all_raw_data[name]
        r_mask = (radial_distances >= RADIAL_BAND_MIN) & (radial_distances <= RADIAL_BAND_MAX)
        selected_indices = np.where(r_mask)[0]

        # Find peak from pooled data in radial band
        all_sp = np.concatenate([raw_data_list[i][0] for i in selected_indices])
        all_pw = np.concatenate([raw_data_list[i][1] for i in selected_indices])
        sort_idx = np.argsort(all_sp)
        lowess_result = lowess(all_pw[sort_idx], all_sp[sort_idx], frac=0.1)
        lx, uniq = np.unique(lowess_result[:, 0], return_index=True)
        ly = lowess_result[uniq, 1]
        interp_fn = interp1d(lx, ly, kind="linear", bounds_error=False,
                             fill_value=(ly[0], ly[-1]))
        power_fit = interp_fn(SPACING_FIT)
        peak_mask = (SPACING_FIT >= PEAK_SEARCH_MIN) & (SPACING_FIT <= PEAK_SEARCH_MAX)
        peak_idx = int(np.argmax(power_fit[peak_mask]))
        peak_spacings[name] = float(SPACING_FIT[peak_mask][peak_idx])
        peak_powers[name] = float(power_fit[peak_mask][peak_idx])

        # Bootstrap
        _, per_particle_data = all_per_particle[name]
        n_particles = len(per_particle_data[selected_indices[0]])
        print(f"  Spectrum bootstrap: {name}...")
        curves = Parallel(n_jobs=-1)(
            delayed(_bootstrap_spectrum_iter)(
                b, per_particle_data, selected_indices, n_particles
            )
            for b in tqdm(range(N_BOOTSTRAP), desc="    bootstrap", leave=False)
        )
        arr = np.array(curves)
        spectrum_boot[name] = (np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1))

    # --- Amplitude bootstrap (Figure 6) ---
    for name in DATASET_NAMES:
        radial_distances, per_particle_data = all_per_particle[name]
        n_particles = len(per_particle_data[0])
        n_radial = len(radial_distances)
        target = peak_spacings[name]
        print(f"  Amplitude bootstrap: {name} (target spacing {target:.2f} nm)...")
        values = Parallel(n_jobs=-1)(
            delayed(_bootstrap_amplitude_iter)(
                b, per_particle_data, n_particles, n_radial, target
            )
            for b in tqdm(range(N_BOOTSTRAP), desc="    bootstrap", leave=False)
        )
        arr = np.array(values)  # (N_BOOTSTRAP, n_radial)

        # Interpolate each bootstrap result to x_coords then average
        boot_interp = np.zeros((N_BOOTSTRAP, len(x_coords)))
        for b in range(N_BOOTSTRAP):
            fn = interp1d(radial_distances, arr[b], kind="linear",
                          bounds_error=False, fill_value="extrapolate")
            boot_interp[b] = fn(x_coords)

        amplitude_boot[name] = (
            np.mean(boot_interp, axis=0),
            np.std(boot_interp, axis=0, ddof=1),
        )

    result = {
        "peak_spacings": peak_spacings,
        "peak_powers": peak_powers,
        "spectrum_boot": spectrum_boot,
        "amplitude_boot": amplitude_boot,
        "x_coords": x_coords,
    }
    print(f"Saving bootstrap cache to {BOOTSTRAP_CACHE}...")
    with open(BOOTSTRAP_CACHE, "wb") as f:
        pickle.dump(result, f)
    return result


def compute_or_load_analysis() -> tuple[
    dict[str, tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]],
    dict[str, tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]],
    dict[str, tuple[np.ndarray, list[list[tuple[np.ndarray, np.ndarray]]]]],
]:
    """Return per-particle data & LOWESS from cache, or compute and cache."""
    if ANALYSIS_CACHE.exists():
        print("Loading analysis cache...")
        with open(ANALYSIS_CACHE, "rb") as f:
            cache = pickle.load(f)
        return (
            cache["all_dataset_curves"],
            cache["all_dataset_raw_data"],
            cache["all_dataset_per_particle_data"],
        )

    all_curves: dict[str, tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]] = {}
    all_raw: dict[str, tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]] = {}
    all_per_particle: dict[str, tuple[np.ndarray, list[list[tuple[np.ndarray, np.ndarray]]]]] = {}

    for name, h5_path in H5_FILES.items():
        print(f"\nProcessing {name}...")
        radial_distances, lowess_curves, per_particle_data = process_h5_file(h5_path)

        all_curves[name] = (radial_distances, lowess_curves)
        all_per_particle[name] = (radial_distances, per_particle_data)

        # Flatten per-particle → raw_data_list (combined per radial band)
        raw_data_list: list[tuple[np.ndarray, np.ndarray]] = []
        for r_idx in range(len(per_particle_data)):
            sp = np.concatenate([sp for sp, _ in per_particle_data[r_idx]])
            pw = np.concatenate([pw for _, pw in per_particle_data[r_idx]])
            sort_idx = np.argsort(sp)
            raw_data_list.append((sp[sort_idx], pw[sort_idx]))
        all_raw[name] = (radial_distances, raw_data_list)

    print(f"\nSaving analysis cache to {ANALYSIS_CACHE}...")
    with open(ANALYSIS_CACHE, "wb") as f:
        pickle.dump(
            {
                "all_dataset_curves": all_curves,
                "all_dataset_raw_data": all_raw,
                "all_dataset_per_particle_data": all_per_particle,
            },
            f,
        )
    return all_curves, all_raw, all_per_particle


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_ridgeline(
    name: str,
    radial_distances: np.ndarray,
    lowess_curves: list[tuple[np.ndarray, np.ndarray]],
    global_max: float = 1.0,
) -> plt.Figure:
    """Multi-line plot: one line per radial distance, coloured by distance."""
    cmap = cm.plasma
    norm_cm = Normalize(radial_distances[0], radial_distances[-1])

    fig, ax = plt.subplots(figsize=(9 / 2.54, 7 / 2.54))

    for r_idx, (_, power_fit) in enumerate(lowess_curves):
        normalised = power_fit / global_max if global_max > 0 else power_fit
        color = cmap(norm_cm(radial_distances[r_idx]))
        ax.plot(SPACING_FIT, normalised, color=color, linewidth=0.8, alpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_cm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.06)
    cbar.set_label("Distance from membrane / nm", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xlabel("Spacing / nm")
    ax.set_ylabel(Y_LABEL)
    ax.set_title(DATASET_TITLES.get(name, name))
    ax.set_xlim(SPACING_MIN, SPACING_MAX)
    ax.set_ylim(Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def _plot_comparison_spectrum(boot: dict[str, Any]) -> plt.Figure:
    """Figure 5: LOWESS in 2–5 nm band, all conditions, peaks starred."""
    fig, ax = plt.subplots(figsize=(5.1, 3.06))

    for name in DATASET_NAMES:
        mean_curve, se_curve = boot["spectrum_boot"][name]
        peak_sp = boot["peak_spacings"][name]
        peak_pw = boot["peak_powers"][name]
        color = COLORS[name]

        ax.fill_between(SPACING_FIT, mean_curve - se_curve, mean_curve + se_curve,
                        alpha=0.3, color=color, linewidth=0)
        ax.plot(SPACING_FIT, mean_curve, color=color, linewidth=1.5,
                label=f"{DATASET_TITLES.get(name, name)} ({peak_sp:.1f} nm)")
        ax.plot(peak_sp, peak_pw, marker="*", markersize=10, color=color,
                markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xlabel("Spacing / nm")
    ax.set_ylabel(Y_LABEL)
    ax.set_xlim(SPACING_MIN, SPACING_MAX)
    ax.set_title("LOWESS comparison, 2–5 nm from membrane")
    ax.legend(frameon=False)
    format_axes(ax)
    return fig


def _plot_amplitude_vs_distance(boot: dict[str, Any]) -> plt.Figure:
    """Figure 6: LOWESS amplitude at peak spacing vs distance from membrane."""
    x_coords: np.ndarray = boot["x_coords"]

    fig, ax = plt.subplots(figsize=(5.1, 3.06))

    for idx, name in enumerate(DATASET_NAMES):
        mean_vals, se_vals = boot["amplitude_boot"][name]
        peak_sp = boot["peak_spacings"][name]
        color = COLORS[name]

        ax.fill_between(x_coords, mean_vals - se_vals, mean_vals + se_vals,
                        alpha=0.3, color=color, linewidth=0)
        ax.plot(x_coords, mean_vals, color=color, linewidth=1.5,
                marker=MARKERS[idx], markersize=4,
                markerfacecolor=color, markeredgecolor="black", markeredgewidth=0.5,
                label=f"{DATASET_TITLES.get(name, name)} ({peak_sp:.1f} nm)")

    ax.set_xlabel("Distance from membrane / nm")
    ax.set_ylabel(Y_LABEL)
    ax.set_xlim(math.floor(x_coords.min()), math.ceil(x_coords.max()))
    ax.set_title("Amplitude at peak spacing vs distance from membrane")
    ax.legend(frameon=False)
    format_axes(ax)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Produce all six figures."""
    # Step 1: compute/load per-particle data and LOWESS curves
    all_curves, all_raw, all_per_particle = compute_or_load_analysis()

    # Step 2: compute/load bootstrap results
    boot = compute_or_load_bootstrap(all_per_particle, all_raw)

    # Figures 1–4: ridgeline plots per condition
    global_max = max(
        power_fit.max()
        for _, lowess_curves in all_curves.values()
        for _, power_fit in lowess_curves
    )

    for name in DATASET_NAMES:
        print(f"Ridgeline: {name}...")
        radial_distances, lowess_curves = all_curves[name]
        fig = _plot_ridgeline(name, radial_distances, lowess_curves, global_max)
        out = str(DATA_DIR / f"ridgeline_{name}")
        fig.savefig(f"{out}.svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}.svg")

    # Figure 5: LOWESS comparison 2–5 nm
    print("LOWESS comparison (2–5 nm)...")
    fig5 = _plot_comparison_spectrum(boot)
    out5 = str(DATA_DIR / "lowess_comparison_2_5nm")
    save_plot(fig5, out5, resize=False)
    plt.close(fig5)
    print(f"  → {out5}.svg")

    # Figure 6: amplitude at peak spacing vs distance
    print("Amplitude vs distance...")
    fig6 = _plot_amplitude_vs_distance(boot)
    out6 = str(DATA_DIR / "peak_spacing_vs_distance")
    save_plot(fig6, out6, resize=False)
    plt.close(fig6)
    print(f"  → {out6}.svg")

    print("\nDone — 6 figures saved.")


if __name__ == "__main__":
    main()
