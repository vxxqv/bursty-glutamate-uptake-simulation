from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SEED = 20260714
PALETTE = {
    "tonic": "#2C7FB8",
    "poisson": "#7A5195",
    "burst": "#E76F51",
    "astrocyte": "#50BFA5",
    "glutamate": "#8E5BD9",
    "dark": "#233044",
    "grid": "#D8DEE9",
    "gold": "#F2C14E",
}
PATTERN_ORDER = ["Tonic", "Poisson", "Burst"]


@dataclass(frozen=True)
class ModelParameters:
    """Reference parameterization in normalized units."""

    dt_ms: float = 0.25
    duration_ms: float = 1_200.0
    n_events: int = 20
    first_event_ms: float = 100.0
    last_event_ms: float = 870.0
    quantum_ngu: float = 1.0
    release_cv: float = 0.15
    vmax_ngu_per_ms: float = 0.08
    km_ngu: float = 0.50
    k_loss_per_ms: float = 0.002
    threshold_ngu: float = 1.0
    clearance_ngu: float = 0.05
    burst_size: int = 5
    burst_isi_ms: float = 5.0


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE["dark"],
            "axes.labelcolor": PALETTE["dark"],
            "text.color": PALETTE["dark"],
            "xtick.color": PALETTE["dark"],
            "ytick.color": PALETTE["dark"],
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
        }
    )


def lognormal_samples(mean: float, cv: float, size: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Draw positive release quanta with a requested arithmetic mean and CV."""
    if cv == 0:
        return np.full(size, mean, dtype=float)
    sigma2 = np.log1p(cv**2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - sigma2 / 2
    return rng.lognormal(mu, sigma, size=size)


def event_times(pattern: str, p: ModelParameters, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate event times while holding the number of release events constant."""
    pattern = pattern.lower()
    if pattern == "tonic":
        return np.linspace(p.first_event_ms, p.last_event_ms, p.n_events)
    if pattern == "burst":
        if p.n_events % p.burst_size:
            raise ValueError("n_events must be divisible by burst_size")
        n_bursts = p.n_events // p.burst_size
        final_start = p.last_event_ms - (p.burst_size - 1) * p.burst_isi_ms
        starts = np.linspace(p.first_event_ms, final_start, n_bursts)
        return np.concatenate(
            [start + np.arange(p.burst_size) * p.burst_isi_ms for start in starts]
        )
    if pattern == "poisson":
        if rng is None:
            raise ValueError("A random generator is required for Poisson timing")
        return np.sort(rng.uniform(p.first_event_ms, p.last_event_ms, p.n_events))
    raise ValueError(f"Unknown pattern: {pattern}")


def gamma_renewal_times(
    n_events: int,
    first_ms: float,
    last_ms: float,
    shape: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an exact-count renewal train and rescale it to a fixed window."""
    gaps = rng.gamma(shape=shape, scale=1.0 / shape, size=n_events - 1)
    gaps *= (last_ms - first_ms) / gaps.sum()
    return np.concatenate(([first_ms], first_ms + np.cumsum(gaps)))


def make_impulses(
    times_by_run: list[np.ndarray],
    amplitudes: np.ndarray,
    p: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert event times and amplitudes into a run-by-time impulse matrix."""
    n_runs = len(times_by_run)
    n_steps = int(round(p.duration_ms / p.dt_ms)) + 1
    impulses = np.zeros((n_runs, n_steps), dtype=np.float64)
    last_indices = np.empty(n_runs, dtype=int)
    for run, times in enumerate(times_by_run):
        indices = np.clip(np.rint(times / p.dt_ms).astype(int), 0, n_steps - 1)
        np.add.at(impulses[run], indices, amplitudes[run])
        last_indices[run] = indices.max()
    return impulses, last_indices


def simulate_batch(
    impulses: np.ndarray,
    last_indices: np.ndarray,
    p: ModelParameters,
    uptake_factor: float | np.ndarray = 1.0,
    vmax: float | np.ndarray | None = None,
    km: float | np.ndarray | None = None,
    k_loss: float | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Integrate the nonlinear clearance equation for many runs in parallel.

    dG/dt = -f_u Vmax G/(Km + G) - k_loss G + sum(q_i delta(t-t_i))

    The impulsive release term is applied at the beginning of each time step.
    Explicit Euler integration is used for the smooth clearance terms.
    """
    n_runs, n_steps = impulses.shape
    vmax_arr = np.broadcast_to(p.vmax_ngu_per_ms if vmax is None else vmax, (n_runs,)).astype(float)
    km_arr = np.broadcast_to(p.km_ngu if km is None else km, (n_runs,)).astype(float)
    k_arr = np.broadcast_to(p.k_loss_per_ms if k_loss is None else k_loss, (n_runs,)).astype(float)
    uptake_arr = np.broadcast_to(uptake_factor, (n_runs,)).astype(float)

    g = np.zeros(n_runs, dtype=float)
    peak = np.zeros(n_runs, dtype=float)
    auc = np.zeros(n_runs, dtype=float)
    excess_auc = np.zeros(n_runs, dtype=float)
    time_above = np.zeros(n_runs, dtype=float)
    near_sat_time = np.zeros(n_runs, dtype=float)
    clearance = np.full(n_runs, np.nan, dtype=float)

    for step in range(n_steps):
        g += impulses[:, step]
        peak = np.maximum(peak, g)
        auc += g * p.dt_ms
        excess_auc += np.maximum(g - p.threshold_ngu, 0.0) * p.dt_ms
        time_above += (g >= p.threshold_ngu) * p.dt_ms
        # Michaelis-Menten flux is at least 80% of capacity when G >= 4 Km.
        near_sat_time += (g >= 4.0 * km_arr) * p.dt_ms

        eligible = (step > last_indices) & np.isnan(clearance) & (g <= p.clearance_ngu)
        clearance[eligible] = (step - last_indices[eligible]) * p.dt_ms

        uptake_flux = uptake_arr * vmax_arr * g / (km_arr + g + 1e-15)
        g = np.maximum(g - p.dt_ms * (uptake_flux + k_arr * g), 0.0)

    clearance[np.isnan(clearance)] = p.duration_ms - last_indices[np.isnan(clearance)] * p.dt_ms
    return {
        "peak_ngu": peak,
        "auc_ngu_ms": auc,
        "excess_auc_ngu_ms": excess_auc,
        "time_above_ms": time_above,
        "near_saturation_ms": near_sat_time,
        "clearance_ms": clearance,
    }


def simulate_batch_multi_uptake(
    impulses: np.ndarray,
    last_indices: np.ndarray,
    p: ModelParameters,
    uptake_factors: list[float],
) -> dict[str, np.ndarray]:
    """Integrate all uptake conditions together to reduce Python-loop overhead.

    Returned arrays have shape (number of uptake factors, number of runs).
    This is mathematically identical to repeated calls to ``simulate_batch``
    when the remaining model parameters are shared.
    """
    n_runs, n_steps = impulses.shape
    uptake = np.asarray(uptake_factors, dtype=float)[:, None]
    n_uptake = uptake.shape[0]
    g = np.zeros((n_uptake, n_runs), dtype=float)
    peak = np.zeros_like(g)
    auc = np.zeros_like(g)
    excess_auc = np.zeros_like(g)
    time_above = np.zeros_like(g)
    near_sat_time = np.zeros_like(g)
    clearance = np.full_like(g, np.nan)

    for step in range(n_steps):
        g += impulses[:, step][None, :]
        peak = np.maximum(peak, g)
        auc += g * p.dt_ms
        excess_auc += np.maximum(g - p.threshold_ngu, 0.0) * p.dt_ms
        time_above += (g >= p.threshold_ngu) * p.dt_ms
        near_sat_time += (g >= 4.0 * p.km_ngu) * p.dt_ms
        eligible = ((step > last_indices[None, :]) & np.isnan(clearance)
                    & (g <= p.clearance_ngu))
        clearance[eligible] = np.broadcast_to(
            (step - last_indices)[None, :] * p.dt_ms, g.shape
        )[eligible]
        flux = uptake * p.vmax_ngu_per_ms * g / (p.km_ngu + g + 1e-15)
        g = np.maximum(g - p.dt_ms * (flux + p.k_loss_per_ms * g), 0.0)

    missing = np.isnan(clearance)
    fallback = np.broadcast_to(
        (p.duration_ms - last_indices * p.dt_ms)[None, :], g.shape
    )
    clearance[missing] = fallback[missing]
    return {
        "peak_ngu": peak,
        "auc_ngu_ms": auc,
        "excess_auc_ngu_ms": excess_auc,
        "time_above_ms": time_above,
        "near_saturation_ms": near_sat_time,
        "clearance_ms": clearance,
    }


def simulate_trace(
    times: np.ndarray,
    amplitudes: np.ndarray,
    p: ModelParameters,
    uptake_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, glutamate, and uptake-flux traces for one run."""
    impulses, _ = make_impulses([times], amplitudes[None, :], p)
    n_steps = impulses.shape[1]
    t = np.arange(n_steps) * p.dt_ms
    g = np.zeros(n_steps, dtype=float)
    flux = np.zeros(n_steps, dtype=float)
    state = 0.0
    for step in range(n_steps):
        state += impulses[0, step]
        g[step] = state
        flux[step] = uptake_factor * p.vmax_ngu_per_ms * state / (p.km_ngu + state + 1e-15)
        state = max(state - p.dt_ms * (flux[step] + p.k_loss_per_ms * state), 0.0)
    return t, g, flux


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "peak_ngu",
        "auc_ngu_ms",
        "excess_auc_ngu_ms",
        "time_above_ms",
        "near_saturation_ms",
        "clearance_ms",
    ]
    rows: list[dict[str, float | str]] = []
    for (pattern, uptake), group in results.groupby(["pattern", "uptake_factor"], sort=False):
        for metric in metrics:
            values = group[metric].to_numpy()
            sem = values.std(ddof=1) / np.sqrt(values.size)
            rows.append(
                {
                    "pattern": pattern,
                    "uptake_factor": uptake,
                    "metric": metric,
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "sem": sem,
                    "ci95_low": values.mean() - 1.96 * sem,
                    "ci95_high": values.mean() + 1.96 * sem,
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 10_000,
) -> tuple[float, float, float]:
    """Bootstrap a ratio of paired-run means."""
    n = numerator.size
    indices = rng.integers(0, n, size=(n_boot, n))
    ratios = numerator[indices].mean(axis=1) / denominator[indices].mean(axis=1)
    return (
        float(numerator.mean() / denominator.mean()),
        float(np.quantile(ratios, 0.025)),
        float(np.quantile(ratios, 0.975)),
    )


def run_main_experiment(
    p: ModelParameters,
    n_runs: int,
    uptake_factors: list[float],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    amplitudes = lognormal_samples(p.quantum_ngu, p.release_cv, (n_runs, p.n_events), rng)
    time_sets: dict[str, list[np.ndarray]] = {
        "Tonic": [event_times("tonic", p) for _ in range(n_runs)],
        "Poisson": [event_times("poisson", p, rng) for _ in range(n_runs)],
        "Burst": [event_times("burst", p) for _ in range(n_runs)],
    }
    impulse_sets = {
        pattern: make_impulses(times, amplitudes, p) for pattern, times in time_sets.items()
    }

    frames: list[pd.DataFrame] = []
    for pattern in PATTERN_ORDER:
        impulses, last_indices = impulse_sets[pattern]
        all_metrics = simulate_batch_multi_uptake(impulses, last_indices, p, uptake_factors)
        for uptake_index, uptake in enumerate(uptake_factors):
            metrics = {name: values[uptake_index] for name, values in all_metrics.items()}
            frame = pd.DataFrame(metrics)
            frame.insert(0, "run", np.arange(n_runs))
            frame.insert(1, "pattern", pattern)
            frame.insert(2, "uptake_factor", uptake)
            frames.append(frame)
    results = pd.concat(frames, ignore_index=True)

    comparison_rows = []
    boot_rng = np.random.default_rng(SEED + 11)
    for uptake in uptake_factors:
        selected = results[results["uptake_factor"] == uptake]
        for metric in ["peak_ngu", "auc_ngu_ms", "time_above_ms", "clearance_ms"]:
            pivot = selected.pivot(index="run", columns="pattern", values=metric)
            ratio, low, high = paired_bootstrap_ratio(
                pivot["Burst"].to_numpy(), pivot["Tonic"].to_numpy(), boot_rng
            )
            comparison_rows.append(
                {
                    "uptake_factor": uptake,
                    "metric": metric,
                    "comparison": "Burst/Tonic",
                    "ratio": ratio,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return results, pd.DataFrame(comparison_rows)


def run_burstiness_continuum(
    p: ModelParameters,
    n_trains: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shapes = np.exp(rng.uniform(np.log(0.25), np.log(20.0), n_trains))
    trains = [
        gamma_renewal_times(p.n_events, p.first_event_ms, p.last_event_ms, shape, rng)
        for shape in shapes
    ]
    amplitudes = lognormal_samples(p.quantum_ngu, p.release_cv, (n_trains, p.n_events), rng)
    impulses, last_indices = make_impulses(trains, amplitudes, p)
    isi_cv = np.array([np.std(np.diff(t), ddof=1) / np.mean(np.diff(t)) for t in trains])
    min_isi = np.array([np.min(np.diff(t)) for t in trains])

    frames = []
    correlations = []
    uptake_factors = [0.5, 1.0]
    all_metrics = simulate_batch_multi_uptake(impulses, last_indices, p, uptake_factors)
    for uptake_index, uptake in enumerate(uptake_factors):
        metrics = {name: values[uptake_index] for name, values in all_metrics.items()}
        frame = pd.DataFrame(metrics)
        frame.insert(0, "train", np.arange(n_trains))
        frame.insert(1, "gamma_shape", shapes)
        frame.insert(2, "isi_cv", isi_cv)
        frame.insert(3, "minimum_isi_ms", min_isi)
        frame.insert(4, "uptake_factor", uptake)
        frames.append(frame)
        for metric in ["peak_ngu", "auc_ngu_ms"]:
            rho, p_value = spearmanr(isi_cv, frame[metric])
            correlations.append(
                {
                    "uptake_factor": uptake,
                    "metric": metric,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "n_trains": n_trains,
                }
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(correlations)


def run_global_sensitivity(
    p: ModelParameters,
    n_sets: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Test the Burst/Tonic conclusion across broad parameter ranges."""
    q = rng.uniform(0.5, 1.5, n_sets)
    vmax = rng.uniform(0.04, 0.12, n_sets)
    km = rng.uniform(0.25, 1.0, n_sets)
    k_loss = rng.uniform(0.001, 0.004, n_sets)
    uptake = rng.uniform(0.35, 1.5, n_sets)

    amp = np.repeat(q[:, None], p.n_events, axis=1)
    tonic_times = [event_times("tonic", p) for _ in range(n_sets)]
    burst_times = [event_times("burst", p) for _ in range(n_sets)]
    tonic_impulses, tonic_last = make_impulses(tonic_times, amp, p)
    burst_impulses, burst_last = make_impulses(burst_times, amp, p)
    tonic = simulate_batch(tonic_impulses, tonic_last, p, uptake, vmax, km, k_loss)
    burst = simulate_batch(burst_impulses, burst_last, p, uptake, vmax, km, k_loss)
    return pd.DataFrame(
        {
            "parameter_set": np.arange(n_sets),
            "quantum_ngu": q,
            "vmax_ngu_per_ms": vmax,
            "km_ngu": km,
            "k_loss_per_ms": k_loss,
            "uptake_factor": uptake,
            "peak_ratio_burst_tonic": burst["peak_ngu"] / tonic["peak_ngu"],
            "auc_ratio_burst_tonic": burst["auc_ngu_ms"] / tonic["auc_ngu_ms"],
            "time_above_difference_ms": burst["time_above_ms"] - tonic["time_above_ms"],
        }
    )


def run_convergence_test(p: ModelParameters) -> pd.DataFrame:
    rows = []
    for dt in [0.5, 0.25, 0.10, 0.05]:
        p_dt = ModelParameters(**{**asdict(p), "dt_ms": dt})
        for pattern in ["tonic", "burst"]:
            times = event_times(pattern, p_dt)
            amp = np.full((1, p_dt.n_events), p_dt.quantum_ngu)
            impulses, last = make_impulses([times], amp, p_dt)
            metrics = simulate_batch(impulses, last, p_dt, uptake_factor=1.0)
            rows.append(
                {
                    "dt_ms": dt,
                    "pattern": pattern.title(),
                    "peak_ngu": metrics["peak_ngu"][0],
                    "auc_ngu_ms": metrics["auc_ngu_ms"][0],
                }
            )
    data = pd.DataFrame(rows)
    reference = data[data["dt_ms"] == 0.05].set_index("pattern")
    data["peak_relative_error_pct"] = data.apply(
        lambda r: 100 * abs(r["peak_ngu"] - reference.loc[r["pattern"], "peak_ngu"])
        / reference.loc[r["pattern"], "peak_ngu"],
        axis=1,
    )
    data["auc_relative_error_pct"] = data.apply(
        lambda r: 100 * abs(r["auc_ngu_ms"] - reference.loc[r["pattern"], "auc_ngu_ms"])
        / reference.loc[r["pattern"], "auc_ngu_ms"],
        axis=1,
    )
    return data


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def plot_patterns_and_traces(p: ModelParameters, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 7.0), sharex=True,
                             gridspec_kw={"width_ratios": [0.36, 1.0]})
    example_rng = np.random.default_rng(SEED + 101)
    amplitudes = lognormal_samples(p.quantum_ngu, p.release_cv, (p.n_events,), example_rng)
    for row, pattern in enumerate(PATTERN_ORDER):
        key = pattern.lower()
        times = event_times(key, p, example_rng if key == "poisson" else None)
        color = PALETTE[key]
        axes[row, 0].vlines(times, 0, amplitudes, color=color, lw=1.3)
        axes[row, 0].scatter(times, amplitudes, s=8, color=color, zorder=3)
        axes[row, 0].set_ylim(0, 1.55)
        axes[row, 0].set_ylabel(f"{pattern}\nrelease (NGU)")
        axes[row, 0].grid(axis="y", alpha=0.6)
        t, g, _ = simulate_trace(times, amplitudes, p, uptake_factor=1.0)
        axes[row, 1].plot(t, g, color=color, lw=1.6)
        axes[row, 1].axhline(p.threshold_ngu, ls="--", lw=0.9, color="#697586")
        axes[row, 1].set_ylabel("G (NGU)")
        axes[row, 1].grid(alpha=0.55)
        axes[row, 1].text(0.985, 0.88, f"peak = {g.max():.2f} NGU",
                          transform=axes[row, 1].transAxes, ha="right", va="top", color=color,
                          fontsize=8.5, weight="bold")
    axes[0, 0].set_title("A  Equal-count input patterns", loc="left", weight="bold")
    axes[0, 1].set_title("B  Resulting extracellular glutamate traces", loc="left", weight="bold")
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlim(50, 1_050)
    axes[-1, 0].set_xlim(50, 1_050)
    fig.suptitle("The same 20 release events produce different glutamate dynamics", y=1.01,
                 fontsize=13, weight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_1_input_patterns_and_traces")


def plot_main_outcomes(summary: pd.DataFrame, output_dir: Path) -> None:
    metric_specs = [
        ("peak_ngu", "Peak G (NGU)", "A"),
        ("auc_ngu_ms", "Glutamate burden, AUC (NGU·ms)", "B"),
        ("time_above_ms", "Time above 1 NGU (ms)", "C"),
        ("clearance_ms", "Post-train clearance (ms)", "D"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    for ax, (metric, ylabel, panel) in zip(axes.flat, metric_specs):
        selected = summary[summary["metric"] == metric]
        for pattern in PATTERN_ORDER:
            data = selected[selected["pattern"] == pattern].sort_values("uptake_factor")
            x = data["uptake_factor"].to_numpy() * 100
            y = data["mean"].to_numpy()
            low = y - data["ci95_low"].to_numpy()
            high = data["ci95_high"].to_numpy() - y
            ax.errorbar(x, y, yerr=np.vstack((low, high)), color=PALETTE[pattern.lower()],
                        marker="o", ms=4.8, lw=1.8, capsize=2.5, label=pattern)
        ax.set_title(f"{panel}  {ylabel.split(' (')[0]}", loc="left", weight="bold")
        ax.set_xlabel("Astrocytic uptake capacity (% of reference)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=3)
    fig.suptitle("Temporal clustering interacts with nonlinear uptake capacity", y=1.07,
                 fontsize=13, weight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_2_uptake_capacity_outcomes")


def plot_mechanism(p: ModelParameters, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex=True)
    amplitudes = np.full(p.n_events, p.quantum_ngu)
    for col, uptake in enumerate([1.0, 0.5]):
        for pattern in ["tonic", "burst"]:
            times = event_times(pattern, p)
            t, g, flux = simulate_trace(times, amplitudes, p, uptake)
            color = PALETTE[pattern]
            axes[0, col].plot(t, g, color=color, lw=1.7, label=pattern.title())
            axes[1, col].plot(t, 100 * flux / (uptake * p.vmax_ngu_per_ms), color=color, lw=1.7)
        axes[0, col].axhline(4 * p.km_ngu, color="#697586", ls="--", lw=0.9,
                            label="80% uptake saturation" if col == 0 else None)
        axes[0, col].set_title(f"{'A' if col == 0 else 'B'}  Uptake capacity = {uptake * 100:.0f}%",
                               loc="left", weight="bold")
        axes[0, col].set_ylabel("G (NGU)")
        axes[1, col].set_ylabel("Uptake flux (% of capacity)")
        axes[1, col].set_xlabel("Time (ms)")
        axes[1, col].set_ylim(0, 105)
        for row in [0, 1]:
            axes[row, col].set_xlim(60, 980)
            axes[row, col].grid(alpha=0.55)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Bursts keep the uptake pathway near capacity for longer", y=1.01,
                 fontsize=13, weight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_3_nonlinear_uptake_mechanism")


def plot_burstiness(data: pd.DataFrame, correlations: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex=True)
    metric_specs = [("peak_ngu", "Peak G (NGU)"), ("auc_ngu_ms", "AUC (NGU·ms)")]
    for row, uptake in enumerate([1.0, 0.5]):
        subset = data[data["uptake_factor"] == uptake]
        for col, (metric, ylabel) in enumerate(metric_specs):
            ax = axes[row, col]
            ax.scatter(subset["isi_cv"], subset[metric], c=subset["minimum_isi_ms"],
                       cmap="viridis_r", s=14, alpha=0.48, edgecolors="none")
            bins = pd.qcut(subset["isi_cv"], q=12, duplicates="drop")
            binned = subset.assign(bin=bins).groupby("bin", observed=True).agg(
                x=("isi_cv", "mean"), y=(metric, "mean")
            )
            ax.plot(binned["x"], binned["y"], color="#1E3A5F", lw=2.2)
            corr = correlations[(correlations["uptake_factor"] == uptake) &
                                (correlations["metric"] == metric)].iloc[0]
            ax.text(0.04, 0.94, f"Spearman ρ = {corr['spearman_rho']:.2f}",
                    transform=ax.transAxes, ha="left", va="top", weight="bold")
            ax.set_title(f"{'ABCD'[row * 2 + col]}  Uptake {uptake * 100:.0f}%: {ylabel}",
                         loc="left", weight="bold")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.55)
    axes[1, 0].set_xlabel("Inter-event interval coefficient of variation")
    axes[1, 1].set_xlabel("Inter-event interval coefficient of variation")
    sm = mpl.cm.ScalarMappable(cmap="viridis_r", norm=mpl.colors.Normalize(
        vmin=data["minimum_isi_ms"].min(), vmax=data["minimum_isi_ms"].quantile(0.98)))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Minimum inter-event interval (ms)")
    fig.suptitle("Burstiness predicts glutamate burden across 600 exact-count event trains",
                 y=1.01, fontsize=13, weight="bold")
    fig.subplots_adjust(top=0.93, bottom=0.09, left=0.09, right=0.91, hspace=0.28, wspace=0.24)
    save_figure(fig, output_dir, "figure_4_burstiness_continuum")


def plot_sensitivity(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6))
    specs = [
        ("peak_ratio_burst_tonic", "Burst/Tonic peak ratio", PALETTE["burst"]),
        ("auc_ratio_burst_tonic", "Burst/Tonic AUC ratio", PALETTE["poisson"]),
    ]
    for idx, (metric, xlabel, color) in enumerate(specs):
        values = data[metric]
        axes[idx].hist(values, bins=28, color=color, alpha=0.82, edgecolor="white")
        axes[idx].axvline(1.0, color=PALETTE["dark"], ls="--", lw=1.1)
        axes[idx].axvline(values.median(), color="#7F1D1D", lw=1.6)
        axes[idx].set_title(f"{'AB'[idx]}  {xlabel}", loc="left", weight="bold")
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel("Parameter sets")
        axes[idx].grid(axis="y", alpha=0.55)
        axes[idx].text(0.97, 0.92, f"median = {values.median():.2f}\n"
                       f">1 in {(values > 1).mean() * 100:.1f}%",
                       transform=axes[idx].transAxes, ha="right", va="top", fontsize=8.4)
    scatter = axes[2].scatter(data["uptake_factor"] * data["vmax_ngu_per_ms"],
                              data["auc_ratio_burst_tonic"], c=data["km_ngu"],
                              cmap="cividis", s=18, alpha=0.65, edgecolors="none")
    axes[2].axhline(1.0, color=PALETTE["dark"], ls="--", lw=1.1)
    axes[2].set_title("C  Effect across uptake regimes", loc="left", weight="bold")
    axes[2].set_xlabel("Effective uptake capacity (NGU/ms)")
    axes[2].set_ylabel("Burst/Tonic AUC ratio")
    axes[2].grid(alpha=0.55)
    cbar = fig.colorbar(scatter, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Kₘ (NGU)")
    fig.suptitle("Main conclusion remains stable across 500 parameter sets", y=1.04,
                 fontsize=13, weight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_5_global_sensitivity")


def write_key_results(
    results: pd.DataFrame,
    comparisons: pd.DataFrame,
    correlations: pd.DataFrame,
    sensitivity: pd.DataFrame,
    convergence: pd.DataFrame,
    output_path: Path,
) -> None:
    def comp(uptake: float, metric: str) -> dict[str, float]:
        return comparisons[(comparisons["uptake_factor"] == uptake) &
                           (comparisons["metric"] == metric)].iloc[0].to_dict()

    baseline_peak = comp(1.0, "peak_ngu")
    low_peak = comp(0.5, "peak_ngu")
    baseline_auc = comp(1.0, "auc_ngu_ms")
    low_auc = comp(0.5, "auc_ngu_ms")
    corr_peak_base = correlations[(correlations["uptake_factor"] == 1.0) &
                                  (correlations["metric"] == "peak_ngu")].iloc[0]
    corr_auc_low = correlations[(correlations["uptake_factor"] == 0.5) &
                                (correlations["metric"] == "auc_ngu_ms")].iloc[0]
    selected_dt = convergence[np.isclose(convergence["dt_ms"], ModelParameters().dt_ms)]
    payload = {
        "n_main_runs_per_condition": int(results["run"].nunique()),
        "n_sensitivity_parameter_sets": int(len(sensitivity)),
        "burst_to_tonic_peak_ratio_reference": baseline_peak,
        "burst_to_tonic_peak_ratio_half_uptake": low_peak,
        "burst_to_tonic_auc_ratio_reference": baseline_auc,
        "burst_to_tonic_auc_ratio_half_uptake": low_auc,
        "burstiness_peak_spearman_reference": float(corr_peak_base["spearman_rho"]),
        "burstiness_auc_spearman_half_uptake": float(corr_auc_low["spearman_rho"]),
        "sensitivity_peak_ratio_gt_one_pct": float(100 * (sensitivity["peak_ratio_burst_tonic"] > 1).mean()),
        "sensitivity_auc_ratio_gt_one_pct": float(100 * (sensitivity["auc_ratio_burst_tonic"] > 1).mean()),
        "sensitivity_peak_ratio_median": float(sensitivity["peak_ratio_burst_tonic"].median()),
        "sensitivity_auc_ratio_median": float(sensitivity["auc_ratio_burst_tonic"].median()),
        "selected_dt_ms": ModelParameters().dt_ms,
        "selected_dt_max_peak_error_pct": float(selected_dt["peak_relative_error_pct"].max()),
        "selected_dt_max_auc_error_pct": float(selected_dt["auc_relative_error_pct"].max()),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Project folder that will receive data/ and figures/ outputs.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--runs", type=int, default=300, help="Monte Carlo runs per condition")
    parser.add_argument("--burstiness-trains", type=int, default=600)
    parser.add_argument("--sensitivity-sets", type=int, default=500)
    args = parser.parse_args()

    root = args.output_dir.resolve()
    data_dir = root / "data"
    figures_dir = root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    p = ModelParameters()
    rng = np.random.default_rng(args.seed)
    uptake_factors = [0.50, 0.75, 1.00, 1.25, 1.50]

    results, comparisons = run_main_experiment(p, args.runs, uptake_factors, rng)
    summary = summarize_results(results)
    burstiness, correlations = run_burstiness_continuum(p, args.burstiness_trains, rng)
    sensitivity = run_global_sensitivity(p, args.sensitivity_sets, rng)
    convergence = run_convergence_test(p)

    results.to_csv(data_dir / "main_simulation_runs.csv", index=False)
    summary.to_csv(data_dir / "main_simulation_summary.csv", index=False)
    comparisons.to_csv(data_dir / "burst_tonic_comparisons.csv", index=False)
    burstiness.to_csv(data_dir / "burstiness_continuum.csv", index=False)
    correlations.to_csv(data_dir / "burstiness_correlations.csv", index=False)
    sensitivity.to_csv(data_dir / "global_sensitivity.csv", index=False)
    convergence.to_csv(data_dir / "numerical_convergence.csv", index=False)
    (data_dir / "model_parameters.json").write_text(
        json.dumps({**asdict(p), "seed": args.seed, "uptake_factors": uptake_factors}, indent=2),
        encoding="utf-8",
    )
    write_key_results(results, comparisons, correlations, sensitivity, convergence,
                      data_dir / "key_results.json")

    plot_patterns_and_traces(p, figures_dir)
    plot_main_outcomes(summary, figures_dir)
    plot_mechanism(p, figures_dir)
    plot_burstiness(burstiness, correlations, figures_dir)
    plot_sensitivity(sensitivity, figures_dir)
    print(f"Wrote reproducible outputs to {root}")


if __name__ == "__main__":
    main()
