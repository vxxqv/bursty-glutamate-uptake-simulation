"""Reproducible simulations for glutamate release timing and uptake."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr

SEED = 20260920
PATTERNS = ("Tonic", "Poisson", "Burst")
MODELS = ("Linear", "Michaelis-Menten", "Transporter-state")
COLORS = {
    "Tonic": "#2166AC", "Poisson": "#7B3294", "Burst": "#D6604D",
    "Linear": "#4D4D4D", "Michaelis-Menten": "#1B9E77",
    "Transporter-state": "#D95F02", "grid": "#D9D9D9", "text": "#1F2933",
}


@dataclass(frozen=True)
class Protocol:
    dt_ms: float = 0.02
    duration_ms: float = 1150.0
    n_events: int = 20
    first_event_ms: float = 100.0
    last_event_ms: float = 860.0
    burst_size: int = 5
    burst_isi_ms: float = 20.0
    pulse_size_km: float = 0.25
    release_cv: float = 0.0
    km: float = 1.0
    k_loss_per_ms: float = 0.001
    clearance_fraction: float = 0.01


@dataclass(frozen=True)
class TransporterParameters:
    capacity_km: float = 0.60
    k_bind_per_km_ms: float = 8.0
    k_unbind_per_ms: float = 1.0
    k_translocate_per_ms: float = 4.0
    k_recover_per_ms: float = 0.05


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": COLORS["text"], "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"], "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"], "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7, "legend.frameon": False, "figure.dpi": 160,
        "savefig.dpi": 320, "savefig.bbox": "tight", "svg.fonttype": "none",
    })


def release_amplitudes(mean: float, cv: float, shape: tuple[int, ...],
                       rng: np.random.Generator) -> np.ndarray:
    if cv == 0:
        return np.full(shape, mean, dtype=float)
    variance = np.log1p(cv**2)
    return rng.lognormal(np.log(mean) - variance / 2, np.sqrt(variance), size=shape)


def event_times(pattern: str, p: Protocol,
                rng: np.random.Generator | None = None) -> np.ndarray:
    pattern = pattern.lower()
    if pattern == "tonic":
        return np.linspace(p.first_event_ms, p.last_event_ms, p.n_events)
    if pattern == "poisson":
        if rng is None:
            raise ValueError("Poisson timing requires a random generator")
        return np.sort(rng.uniform(p.first_event_ms, p.last_event_ms, p.n_events))
    if pattern == "burst":
        if p.n_events % p.burst_size:
            raise ValueError("n_events must be divisible by burst_size")
        n_bursts = p.n_events // p.burst_size
        final_start = p.last_event_ms - (p.burst_size - 1) * p.burst_isi_ms
        starts = np.linspace(p.first_event_ms, final_start, n_bursts)
        return np.concatenate([start + np.arange(p.burst_size) * p.burst_isi_ms
                               for start in starts])
    raise ValueError(f"Unknown pattern: {pattern}")


def gamma_renewal_times(p: Protocol, shape: float,
                        rng: np.random.Generator) -> np.ndarray:
    gaps = rng.gamma(shape=shape, scale=1.0 / shape, size=p.n_events - 1)
    gaps *= (p.last_event_ms - p.first_event_ms) / gaps.sum()
    return np.concatenate(([p.first_event_ms], p.first_event_ms + np.cumsum(gaps)))


def make_impulses(time_sets: list[np.ndarray], amplitudes: np.ndarray,
                  p: Protocol) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(round(p.duration_ms / p.dt_ms)) + 1
    impulses = np.zeros((len(time_sets), n_steps), dtype=np.float64)
    last_indices = np.empty(len(time_sets), dtype=int)
    for run, times in enumerate(time_sets):
        idx = np.clip(np.rint(times / p.dt_ms).astype(int), 0, n_steps - 1)
        np.add.at(impulses[run], idx, amplitudes[run])
        last_indices[run] = idx.max()
    return impulses, last_indices


def _broadcast(value: float | np.ndarray, n: int) -> np.ndarray:
    return np.broadcast_to(value, (n,)).astype(float).copy()


def simulate_batch(model: str, impulses: np.ndarray, last_indices: np.ndarray,
                   p: Protocol, *, linear_rate: float = 1.0, vmax: float = 1.0,
                   transporter: TransporterParameters = TransporterParameters(),
                   capacity: float | np.ndarray | None = None,
                   k_bind: float | np.ndarray | None = None,
                   k_unbind: float | np.ndarray | None = None,
                   k_translocate: float | np.ndarray | None = None,
                   k_recover: float | np.ndarray | None = None,
                   k_loss: float | np.ndarray | None = None,
                   return_trace: bool = False) -> dict[str, np.ndarray]:
    """Integrate a linear, static saturable, or transporter-state model."""
    n_runs, n_steps = impulses.shape
    g = np.zeros(n_runs)
    bound = np.zeros(n_runs)
    recovering = np.zeros(n_runs)
    peak = np.zeros(n_runs)
    auc = np.zeros(n_runs)
    threshold_auc = np.zeros(n_runs)
    clearance = np.full(n_runs, np.nan)
    minimum_free = np.ones(n_runs)
    threshold = p.pulse_size_km
    clearance_level = p.pulse_size_km * p.clearance_fraction
    loss = _broadcast(p.k_loss_per_ms if k_loss is None else k_loss, n_runs)
    cap = _broadcast(transporter.capacity_km if capacity is None else capacity, n_runs)
    kb = _broadcast(transporter.k_bind_per_km_ms if k_bind is None else k_bind, n_runs)
    ku = _broadcast(transporter.k_unbind_per_ms if k_unbind is None else k_unbind, n_runs)
    kt = _broadcast(transporter.k_translocate_per_ms if k_translocate is None
                    else k_translocate, n_runs)
    kr = _broadcast(transporter.k_recover_per_ms if k_recover is None else k_recover, n_runs)
    trace = np.zeros((n_runs, n_steps)) if return_trace else None
    free_trace = np.ones((n_runs, n_steps)) if return_trace else None

    for step in range(n_steps):
        g += impulses[:, step]
        if return_trace:
            trace[:, step] = g
        peak = np.maximum(peak, g)
        auc += g * p.dt_ms
        threshold_auc += np.maximum(g - threshold, 0.0) * p.dt_ms
        eligible = (step > last_indices) & np.isnan(clearance) & (g <= clearance_level)
        clearance[eligible] = (step - last_indices[eligible]) * p.dt_ms
        if model == "Linear":
            g = np.maximum(g - p.dt_ms * (linear_rate + loss) * g, 0.0)
        elif model == "Michaelis-Menten":
            uptake = vmax * g / (p.km + g + 1e-15)
            g = np.maximum(g - p.dt_ms * (uptake + loss * g), 0.0)
        elif model == "Transporter-state":
            free = np.clip(1.0 - bound - recovering, 0.0, 1.0)
            minimum_free = np.minimum(minimum_free, free)
            if return_trace:
                free_trace[:, step] = free
            bind_flux = cap * kb * g * free
            unbind_flux = cap * ku * bound
            d_bound = kb * g * free - (ku + kt) * bound
            d_recover = kt * bound - kr * recovering
            g = np.maximum(g + p.dt_ms * (-bind_flux + unbind_flux - loss * g), 0.0)
            bound = np.maximum(bound + p.dt_ms * d_bound, 0.0)
            recovering = np.maximum(recovering + p.dt_ms * d_recover, 0.0)
            occupied = bound + recovering
            over = occupied > 1.0
            bound[over] /= occupied[over]
            recovering[over] /= occupied[over]
        else:
            raise ValueError(f"Unknown model: {model}")

    clearance[np.isnan(clearance)] = p.duration_ms - last_indices[np.isnan(clearance)] * p.dt_ms
    out = {"peak_km": peak, "auc_km_ms": auc,
           "excess_auc_km_ms": threshold_auc, "clearance_ms": clearance,
           "minimum_free_fraction": minimum_free}
    if return_trace:
        out["trace"] = trace
        out["free_trace"] = free_trace
    return out


def single_pulse_t90(model: str, p: Protocol, *, linear_rate: float = 1.0,
                     vmax: float = 1.0,
                     transporter: TransporterParameters = TransporterParameters()) -> float:
    pulse_p = replace(p, duration_ms=120.0, first_event_ms=20.0,
                      last_event_ms=20.0, n_events=1, burst_size=1)
    impulses, last = make_impulses([np.array([20.0])],
                                   np.array([[pulse_p.pulse_size_km]]), pulse_p)
    result = simulate_batch(model, impulses, last, pulse_p, linear_rate=linear_rate,
                            vmax=vmax, transporter=transporter, return_trace=True)
    trace = result["trace"][0]
    start = int(round(20.0 / pulse_p.dt_ms))
    below = np.flatnonzero(trace[start:] <= 0.1 * pulse_p.pulse_size_km)
    return float(below[0] * pulse_p.dt_ms) if below.size else np.nan


def calibrate_models(p: Protocol, transporter: TransporterParameters
                     ) -> tuple[float, float, pd.DataFrame]:
    state_t90 = single_pulse_t90("Transporter-state", p, transporter=transporter)
    linear_rate = np.log(10.0) / state_t90
    def objective(candidate: float) -> float:
        return single_pulse_t90("Michaelis-Menten", p, vmax=candidate) - state_t90
    vmax = brentq(objective, 0.05, 20.0)
    availability_t90 = np.log(10.0) / transporter.k_recover_per_ms
    rows = [
        {"quantity": "single_pulse_t90_ms",
         "target_basis": "adult hippocampal extrasynaptic clearance within approximately 1 ms",
         "value": state_t90},
        {"quantity": "transporter_availability_t90_ms",
         "target_basis": "activity-dependent slowing returns toward baseline within 50 ms",
         "value": availability_t90},
        {"quantity": "matched_linear_rate_per_ms", "target_basis": "matched t90",
         "value": linear_rate},
        {"quantity": "matched_mm_vmax_km_per_ms", "target_basis": "matched t90",
         "value": vmax},
    ]
    return linear_rate, vmax, pd.DataFrame(rows)


def metrics_frame(model: str, pattern: str, result: dict[str, np.ndarray],
                  release_cv: float) -> pd.DataFrame:
    frame = pd.DataFrame({k: v for k, v in result.items()
                          if k not in {"trace", "free_trace"}})
    frame.insert(0, "run", np.arange(len(frame)))
    frame.insert(1, "model", model)
    frame.insert(2, "pattern", pattern)
    frame.insert(3, "release_cv", release_cv)
    return frame


def run_equal_count_experiment(p: Protocol, transporter: TransporterParameters,
                               linear_rate: float, vmax: float, n_runs: int,
                               release_cv: float,
                               rng: np.random.Generator) -> pd.DataFrame:
    amplitudes = release_amplitudes(p.pulse_size_km, release_cv,
                                    (n_runs, p.n_events), rng)
    time_sets = {
        "Tonic": [event_times("tonic", p) for _ in range(n_runs)],
        "Poisson": [event_times("poisson", p, rng) for _ in range(n_runs)],
        "Burst": [event_times("burst", p) for _ in range(n_runs)],
    }
    frames = []
    for pattern in PATTERNS:
        impulses, last = make_impulses(time_sets[pattern], amplitudes, p)
        for model in MODELS:
            result = simulate_batch(model, impulses, last, p, linear_rate=linear_rate,
                                    vmax=vmax, transporter=transporter)
            frames.append(metrics_frame(model, pattern, result, release_cv))
    return pd.concat(frames, ignore_index=True)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    metrics = ["peak_km", "auc_km_ms", "excess_auc_km_ms", "clearance_ms",
               "minimum_free_fraction"]
    rows = []
    for (model, pattern, cv), group in data.groupby(
            ["model", "pattern", "release_cv"], sort=False):
        for metric in metrics:
            values = group[metric].to_numpy()
            sem = values.std(ddof=1) / np.sqrt(values.size)
            rows.append({"model": model, "pattern": pattern, "release_cv": cv,
                         "metric": metric, "mean": values.mean(),
                         "sd": values.std(ddof=1),
                         "ci95_low": values.mean() - 1.96 * sem,
                         "ci95_high": values.mean() + 1.96 * sem})
    return pd.DataFrame(rows)


def burst_tonic_ratios(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["peak_km", "auc_km_ms", "excess_auc_km_ms", "clearance_ms"]
    for (model, cv), group in data.groupby(["model", "release_cv"]):
        for metric in metrics:
            pivot = group.pivot(index="run", columns="pattern", values=metric)
            valid = ~np.isclose(pivot["Tonic"], 0.0)
            ratios = pivot.loc[valid, "Burst"] / pivot.loc[valid, "Tonic"]
            tonic_mean = pivot["Tonic"].mean()
            ratio_of_means = (pivot["Burst"].mean() / tonic_mean
                              if not np.isclose(tonic_mean, 0.0) else np.nan)
            rows.append({"model": model, "release_cv": cv, "metric": metric,
                         "ratio_of_means": ratio_of_means,
                         "median_paired_ratio": ratios.median(),
                         "ci95_low": ratios.quantile(0.025),
                         "ci95_high": ratios.quantile(0.975)})
    return pd.DataFrame(rows)


def model_trace(model: str, times: np.ndarray, p: Protocol, linear_rate: float,
                vmax: float, transporter: TransporterParameters
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amp = np.full((1, p.n_events), p.pulse_size_km)
    impulses, last = make_impulses([times], amp, p)
    result = simulate_batch(model, impulses, last, p, linear_rate=linear_rate,
                            vmax=vmax, transporter=transporter, return_trace=True)
    return np.arange(impulses.shape[1]) * p.dt_ms, result["trace"][0], result["free_trace"][0]


def dimensionless_protocol(alpha: float, theta: float, dt: float = 0.04) -> Protocol:
    burst_isi = max(theta, 0.04)
    first = 5.0
    last = first + 38.0 * burst_isi
    duration = last + max(30.0, 12.0 * burst_isi)
    return Protocol(dt_ms=min(dt, burst_isi / 20), duration_ms=duration,
                    first_event_ms=first, last_event_ms=last,
                    burst_isi_ms=burst_isi, pulse_size_km=alpha,
                    k_loss_per_ms=0.0)


def run_regime_maps(transporter: TransporterParameters, linear_rate: float,
                    vmax: float, reference_t90_ms: float,
                    n_alpha: int = 26, n_theta: int = 26) -> pd.DataFrame:
    rows = []
    alphas = np.geomspace(0.02, 2.0, n_alpha)
    amplitudes = np.repeat(alphas[:, None], Protocol().n_events, axis=1)
    for theta in np.geomspace(0.08, 8.0, n_theta):
        p = dimensionless_protocol(1.0, theta * reference_t90_ms)
        values = {}
        for pattern in ("Tonic", "Burst"):
            times = event_times(pattern.lower(), p)
            impulses, last = make_impulses([times for _ in alphas], amplitudes, p)
            for model in MODELS:
                result = simulate_batch(model, impulses, last, p,
                                        linear_rate=linear_rate, vmax=vmax,
                                        transporter=transporter)
                values[(model, pattern)] = result["auc_km_ms"]
        for model in MODELS:
            model_ratios = values[(model, "Burst")] / values[(model, "Tonic")]
            rows.extend({"model": model, "pulse_size_over_km": alpha,
                         "burst_interval_over_single_pulse_t90": theta,
                         "burst_tonic_auc_ratio": ratio}
                        for alpha, ratio in zip(alphas, model_ratios))
    return pd.DataFrame(rows)


def run_burstiness_continuum(p: Protocol, transporter: TransporterParameters,
                             linear_rate: float, vmax: float, n_trains: int,
                             rng: np.random.Generator
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    shapes = np.exp(rng.uniform(np.log(0.2), np.log(20.0), n_trains))
    trains = [gamma_renewal_times(p, shape, rng) for shape in shapes]
    amps = np.full((n_trains, p.n_events), p.pulse_size_km)
    impulses, last = make_impulses(trains, amps, p)
    isi_cv = np.array([np.std(np.diff(t), ddof=1) / np.mean(np.diff(t)) for t in trains])
    minimum_isi = np.array([np.min(np.diff(t)) for t in trains])
    frames, correlations = [], []
    for model in MODELS:
        result = simulate_batch(model, impulses, last, p, linear_rate=linear_rate,
                                vmax=vmax, transporter=transporter)
        frame = metrics_frame(model, "Renewal", result, 0.0)
        frame.insert(4, "isi_cv", isi_cv)
        frame.insert(5, "minimum_isi_ms", minimum_isi)
        frame.insert(6, "gamma_shape", shapes)
        frames.append(frame)
        for metric in ("peak_km", "auc_km_ms"):
            rho, p_value = spearmanr(isi_cv, frame[metric])
            correlations.append({"model": model, "metric": metric,
                                 "spearman_rho": rho, "p_value": p_value,
                                 "n_trains": n_trains})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(correlations)


def run_global_sensitivity(p: Protocol, n_sets: int,
                           rng: np.random.Generator) -> pd.DataFrame:
    capacity = rng.uniform(0.3, 1.5, n_sets)
    k_bind = rng.uniform(4.0, 12.0, n_sets)
    k_unbind = rng.uniform(0.5, 2.0, n_sets)
    k_trans = rng.uniform(2.0, 8.0, n_sets)
    recovery_tau = rng.uniform(10.0, 50.0, n_sets)
    k_recover = 1.0 / recovery_tau
    pulse_size = rng.uniform(0.05, 0.80, n_sets)
    k_loss = rng.uniform(0.0, 0.005, n_sets)
    p_sens = replace(p, pulse_size_km=1.0)
    amplitudes = np.repeat(pulse_size[:, None], p.n_events, axis=1)
    outputs = {}
    for pattern in ("Tonic", "Burst"):
        times = [event_times(pattern.lower(), p_sens) for _ in range(n_sets)]
        impulses, last = make_impulses(times, amplitudes, p_sens)
        outputs[pattern] = simulate_batch(
            "Transporter-state", impulses, last, p_sens, capacity=capacity,
            k_bind=k_bind, k_unbind=k_unbind, k_translocate=k_trans,
            k_recover=k_recover, k_loss=k_loss)
    return pd.DataFrame({
        "parameter_set": np.arange(n_sets), "capacity_km": capacity,
        "k_bind_per_km_ms": k_bind, "k_unbind_per_ms": k_unbind,
        "k_translocate_per_ms": k_trans, "recovery_tau_ms": recovery_tau,
        "pulse_size_over_km": pulse_size, "k_loss_per_ms": k_loss,
        "peak_ratio_burst_tonic": outputs["Burst"]["peak_km"] / outputs["Tonic"]["peak_km"],
        "auc_ratio_burst_tonic": outputs["Burst"]["auc_km_ms"] / outputs["Tonic"]["auc_km_ms"],
        "clearance_ratio_burst_tonic": outputs["Burst"]["clearance_ms"] /
        outputs["Tonic"]["clearance_ms"]})


def run_convergence(p: Protocol, transporter: TransporterParameters,
                    linear_rate: float, vmax: float) -> pd.DataFrame:
    rows = []
    for dt in (0.08, 0.04, 0.02, 0.01):
        p_dt = replace(p, dt_ms=dt)
        amps = np.full((1, p_dt.n_events), p_dt.pulse_size_km)
        for model in MODELS:
            for pattern in ("Tonic", "Burst"):
                impulses, last = make_impulses([event_times(pattern.lower(), p_dt)], amps, p_dt)
                result = simulate_batch(model, impulses, last, p_dt,
                                        linear_rate=linear_rate, vmax=vmax,
                                        transporter=transporter)
                rows.append({"dt_ms": dt, "model": model, "pattern": pattern,
                             "peak_km": result["peak_km"][0],
                             "auc_km_ms": result["auc_km_ms"][0]})
    data = pd.DataFrame(rows)
    ref = data[data["dt_ms"] == 0.01].set_index(["model", "pattern"])
    data["peak_error_pct"] = data.apply(
        lambda r: 100 * abs(r["peak_km"] - ref.loc[(r["model"], r["pattern"]), "peak_km"])
        / ref.loc[(r["model"], r["pattern"]), "peak_km"], axis=1)
    data["auc_error_pct"] = data.apply(
        lambda r: 100 * abs(r["auc_km_ms"] - ref.loc[(r["model"], r["pattern"]), "auc_km_ms"])
        / ref.loc[(r["model"], r["pattern"]), "auc_km_ms"], axis=1)
    return data


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", facecolor="white")
    fig.savefig(out / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def figure_inputs_and_traces(p: Protocol, transporter: TransporterParameters,
                             linear_rate: float, vmax: float, out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex="col",
                             gridspec_kw={"width_ratios": [0.38, 1.0]})
    rng = np.random.default_rng(SEED + 1)
    for row, pattern in enumerate(PATTERNS):
        times = event_times(pattern.lower(), p, rng if pattern == "Poisson" else None)
        axes[row, 0].vlines(times, 0, p.pulse_size_km, color=COLORS[pattern], lw=1.3)
        axes[row, 0].scatter(times, np.full(len(times), p.pulse_size_km), s=10,
                             color=COLORS[pattern])
        axes[row, 0].set_ylabel(f"{pattern}\nrelease")
        axes[row, 0].set_ylim(0, p.pulse_size_km * 1.35)
        axes[row, 0].grid(axis="y", alpha=0.55)
        t, g, _ = model_trace("Transporter-state", times, p, linear_rate, vmax, transporter)
        axes[row, 1].plot(t, g, color=COLORS[pattern], lw=1.5)
        axes[row, 1].set_ylabel("G / Kₘ")
        axes[row, 1].grid(alpha=0.55)
        axes[row, 1].text(0.98, 0.88, f"peak {g.max():.3f}",
                          transform=axes[row, 1].transAxes, ha="right", va="top",
                          color=COLORS[pattern], weight="bold")
    axes[0, 0].set_title("A  Equal-count release protocols", loc="left", weight="bold")
    axes[0, 1].set_title("B  Transporter-state model traces", loc="left", weight="bold")
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    axes[-1, 0].set_xlim(70, 930)
    axes[-1, 1].set_xlim(70, 930)
    fig.suptitle("Release timing changes peak glutamate while event count and amplitude remain fixed",
                 y=1.01, fontsize=12.5, weight="bold")
    fig.tight_layout()
    save_figure(fig, out, "figure_1_release_protocols_and_traces")


def figure_calibration(p: Protocol, transporter: TransporterParameters,
                       linear_rate: float, vmax: float, out: Path) -> None:
    pulse_p = replace(p, duration_ms=100.0, first_event_ms=10.0,
                      last_event_ms=10.0, n_events=1, burst_size=1)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for model in MODELS:
        t, g, free = model_trace(model, np.array([10.0]), pulse_p,
                                 linear_rate, vmax, transporter)
        axes[0].plot(t - 10, g, label=model, color=COLORS[model], lw=1.7)
        if model == "Transporter-state":
            axes[1].plot(t - 10, free, color=COLORS[model], lw=1.7)
    axes[0].set(xlim=(0, 8), xlabel="Time after pulse (ms)", ylabel="G / Kₘ")
    axes[0].set_title("A  Matched single-pulse clearance", loc="left", weight="bold")
    axes[0].grid(alpha=0.55)
    axes[0].legend(fontsize=7.5)
    axes[1].set(xlim=(0, 70), ylim=(0, 1.02), xlabel="Time after pulse (ms)",
                ylabel="Free transporter fraction")
    axes[1].set_title("B  Transporter availability", loc="left", weight="bold")
    axes[1].grid(alpha=0.55)

    t90 = single_pulse_t90("Transporter-state", p, transporter=transporter)
    intervals = np.geomspace(0.2, 40, 14)
    linear_ratios, state_ratios = [], []
    for interval in intervals:
        q = dimensionless_protocol(p.pulse_size_km, interval * t90, dt=0.04)
        amps = np.full((1, q.n_events), q.pulse_size_km)
        values = {}
        for pattern in ("Tonic", "Burst"):
            imp, last = make_impulses([event_times(pattern.lower(), q)], amps, q)
            for model in ("Linear", "Transporter-state"):
                res = simulate_batch(model, imp, last, q, linear_rate=linear_rate,
                                     transporter=transporter)
                values[(model, pattern)] = res["auc_km_ms"][0]
        linear_ratios.append(values[("Linear", "Burst")] / values[("Linear", "Tonic")])
        state_ratios.append(values[("Transporter-state", "Burst")] /
                            values[("Transporter-state", "Tonic")])
    axes[2].semilogx(intervals, linear_ratios, color=COLORS["Linear"], lw=1.8,
                     label="Linear")
    axes[2].semilogx(intervals, state_ratios, color=COLORS["Transporter-state"],
                     lw=1.8, label="Transporter-state")
    axes[2].axhline(1, color="#777777", ls="--", lw=0.9)
    axes[2].set(xlabel="Within-burst interval / single-pulse t90",
                ylabel="Burst / tonic AUC")
    axes[2].set_title("C  Structural control", loc="left", weight="bold")
    axes[2].grid(alpha=0.55)
    axes[2].legend(fontsize=8)
    fig.suptitle("Calibration separates timing effects from the clearance law",
                 y=1.03, fontsize=12.5, weight="bold")
    fig.tight_layout()
    save_figure(fig, out, "figure_2_calibration_and_structural_control")


def figure_main_results(summary: pd.DataFrame, out: Path) -> None:
    data = summary[summary["release_cv"] == 0.0]
    specs = [("peak_km", "Peak G / Kₘ"), ("auc_km_ms", "AUC (Kₘ·ms)"),
             ("clearance_ms", "Post-train clearance (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    width, x = 0.23, np.arange(len(MODELS))
    for ax, (metric, ylabel) in zip(axes, specs):
        selected = data[data["metric"] == metric]
        for i, pattern in enumerate(PATTERNS):
            means = [selected[(selected["model"] == m) &
                              (selected["pattern"] == pattern)]["mean"].iloc[0]
                     for m in MODELS]
            ax.bar(x + (i - 1) * width, means, width, color=COLORS[pattern], label=pattern)
        ax.set_xticks(x, ["Linear", "Static MM", "State model"], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.55)
    for ax, title in zip(axes, ("A  Peak concentration", "B  Integrated exposure",
                               "C  Clearance time")):
        ax.set_title(title, loc="left", weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Equal release amplitudes reveal model-dependent timing effects",
                 y=1.10, fontsize=12.5, weight="bold")
    fig.tight_layout()
    save_figure(fig, out, "figure_3_model_comparison")


def figure_amplitude_variability(ratios: pd.DataFrame, out: Path) -> None:
    selected = ratios[ratios["metric"].isin(["peak_km", "auc_km_ms"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for ax, metric, title in zip(axes, ("peak_km", "auc_km_ms"),
                                 ("A  Peak ratio", "B  AUC ratio")):
        part, x, width = selected[selected["metric"] == metric], np.arange(len(MODELS)), 0.32
        for i, cv in enumerate((0.0, 0.15)):
            values = [part[(part["model"] == model) &
                           (part["release_cv"] == cv)]["ratio_of_means"].iloc[0]
                      for model in MODELS]
            ax.bar(x + (i - 0.5) * width, values, width,
                   color="#4C78A8" if cv == 0 else "#F58518", label=f"CV = {cv:.2f}")
        ax.axhline(1, color="#666666", ls="--", lw=0.9)
        ax.set_xticks(x, ["Linear", "Static MM", "State model"], rotation=18, ha="right")
        ax.set_ylabel("Burst / tonic ratio")
        ax.set_title(title, loc="left", weight="bold")
        ax.grid(axis="y", alpha=0.55)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Release-amplitude variability does not determine the direction of the result",
                 y=1.03, fontsize=12, weight="bold")
    fig.tight_layout()
    save_figure(fig, out, "figure_4_amplitude_variability")


def figure_regime_maps(regime: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True, sharey=True)
    lower = min(0.80, regime["burst_tonic_auc_ratio"].quantile(0.02))
    upper = max(1.40, regime["burst_tonic_auc_ratio"].quantile(0.98))
    levels = np.linspace(lower, upper, 21)
    norm = TwoSlopeNorm(vmin=lower, vcenter=1.0, vmax=upper)
    contour = None
    for ax, model, panel in zip(axes, MODELS, "ABC"):
        part = regime[regime["model"] == model]
        grid = part.pivot(index="pulse_size_over_km",
                          columns="burst_interval_over_single_pulse_t90",
                          values="burst_tonic_auc_ratio")
        x, y, z = grid.columns.to_numpy(), grid.index.to_numpy(), grid.to_numpy()
        contour = ax.contourf(x, y, z, levels=levels, cmap="coolwarm", norm=norm,
                              extend="both")
        if np.nanmin(z) <= 1.05 <= np.nanmax(z):
            ax.contour(x, y, z, levels=[1.05], colors="white", linewidths=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{panel}  {model}", loc="left", weight="bold")
        ax.set_xlabel("Burst interval / single-pulse t90")
        if model == "Linear":
            ax.text(0.5, 0.5, "ratio = 1 throughout", transform=ax.transAxes,
                    ha="center", va="center", weight="bold", color="#333333")
    axes[0].set_ylabel("Pulse size / Kₘ")
    cbar = fig.colorbar(contour, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Burst / tonic AUC")
    fig.suptitle("Dimensionless maps identify regimes with and without increased burst exposure",
                 y=1.02, fontsize=12.2, weight="bold")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.17, top=0.88, wspace=0.16)
    save_figure(fig, out, "figure_5_dimensionless_regime_maps")


def figure_robustness(sensitivity: pd.DataFrame, continuum: pd.DataFrame,
                      correlations: pd.DataFrame, convergence: pd.DataFrame,
                      out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].hist(sensitivity["auc_ratio_burst_tonic"], bins=32, color="#D95F02",
                 alpha=0.82, edgecolor="white")
    axes[0].axvline(1, color="#555555", ls="--", lw=1)
    axes[0].set(xlabel="Burst / tonic AUC", ylabel="Parameter sets")
    axes[0].set_title("A  Parameter sensitivity", loc="left", weight="bold")
    axes[0].grid(axis="y", alpha=0.55)
    part = continuum[continuum["model"] == "Transporter-state"]
    axes[1].scatter(part["isi_cv"], part["auc_km_ms"], s=12, alpha=0.40,
                    color="#7B3294", edgecolors="none")
    bins = pd.qcut(part["isi_cv"], 12, duplicates="drop")
    line = part.assign(bin=bins).groupby("bin", observed=True).agg(
        x=("isi_cv", "mean"), y=("auc_km_ms", "mean"))
    axes[1].plot(line["x"], line["y"], color="#3F007D", lw=2)
    rho = correlations[(correlations["model"] == "Transporter-state") &
                       (correlations["metric"] == "auc_km_ms")]["spearman_rho"].iloc[0]
    axes[1].text(0.04, 0.94, f"Spearman ρ = {rho:.2f}",
                 transform=axes[1].transAxes, ha="left", va="top", weight="bold")
    axes[1].set(xlabel="Inter-event interval CV", ylabel="AUC (Kₘ·ms)")
    axes[1].set_title("B  Timing continuum", loc="left", weight="bold")
    axes[1].grid(alpha=0.55)
    selected = convergence[np.isclose(convergence["dt_ms"], 0.02)]
    labels = [f"{m[:5]}\n{p[:1]}" for m, p in zip(selected["model"], selected["pattern"])]
    axes[2].bar(np.arange(len(selected)), selected["auc_error_pct"], color="#4C78A8")
    axes[2].set_xticks(np.arange(len(selected)), labels, fontsize=7.5)
    axes[2].set_ylabel("AUC error vs 0.01 ms (%)")
    axes[2].set_title("C  Numerical convergence", loc="left", weight="bold")
    axes[2].grid(axis="y", alpha=0.55)
    fig.suptitle("The conclusions were tested across timing, parameters, and time steps",
                 y=1.03, fontsize=12.2, weight="bold")
    fig.tight_layout()
    save_figure(fig, out, "figure_6_robustness_and_convergence")


def write_key_results(ratios: pd.DataFrame, sensitivity: pd.DataFrame,
                      correlations: pd.DataFrame, convergence: pd.DataFrame,
                      calibration: pd.DataFrame, output: Path) -> None:
    def ratio(model: str, cv: float, metric: str) -> float:
        return float(ratios[(ratios["model"] == model) &
                            (ratios["release_cv"] == cv) &
                            (ratios["metric"] == metric)]["ratio_of_means"].iloc[0])
    selected_dt = convergence[np.isclose(convergence["dt_ms"], 0.02)]
    state_corr = correlations[(correlations["model"] == "Transporter-state") &
                              (correlations["metric"] == "auc_km_ms")].iloc[0]
    payload = {
        "calibrated_single_pulse_t90_ms": float(calibration[
            calibration["quantity"] == "single_pulse_t90_ms"]["value"].iloc[0]),
        "transporter_availability_t90_ms": float(calibration[
            calibration["quantity"] == "transporter_availability_t90_ms"]["value"].iloc[0]),
        "linear_auc_ratio_cv0": ratio("Linear", 0.0, "auc_km_ms"),
        "static_mm_auc_ratio_cv0": ratio("Michaelis-Menten", 0.0, "auc_km_ms"),
        "state_auc_ratio_cv0": ratio("Transporter-state", 0.0, "auc_km_ms"),
        "state_auc_ratio_cv015": ratio("Transporter-state", 0.15, "auc_km_ms"),
        "state_peak_ratio_cv0": ratio("Transporter-state", 0.0, "peak_km"),
        "sensitivity_auc_ratio_median": float(sensitivity["auc_ratio_burst_tonic"].median()),
        "sensitivity_auc_ratio_gt_1_pct": float(100 * (sensitivity[
            "auc_ratio_burst_tonic"] > 1).mean()),
        "sensitivity_auc_ratio_gt_1_05_pct": float(100 * (sensitivity[
            "auc_ratio_burst_tonic"] > 1.05).mean()),
        "burstiness_auc_spearman_state": float(state_corr["spearman_rho"]),
        "dt_0_02_max_auc_error_pct": float(selected_dt["auc_error_pct"].max()),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--continuum-trains", type=int, default=600)
    parser.add_argument("--sensitivity-sets", type=int, default=800)
    parser.add_argument("--regime-grid", type=int, default=26)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    data_dir, figures_dir = root / "data", root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    p, transporter = Protocol(), TransporterParameters()
    rng = np.random.default_rng(args.seed)
    linear_rate, vmax, calibration = calibrate_models(p, transporter)
    cv0 = run_equal_count_experiment(p, transporter, linear_rate, vmax,
                                     args.runs, 0.0, rng)
    cv15 = run_equal_count_experiment(p, transporter, linear_rate, vmax,
                                      args.runs, 0.15, rng)
    main_data = pd.concat([cv0, cv15], ignore_index=True)
    summary, ratios = summarize(main_data), burst_tonic_ratios(main_data)
    reference_t90 = float(calibration[
        calibration["quantity"] == "single_pulse_t90_ms"]["value"].iloc[0])
    regime = run_regime_maps(transporter, linear_rate, vmax, reference_t90,
                             args.regime_grid, args.regime_grid)
    continuum, correlations = run_burstiness_continuum(
        p, transporter, linear_rate, vmax, args.continuum_trains, rng)
    sensitivity = run_global_sensitivity(p, args.sensitivity_sets, rng)
    convergence = run_convergence(p, transporter, linear_rate, vmax)
    outputs = {
        "equal_count_runs.csv": main_data, "equal_count_summary.csv": summary,
        "burst_tonic_ratios.csv": ratios, "calibration_targets.csv": calibration,
        "dimensionless_regime_map.csv": regime, "burstiness_continuum.csv": continuum,
        "burstiness_correlations.csv": correlations,
        "global_sensitivity.csv": sensitivity,
        "numerical_convergence.csv": convergence,
    }
    for name, frame in outputs.items():
        frame.to_csv(data_dir / name, index=False)
    (data_dir / "model_parameters.json").write_text(json.dumps({
        "protocol": asdict(p), "transporter": asdict(transporter),
        "matched_linear_rate_per_ms": linear_rate,
        "matched_mm_vmax_km_per_ms": vmax, "seed": args.seed}, indent=2),
        encoding="utf-8")
    write_key_results(ratios, sensitivity, correlations, convergence, calibration,
                      data_dir / "key_results.json")
    figure_inputs_and_traces(p, transporter, linear_rate, vmax, figures_dir)
    figure_calibration(p, transporter, linear_rate, vmax, figures_dir)
    figure_main_results(summary, figures_dir)
    figure_amplitude_variability(ratios, figures_dir)
    figure_regime_maps(regime, figures_dir)
    figure_robustness(sensitivity, continuum, correlations, convergence, figures_dir)
    print(f"Wrote revised analysis to {root}")


if __name__ == "__main__":
    main()
