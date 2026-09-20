import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import MODELS, PATTERNS, Protocol, TransporterParameters
from .core import (
    event_times,
    gamma_renewal_times,
    make_impulses,
    release_amplitudes,
    simulate_batch,
)


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
