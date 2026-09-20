from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .config import Protocol, TransporterParameters


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

    missing = np.isnan(clearance)
    clearance[missing] = p.duration_ms - last_indices[missing] * p.dt_ms
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
