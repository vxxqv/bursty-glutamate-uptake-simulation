from dataclasses import dataclass


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
