from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from .analysis import dimensionless_protocol, model_trace
from .config import COLORS, MODELS, PATTERNS, SEED, Protocol, TransporterParameters
from .core import (
    event_times,
    make_impulses,
    simulate_batch,
    single_pulse_t90,
    transporter_recovery_t90,
)


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


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", facecolor="white", bbox_inches="tight",
                pad_inches=0.08)
    fig.savefig(out / f"{stem}.svg", facecolor="white", bbox_inches="tight",
                pad_inches=0.08)
    plt.close(fig)


def figure_inputs_and_traces(p: Protocol, transporter: TransporterParameters,
                             linear_rate: float, vmax: float, out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 7.2), sharex="col",
                             layout="constrained",
                             gridspec_kw={"width_ratios": [0.38, 1.0]})
    rng = np.random.default_rng(SEED + 1)
    labels = {"Tonic": "Tonic", "Poisson": "Fixed-count\nrandom", "Burst": "Burst"}
    for row, pattern in enumerate(PATTERNS):
        times = event_times(pattern.lower(), p, rng if pattern == "Poisson" else None)
        axes[row, 0].vlines(times, 0, p.pulse_size_km, color=COLORS[pattern], lw=1.3)
        axes[row, 0].scatter(times, np.full(len(times), p.pulse_size_km), s=10,
                             color=COLORS[pattern])
        axes[row, 0].set_ylabel(f"{labels[pattern]}\nrelease")
        axes[row, 0].set_ylim(0, p.pulse_size_km * 1.35)
        axes[row, 0].grid(axis="y", alpha=0.55)
        t, g, _ = model_trace("Transporter-state", times, p, linear_rate, vmax, transporter)
        axes[row, 1].plot(t, g, color=COLORS[pattern], lw=1.5)
        axes[row, 1].set_ylabel("G / Kₘ")
        axes[row, 1].grid(alpha=0.55)
        axes[row, 1].text(0.98, 0.88, f"peak {g.max():.3f}",
                          transform=axes[row, 1].transAxes, ha="right", va="top",
                          color=COLORS[pattern], weight="bold")
    axes[0, 0].set_title("A  Equal-count release inputs", loc="left", weight="bold")
    axes[0, 1].set_title("B  Transporter-state glutamate traces", loc="left", weight="bold")
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    axes[-1, 0].set_xlim(70, 930)
    axes[-1, 1].set_xlim(70, 930)
    fig.suptitle("Equal-count release protocols and extracellular glutamate traces",
                 fontsize=12.5, weight="bold")
    save_figure(fig, out, "figure_1_release_protocols_and_traces")


def figure_calibration(p: Protocol, transporter: TransporterParameters,
                       linear_rate: float, vmax: float, out: Path) -> None:
    pulse_p = replace(p, duration_ms=100.0, first_event_ms=10.0,
                      last_event_ms=10.0, n_events=1, burst_size=1)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), layout="constrained")
    for model in MODELS:
        t, g, free = model_trace(model, np.array([10.0]), pulse_p,
                                 linear_rate, vmax, transporter)
        axes[0].plot(t - 10, g, label=model, color=COLORS[model], lw=1.7)
        if model == "Transporter-state":
            axes[1].plot(t - 10, free, color=COLORS[model], lw=1.7)
    axes[0].set(xlim=(0, 8), xlabel="Time after pulse (ms)", ylabel="G / Kₘ")
    axes[0].set_title("A  Matched single-pulse clearance", loc="left", weight="bold")
    axes[0].grid(alpha=0.55); axes[0].legend(fontsize=7.5)
    axes[0].text(0.98, 0.88, "t₉₀ = 0.86 ms", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=8.5, weight="bold")
    axes[1].set(xlim=(0, 70), ylim=(0, 1.02), xlabel="Time after pulse (ms)",
                ylabel="Free transporter fraction")
    axes[1].set_title("B  Transporter availability", loc="left", weight="bold")
    axes[1].grid(alpha=0.55)
    t90 = single_pulse_t90("Transporter-state", p, transporter=transporter)
    recovery_t90 = transporter_recovery_t90(p, transporter)
    axes[1].text(0.98, 0.08, f"90% recovery = {recovery_t90:.1f} ms",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=8.5, weight="bold")
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
    axes[2].set(xlabel="Within-burst interval / single-pulse t₉₀",
                ylabel="Burst / tonic AUC")
    axes[2].set_title("C  Structural control", loc="left", weight="bold")
    axes[2].grid(alpha=0.55); axes[2].legend(fontsize=8)
    fig.suptitle("Matched calibration and timing response by model",
                 fontsize=12.5, weight="bold")
    save_figure(fig, out, "figure_2_calibration_and_structural_control")


def figure_main_results(summary: pd.DataFrame, out: Path) -> None:
    data = summary[summary["release_cv"] == 0.0]
    specs = [("peak_km", "Peak G / Kₘ"), ("auc_km_ms", "AUC (Kₘ·ms)"),
             ("clearance_ms", "Post-train clearance (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.8))
    width, x = 0.23, np.arange(len(MODELS))
    for ax, (metric, ylabel) in zip(axes, specs):
        selected = data[data["metric"] == metric]
        for i, pattern in enumerate(PATTERNS):
            means = [selected[(selected["model"] == m) &
                              (selected["pattern"] == pattern)]["mean"].iloc[0]
                     for m in MODELS]
            lows = [selected[(selected["model"] == m) &
                             (selected["pattern"] == pattern)]["ci95_low"].iloc[0]
                    for m in MODELS]
            highs = [selected[(selected["model"] == m) &
                              (selected["pattern"] == pattern)]["ci95_high"].iloc[0]
                     for m in MODELS]
            errors = np.vstack([np.array(means) - np.array(lows),
                                np.array(highs) - np.array(means)])
            ax.bar(x + (i - 1) * width, means, width, color=COLORS[pattern],
                   label="Fixed-count random" if pattern == "Poisson" else pattern,
                   yerr=errors, capsize=2, error_kw={"lw": 0.8, "capthick": 0.8})
        ax.set_xticks(x, ["Linear", "Static\nMichaelis-Menten", "Transporter-\nstate"])
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.55)
    for ax, title in zip(axes, ("A  Peak concentration", "B  Integrated exposure",
                               "C  Clearance time")):
        ax.set_title(title, loc="left", weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.88))
    fig.suptitle("Equal-amplitude timing patterns across matched clearance models",
                 y=0.99, fontsize=12.5, weight="bold")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.72, wspace=0.32)
    save_figure(fig, out, "figure_3_model_comparison")


def figure_amplitude_variability(ratios: pd.DataFrame, out: Path) -> None:
    selected = ratios[ratios["metric"].isin(["peak_km", "auc_km_ms"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), sharey=True)
    model_labels = ["Linear", "Static Michaelis-Menten", "Transporter-state"]
    y = np.arange(len(MODELS))
    styles = {0.0: ("#4C78A8", "o"), 0.15: ("#F58518", "s")}
    for ax, metric, title in zip(
        axes,
        ("peak_km", "auc_km_ms"),
        ("A  Peak ratio (expanded scale)", "B  AUC ratio"),
    ):
        part = selected[selected["metric"] == metric]
        for i, cv in enumerate((0.0, 0.15)):
            values = [part[(part["model"] == model) &
                           (part["release_cv"] == cv)]["ratio_of_means"].iloc[0]
                      for model in MODELS]
            color, marker = styles[cv]
            offsets = y + (-0.10 if i == 0 else 0.10)
            ax.scatter(values, offsets, s=52, color=color, marker=marker,
                       label=f"CV = {cv:.2f}", zorder=3)
            label_offset = 0.00025 if metric == "peak_km" else 0.006
            for value, ypos in zip(values, offsets):
                ax.text(value + label_offset, ypos, f"{value:.3f}", va="center",
                        fontsize=8, color=color)
        ax.axvline(1, color="#666666", ls="--", lw=1)
        ax.set_xlim((0.995, 1.005) if metric == "peak_km" else (0.97, 1.23))
        ax.set_yticks(y, model_labels)
        ax.set_xlabel("Burst / tonic ratio")
        ax.set_title(title, loc="left", weight="bold")
        ax.grid(axis="x", alpha=0.55)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.88))
    fig.suptitle("Amplitude variability as a separate robustness analysis",
                 y=0.99, fontsize=12, weight="bold")
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.70, wspace=0.20)
    save_figure(fig, out, "figure_4_amplitude_variability")


def figure_regime_maps(regime: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(10.8, 3.9), layout="constrained")
    grid_spec = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055])
    axes = [fig.add_subplot(grid_spec[0, 0])]
    axes.append(fig.add_subplot(grid_spec[0, 1], sharex=axes[0], sharey=axes[0]))
    axes.append(fig.add_subplot(grid_spec[0, 2], sharex=axes[0], sharey=axes[0]))
    color_axis = fig.add_subplot(grid_spec[0, 3])
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
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{panel}  {model}", loc="left", weight="bold")
        ax.set_xlabel("Burst interval / single-pulse t₉₀")
        if model == "Linear":
            ax.text(0.5, 0.5, "ratio = 1.000 throughout", transform=ax.transAxes,
                    ha="center", va="center", weight="bold", color="#333333")
    axes[0].set_ylabel("Pulse size / Kₘ")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    cbar = fig.colorbar(contour, cax=color_axis)
    cbar.set_label("Burst / tonic AUC")
    cbar.set_ticks(np.linspace(0.8, 1.4, 7))
    fig.suptitle("Burst-to-tonic AUC across dimensionless release regimes",
                 fontsize=12.2, weight="bold")
    save_figure(fig, out, "figure_5_dimensionless_regime_maps")


def figure_robustness(sensitivity: pd.DataFrame, continuum: pd.DataFrame,
                      correlations: pd.DataFrame, convergence: pd.DataFrame,
                      out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    ratios = sensitivity["auc_ratio_burst_tonic"].to_numpy()
    bins = np.geomspace(max(0.98, ratios.min()), ratios.max(), 30)
    axes[0].hist(ratios, bins=bins, color="#D95F02",
                 alpha=0.82, edgecolor="white")
    axes[0].axvline(1, color="#555555", ls="--", lw=1)
    median = float(np.median(ratios))
    axes[0].axvline(median, color="#1F2933", ls=":", lw=1.2)
    axes[0].set_xscale("log")
    axes[0].set_xticks([1, 2, 3, 5, 10], ["1", "2", "3", "5", "10"])
    axes[0].set(xlabel="Burst / tonic AUC", ylabel="Parameter sets")
    axes[0].set_title("A  Parameter sensitivity", loc="left", weight="bold")
    axes[0].grid(axis="y", alpha=0.55)
    axes[0].text(0.97, 0.94, f"median = {median:.3f}\nn = {len(ratios)}",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=8.5, weight="bold")
    part = continuum[continuum["model"] == "Transporter-state"]
    axes[1].scatter(part["isi_cv"], part["auc_km_ms"], s=12, alpha=0.40,
                    color="#7B3294", edgecolors="none")
    bins = pd.qcut(part["isi_cv"], 12, duplicates="drop")
    line = part.assign(bin=bins).groupby("bin", observed=True).agg(
        x=("isi_cv", "mean"), y=("auc_km_ms", "mean"))
    axes[1].plot(line["x"], line["y"], color="#3F007D", lw=2)
    rho = correlations[(correlations["model"] == "Transporter-state") &
                       (correlations["metric"] == "auc_km_ms")]["spearman_rho"].iloc[0]
    axes[1].text(0.04, 0.94, f"Spearman ρ = {rho:.3f}\nn = {len(part)}",
                 transform=axes[1].transAxes, ha="left", va="top", weight="bold")
    axes[1].set(xlabel="Inter-event interval CV", ylabel="AUC (Kₘ·ms)")
    axes[1].set_title("B  Timing continuum", loc="left", weight="bold")
    axes[1].grid(alpha=0.55)
    selected = convergence[np.isclose(convergence["dt_ms"], 0.02)]
    x = np.arange(len(MODELS))
    width = 0.32
    model_labels = ["Linear", "Static\nMichaelis-Menten", "Transporter-\nstate"]
    for i, pattern in enumerate(("Tonic", "Burst")):
        values = [selected[(selected["model"] == model) &
                           (selected["pattern"] == pattern)]["auc_error_pct"].iloc[0]
                  for model in MODELS]
        bars = axes[2].bar(x + (i - 0.5) * width, values, width,
                           color=COLORS[pattern], label=pattern)
        for bar, value in zip(bars, values):
            axes[2].text(bar.get_x() + bar.get_width() / 2,
                         max(value, 0) + 0.008, f"{value:.3f}", ha="center",
                         va="bottom", fontsize=7)
    axes[2].set_xticks(x, model_labels)
    axes[2].set_ylabel("AUC error vs 0.01 ms (%)")
    axes[2].set_title("C  Numerical convergence", loc="left", weight="bold")
    axes[2].grid(axis="y", alpha=0.55)
    axes[2].legend(loc="upper left", fontsize=7.5)
    axes[2].set_ylim(0, selected["auc_error_pct"].max() * 1.25)
    fig.suptitle("Sensitivity, timing continuum, and numerical convergence",
                 y=0.99, fontsize=12.2, weight="bold")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.77, wspace=0.34)
    save_figure(fig, out, "figure_6_robustness_and_convergence")
