import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    burst_tonic_ratios,
    run_burstiness_continuum,
    run_convergence,
    run_equal_count_experiment,
    run_global_sensitivity,
    run_regime_maps,
    summarize,
    write_key_results,
)
from .config import SEED, Protocol, TransporterParameters
from .core import calibrate_models
from .figures import (
    configure_plotting,
    figure_amplitude_variability,
    figure_calibration,
    figure_inputs_and_traces,
    figure_main_results,
    figure_regime_maps,
    figure_robustness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run glutamate release and uptake simulations."
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--continuum-trains", type=int, default=600)
    parser.add_argument("--sensitivity-sets", type=int, default=800)
    parser.add_argument("--regime-grid", type=int, default=26)
    return parser


def run_pipeline(args: argparse.Namespace) -> Path:
    root = args.output_dir.resolve()
    data_dir, figures_dir = root / "data", root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    protocol = Protocol()
    transporter = TransporterParameters()
    rng = np.random.default_rng(args.seed)
    linear_rate, vmax, calibration = calibrate_models(protocol, transporter)

    cv0 = run_equal_count_experiment(
        protocol, transporter, linear_rate, vmax, args.runs, 0.0, rng
    )
    cv15 = run_equal_count_experiment(
        protocol, transporter, linear_rate, vmax, args.runs, 0.15, rng
    )
    equal_count = pd.concat([cv0, cv15], ignore_index=True)
    summary = summarize(equal_count)
    ratios = burst_tonic_ratios(equal_count)
    reference_t90 = float(calibration.loc[
        calibration["quantity"] == "single_pulse_t90_ms", "value"
    ].iloc[0])
    regime = run_regime_maps(
        transporter, linear_rate, vmax, reference_t90,
        args.regime_grid, args.regime_grid,
    )
    continuum, correlations = run_burstiness_continuum(
        protocol, transporter, linear_rate, vmax, args.continuum_trains, rng
    )
    sensitivity = run_global_sensitivity(protocol, args.sensitivity_sets, rng)
    convergence = run_convergence(protocol, transporter, linear_rate, vmax)

    outputs = {
        "equal_count_runs.csv": equal_count,
        "equal_count_summary.csv": summary,
        "burst_tonic_ratios.csv": ratios,
        "calibration_targets.csv": calibration,
        "dimensionless_regime_map.csv": regime,
        "burstiness_continuum.csv": continuum,
        "burstiness_correlations.csv": correlations,
        "global_sensitivity.csv": sensitivity,
        "numerical_convergence.csv": convergence,
    }
    for name, frame in outputs.items():
        frame.to_csv(data_dir / name, index=False)

    parameters = {
        "protocol": asdict(protocol),
        "transporter": asdict(transporter),
        "matched_linear_rate_per_ms": linear_rate,
        "matched_mm_vmax_km_per_ms": vmax,
        "seed": args.seed,
    }
    (data_dir / "model_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    write_key_results(
        ratios, sensitivity, correlations, convergence, calibration,
        data_dir / "key_results.json",
    )

    figure_inputs_and_traces(protocol, transporter, linear_rate, vmax, figures_dir)
    figure_calibration(protocol, transporter, linear_rate, vmax, figures_dir)
    figure_main_results(summary, figures_dir)
    figure_amplitude_variability(ratios, figures_dir)
    figure_regime_maps(regime, figures_dir)
    figure_robustness(sensitivity, continuum, correlations, convergence, figures_dir)
    return root


def main() -> None:
    output = run_pipeline(build_parser().parse_args())
    print(f"Wrote revised analysis to {output}")


if __name__ == "__main__":
    main()
