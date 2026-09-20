# Temporal clustering and glutamate uptake

This repository contains the reproducible simulations for **Temporal Clustering of Glutamate Release Increases Extracellular Exposure in Recovery-Limited Uptake Regimes**.

The revision compares three clearance structures under matched single-pulse conditions:

1. a linear negative control;
2. a static Michaelis-Menten sink; and
3. a reduced transporter-state model with binding, translocation, and recovery.

The reference protocol contains 20 equal-amplitude release events. Tonic and burst trains have the same event count, total release, first-event time, and last-event time. The primary analysis uses coefficient of variation (CV) = 0. A separate CV = 0.15 analysis tests amplitude variability.

## Main result

At the calibrated 20 ms within-burst interval, the burst-to-tonic AUC ratio is 1.000 for linear clearance, 1.000 for static Michaelis-Menten clearance, and 1.199 for the transporter-state model. Across 800 broad transporter-state parameter sets, 96.4% produce an AUC ratio above 1 and 83.2% produce a ratio above 1.05. The dimensionless maps also show regimes where clustering has little effect or lowers AUC, so the result is presented as conditional on uptake structure and timescale.

## Run the analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python glutamate_uptake_simulation.py
```

The script writes source data to `data/` and publication-ready PNG and SVG figures to `figures/`. All random analyses use a fixed seed recorded in `data/model_parameters.json`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Reproducibility notes

- Python 3.11 or newer is recommended.
- The integration step is 0.02 ms in the reference analysis.
- Numerical convergence is checked against a 0.01 ms calculation.
- Model parameters, calibration targets, raw runs, summaries, sensitivity results, and dimensionless maps are included as machine-readable files.

## License

The code is released under the MIT License. Data and figures may be reused with attribution to the associated release and archival DOI.
