# Temporal clustering and glutamate uptake

Reproducible simulations for **Temporal Clustering of Glutamate Release Increases Extracellular Exposure in Recovery-Limited Uptake Regimes**.

DOI: [10.5281/zenodo.22864139](https://doi.org/10.5281/zenodo.22864139)

The analysis compares linear, static Michaelis-Menten, and transporter-state clearance under matched single-pulse conditions. Tonic and burst protocols use the same event count, total release, and time window. The main analysis uses equal release amplitudes. A separate analysis uses CV = 0.15.

## Run the analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m glutamate_uptake
```

Results are written to `data/` and `figures/`. The random seed is recorded in `data/model_parameters.json`.

`config.py` defines parameters, `core.py` contains the models, `analysis.py` runs experiments, and `figures.py` builds the figures. `glutamate_uptake_simulation.py` remains as a compatible entry point.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT
