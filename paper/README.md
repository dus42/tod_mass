# Reproducing the paper

All commands are run from this folder. The scripts import `pipeline_tod.py` from the repository root, which `run_all.sh` puts on the path; when running a script on its own, set `PYTHONPATH=..`.

```bash
./run_all.sh          # every step, from the raw data
./run_all.sh train    # resume from a step, e.g. from the model training
```

## Data flow

```
data/24challenge_data/ ─1─► data/processed_tod/ ─2─► data/traj_tod_full.parquet ─3─► data/todmass_full.csv ─4─► ../models/
  raw trajectories          15 s resampling,          descent features and          + flight list, airport        four LightGBM
  (PRC Data Challenge)      flight phases,            reference TOD mass,           positions and MTOW:           models
                            propagated mass           one row per flight            the modelling frame
```

1. `processing_tod.py` resamples each trajectory, labels its flight phases and propagates the disclosed takeoff weight with the OpenAP fuel flow model.
2. `features_tod_full.py` aggregates the descent features and reads the reference mass at TOD, one row per flight.
3. `all_features_tod.py` joins these with the flight list, the airport positions and the MTOW, keeping only the columns the models and experiments use.
4. `train_models.py` trains the four models. The experiments also start from `data/todmass_full.csv`, and those on incomplete descents also read the raw trajectories.

## Data

Trajectories are not redistributed. Place the inputs in `data/`:

| Path | Source |
|---|---|
| `data/24challenge_data/*.parquet` | raw trajectories of the [PRC Data Challenge 2024](https://ansperformance.eu/study/data-challenge/) |
| `data/flight_list.csv` | flight list of the PRC Data Challenge 2024 |
| `data/todmass_full.csv` | TOD features and reference masses, from Zenodo: **[DOI to be added]** |

With `data/todmass_full.csv` from Zenodo, the first three steps can be skipped (`./run_all.sh train`). The experiments on incomplete descents also need the raw trajectories, and the pipeline check also needs `data/traj_tod_full.parquet` from the `features` step.

## Steps and outputs

| Step | Script | Produces |
|---|---|---|
| `processing` | `processing_tod.py` | `data/processed_tod/`: resampled trajectories with flight phases and propagated mass |
| `features` | `features_tod_full.py` | `data/traj_tod_full.parquet`: descent features and TOD mass |
| `frame` | `all_features_tod.py` | `data/todmass_full.csv`: modelling frame |
| `train` | `train_models.py` | `../models/`, Table 2, per-type results, test-set predictions |
| `scatter` | `plot_test_scatter.py` | Figure 2 |
| `pertype_fig` | `plot_pertype_mape.py` | Figure 3 |
| `pertype_tab` | `make_pertype_table.py` | Table 3 |
| `unseen_mass`, `unseen_ratio` | `experiment_unseen_types.py` | Tables 4 and 5 |
| `missing`, `repeats` | `experiment_missing_descent.py` | Table 6, mask library |
| `masks_fig` | `plot_mask_coverage.py` | Figure 4 |
| `numbers` | `paper_numbers.py` | in-text statistics of Sections 2.4 and 4.2 |
| `verify` | `verify_pipeline.py` | pipeline check of Appendix 1 |

Small results are kept in `results/` and the figures are written to `latex/figures/`. Tables 7 and 8 come from `../estimate_tod_from_opensky.py` with the command given in Appendix 1, which needs OpenSky Trino access.

## Building the paper

```bash
cd latex
latexmk -pdf main.tex
```
