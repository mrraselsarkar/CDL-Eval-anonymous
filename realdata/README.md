# Benchmark on real data

Seven pretrained foundation models applied without fine-tuning (Chronos small and base, Chronos-Bolt small and base,
Chronos-2, TimesFM 2.5, Toto, Sundial, Aurora), a probabilistic seasonal naive baseline, small forecasters trained on
each dataset, and further baselines (exponential smoothing, DeepAR, Lag-Llama), on three held-out datasets and on
post-dated data (Sections V and VI of the paper, Sections S1 and S2 of the Supplementary Material). Every script runs
from this folder; `../scripts/reproduce.py` runs them in order with the command lines of the paper.

## Scripts

| Script | Role |
| --- | --- |
| **Data and windows** | |
| `fresh_data.py` | Downloads and loads AEMO 2015–2019 and ERCOT 2026 (EIA-930); checks the Monash Australian series against AEMO and validates the ERCOT zones (`results/rev2/data_checks.json`) |
| `run.py` | Held-out datasets, test windows (`run.prepare`), small forecasters trained on each dataset; first-round runner |
| `cluster_stats.py` | Calendar weeks of the forecast origins (`results/window_weeks.npz`) and the week-cluster bootstrap used by the report scripts |
| **Benchmark** | |
| `bench.py` | Benchmark protocol: window sets of every dataset, runners of all foundation models and the seasonal naive baseline, scores per window (quantile loss, coverage, CRPS, energy and variogram scores, shuffle test, PIT) |
| `rev2.py` | Format of the stored full forecasts (`results/rev2/<dataset>__H<H>__<model>__s<seed>[__<setting>].npz`) |
| `bench_extra.py` | Added baselines: exponential smoothing, DeepAR, Lag-Llama, small forecasters with 512 context steps |
| `revision_runs.py` | Chunked rollout of Chronos-2, Schaake shuffle and Gaussian copula, spread correction |
| `run_fm.py`, `robust.py`, `aurora_context.py` | First-round runs: quantile models turned into sample paths, robustness runs (context length, decoding, fair scores), Aurora with 96 to 2,880 context steps |
| **Checks** | |
| `shuffle_check.py` | Calibration of the shuffle test and of the rank dependence measures on synthetic forecasts with known dependence |
| `sampler_gauss.py` | The released flow update of Sundial and Aurora against the flow ODE for a Gaussian target |
| `flow_determinism.py` | Sundial and Aurora are fixed by their initial noise |
| `sundial_official.py` | Sundial through its released `generate()` routine (environment `requirements/sundial-generate.txt`) |
| **Reports (CPU)** | |
| `bench_report.py` | Table IV, Fig. 3, Fig. 4, Fig. 5, and the benchmark tables of Section S1 |
| `rank_stats.py` | Table V, Friedman, Iman–Davenport and Nemenyi tests, Wilcoxon tests with Holm's correction, ranks without leaked pairs |
| `revision_report.py` | Fig. 6, Fig. 7, and the tables and figures of Section S2 (needs the full forecasts) |
| `extra_report.py` | Tables of the added baselines (needs the full forecasts) |
| `robust_report.py` | Tables of the first-round robustness and context-length runs |

The report scripts write LaTeX tables to `../outputs/tables/` and figures to `../outputs/figures/`.

## Examples

```bash
python bench.py --data heldout --horizon 24 --models naive bolt chronos2 timesfm sundial aurora --tag main
python bench.py --data aemo --horizon 24 --models toto --tag toto                  # in the Toto environment
python bench.py --data heldout --horizon 720 --n_test 150 --seeds 0 --models chronos2 timesfm --tag long
python bench.py --data ercot2026 --horizon 24 --models sundial aurora --flow_steps 200 --store rev2 --tag test
python bench_extra.py --cpu --models ets --data ercot traffic ausel --horizon 24
python bench_report.py
```

`bench.py` options: `--data heldout | aemo | aemo_mid | ausel_season | ercot2026 | heldout_calib`, `--horizon`,
`--n_test`, `--seeds`, `--models` (`naive chronos chronos_base bolt bolt_base chronos2 timesfm toto sundial aurora`),
`--tag` (one result file per environment, so that runs do not collide), `--store rev2` (keep the full forecasts),
`--context` (512 by default; 2,048 and 4,096 in the paper), `--flow_steps` and `--flow_sampler euler | ode | sde` (the
flow heads of Sundial and Aurora), and `--max_windows`, `--paths` for quick tests. Each run merges its rows into
`results/bench_<tag>.json` and `results/bench_<tag>_perwindow.npz`; the report scripts read every file, and a later
file in name order overrides rows of the same dataset, horizon, model and seed.

Chronos samples in batches of 5 windows (100 paths each) to fit in 16 GB of GPU memory; `robust.py` halves the batch
on an out-of-memory error. Run one GPU job at a time.

## Stored results

`results/` holds the summary rows and per-window scores of every run reported in the paper, and `results/rev2/` the
summaries of the runs with stored forecasts and `report.json` (every number of the Section S2 tables and figures). The
full forecasts (`results/rev2/*__*.npz`) and the trained DeepAR weights are in the results archive (see the main
README).
