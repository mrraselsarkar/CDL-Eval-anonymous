# Simulation study: autoregressive and diffusion forecasters

Controlled simulation of Section S3 of the Supplementary Material: small forecasters of each generation type, trained
on synthetic data whose true forecast distribution is known. Python with `torch`, `numpy` and `matplotlib`; a CUDA GPU
is optional.

## Runs

```bash
python sim.py --exp main                 # 3 datasets x 6 forecasters x 3 seeds (Table S24)
python sim.py --exp steps                # sampling-step sweep, reuses the trained models (Fig. S3a,b)
python sim.py --exp data                 # training-set-size sweep on the Lorenz data
python calib.py                          # PIT histograms for the calibration figure (Fig. S2)
python sim_robust.py                     # longer training runs reused by the budget study
python budget.py --datasets bimodal      # every forecaster at 1k, 9k and 30k training steps (Fig. S3d-f)
python budget.py --datasets lorenz
python plots.py                          # tables and figures into ../outputs/
```

`plots.py` writes Table S24 (`sim_table.tex`), Fig. S2 (`sim_calibration.pdf`) and Fig. S3
(`sim_cost_data.pdf`: cost, training-set size and training budget, from the step sweep, the data sweep and the budget
runs). It also writes `sim_steps_table.tex` and `sim_budget_table.tex` with the values behind that figure; the
Supplementary Material leaves these two tables out.

`python sim.py --exp smoke` runs the whole pipeline with tiny settings as a check. `budget.py` runs on the CPU (a few
hours with four threads). Finished runs are skipped when a script is started again. Run one GPU job at a time: two
CUDA processes on one GPU slowed training by more than an order of magnitude.

`plots.py` prints the numbers quoted in the text. Interval coverage in Table S24 uses sample quantiles, which are
biased low with 50 samples (about 86.5% expected for a calibrated forecaster at the nominal 90% level);
Fig. S2 uses the randomized probability integral transform, which is unbiased.

## Forecasters

| Key           | Forecaster                          | Generation                                      |
|---------------|-------------------------------------|-------------------------------------------------|
| `ar`          | Autoregressive, Gaussian head       | one value at a time, recursive sampling         |
| `direct`      | Direct multi-step                   | whole horizon in one pass, independent steps    |
| `ardiff`      | Autoregressive + diffusion head     | recursive; a diffusion model samples each step  |
| `fulldiff`    | Full-window diffusion               | whole horizon denoised jointly                  |
| `flow`        | Full-window flow matching           | whole horizon transported jointly (Euler steps) |
| `masked`      | Masked diffusion over value tokens  | tokens unmasked by confidence over rounds       |
| `masked_rand` | Masked diffusion, random order      | same model, tokens unmasked in random order     |

The datasets are synthetic (seasonal, regime switch with a bimodal future, noisy Lorenz-63) and generated with fixed
seeds inside `sim.py`. Results are written to `results/`, `results_budget/` and `results_robust/` (stored in the
repository), and trained weights to the `models/` folder inside each (not stored; a run retrains missing weights).
