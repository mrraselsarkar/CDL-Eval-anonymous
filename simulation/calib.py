"""Calibration of the six forecasters: probability integral transform (PIT).

Loads the trained models of the main experiment (results/models), draws the same
50 samples per test window as sim.py, and stores for every dataset, model and
seed a histogram of randomized PIT values over all test windows and forecast
steps. From the histogram, plots.py derives reliability curves (nominal against
empirical coverage of central intervals at every level).

Usage:
  python calib.py            writes results/pit.json
"""

import json
import os

import numpy as np
import torch

import sim

N_BINS = 100


@torch.no_grad()
def pit_values(model, test, S, **kw):
    ctx, y = test[:, :sim.L], test[:, sim.L:]
    mu, sd = sim.inorm(ctx)
    torch.manual_seed(1234)                       # same draws as sim.evaluate
    xs = model.sample((ctx - mu) / sd, S, **kw)
    xs = xs * sd[:, None] + mu[:, None]
    below = (xs < y[:, None]).sum(1).float()      # rank of the observation among the samples
    u = torch.rand_like(below)
    return ((below + u) / (S + 1)).flatten().cpu().numpy()


def main():
    sim.ABAR = sim.cosine_abar(sim.CFG["k_train"])
    torch.backends.cuda.matmul.allow_tf32 = True
    out = []
    for ds in sim.DATASETS:
        train_w, test_w = sim.make_dataset(ds)
        for seed in sim.CFG["seeds"]:
            for name in sim.MODELS:
                m, _ = sim.get_model(ds, name, seed, train_w, sim.CFG["n_train"], sim.CFG["train_steps"])
                runs = [(name, {})] + ([("masked_rand", {"order": "random"})] if name == "masked" else [])
                for label, kw in runs:
                    p = pit_values(m, test_w, sim.CFG["n_samples"], **kw)
                    hist = np.histogram(p, bins=N_BINS, range=(0, 1))[0]
                    out.append(dict(dataset=ds, model=label, seed=seed, hist=hist.tolist()))
                    print(f"[pit] {ds:9s} {label:12s} seed={seed} "
                          f"cov90={np.mean((p > 0.05) & (p < 0.95)):.3f}", flush=True)
                del m
                torch.cuda.empty_cache()
    with open(os.path.join(sim.OUT, "pit.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
