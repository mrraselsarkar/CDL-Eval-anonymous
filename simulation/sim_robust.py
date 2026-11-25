"""Robustness checks for the simulation study (responses to review).

1. Longer training and another parameterization for full-window diffusion:
   noise prediction with 3x and 10x the training steps, velocity (v) prediction,
   and flow matching with 10x the steps for comparison.
2. A parameter-matched autoregressive model (hidden width 640, about 0.84 M
   parameters, as many as the full-window models).
3. Unbiased ("fair") CRPS and energy-score estimators for the main forecasters,
   next to the V-statistics used in the main text.

Usage:  python sim_robust.py      writes results_robust/robust.json and results_robust/fair.json
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sim
from sim import CFG, DEV, mlp, temb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_robust")
MODEL_DIR = os.path.join(OUT, "models")


class FullDiffV(sim.FullDiff):
    """Full-window diffusion with velocity (v) prediction; same networks as FullDiff."""

    def loss(self, w):
        zn = sim.normalize_window(w)
        c = self.enc(zn[:, :sim.L])
        y0 = zn[:, sim.L:]
        k = torch.randint(1, CFG["k_train"] + 1, (y0.shape[0],), device=DEV)
        e = torch.randn_like(y0)
        a = sim.ABAR[k][:, None]
        yk = a.sqrt() * y0 + (1 - a).sqrt() * e
        v = a.sqrt() * e - (1 - a).sqrt() * y0
        return F.mse_loss(self.eps(yk, k, c), v)          # the network output is read as v

    @torch.no_grad()
    def sample(self, ctx_n, S, steps=None, **kw):
        steps = steps or CFG["k_sample"]
        c = self.enc(ctx_n).repeat_interleave(S, 0)

        def eps_fn(yk, k):
            a = sim.ABAR[k][:, None]
            v = self.eps(yk, k, c)
            return (1 - a).sqrt() * yk + a.sqrt() * v       # convert v to the noise estimate

        y = sim.ddim_sample(eps_fn, (c.shape[0], sim.H), steps)
        return y.view(-1, S, sim.H)


class ARWide(sim.ARGauss):
    def __init__(self):
        nn.Module.__init__(self)
        self.net = mlp(sim.L, 640, 2, 4)


sim.CLASSES.update(fulldiff_v=FullDiffV, ar_wide=ARWide)

# (label, class name, training steps, datasets)
RUNS = [("fulldiff_9k", "fulldiff", 9000, ["bimodal", "lorenz"]),
        ("fulldiff_30k", "fulldiff", 30000, ["bimodal", "lorenz"]),
        ("fulldiff_v_9k", "fulldiff_v", 9000, ["bimodal", "lorenz"]),
        ("flow_30k", "flow", 30000, ["bimodal", "lorenz"]),
        ("ar_wide", "ar_wide", 3000, ["seasonal", "bimodal", "lorenz"])]


def fair_scores(xs, y):
    """Unbiased CRPS and energy score; xs (N, S, H), y (N, H)."""
    S = xs.shape[1]
    t1 = (xs - y[:, None]).abs().mean(1)
    srt = xs.sort(1).values
    w = (2 * torch.arange(1, S + 1, device=xs.device) - S - 1).float()
    crps = t1 - (srt * w[None, :, None]).sum(1) / (S * (S - 1))
    e1 = (xs - y[:, None]).norm(dim=2).mean(1)
    e2 = torch.cdist(xs, xs).sum((1, 2)) / (S * (S - 1))
    return float(crps.mean()), float((e1 - 0.5 * e2).mean())


def get_model(ds, label, cls, steps, seed, data):
    path = os.path.join(MODEL_DIR, f"{ds}_{label}_s{seed}.pt")
    m = sim.CLASSES[cls]().to(DEV)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
        m.eval()
        return m, 0.0
    t0 = time.perf_counter()
    m = sim.train(cls, data, seed, steps)
    torch.save(m.state_dict(), path)
    return m, time.perf_counter() - t0


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    sim.ABAR = sim.cosine_abar(CFG["k_train"])
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = []
    data = {ds: sim.make_dataset(ds) for ds in sim.DATASETS}
    for label, cls, steps, dss in RUNS:
        for ds in dss:
            train_w, test_w = data[ds]
            for seed in CFG["seeds"]:
                m, tt = get_model(ds, label, cls, steps, seed, train_w)
                met, xs = sim.evaluate(m, test_w, CFG["n_samples"])
                cf, ef = fair_scores(xs, test_w[:, sim.L:])
                met.update(dataset=ds, model=label, seed=seed, train_steps=steps, train_s=tt,
                           params=sum(p.numel() for p in m.parameters()), crps_fair=cf, es_fair=ef)
                for k in ("mse_h", "crps_h"):
                    met.pop(k)
                rows.append(met)
                print(f"[robust] {ds:9s} {label:14s} seed={seed} mse={met['mse']:.3f} crps={met['crps']:.3f} "
                      f"es={met['es']:.3f} cov90={met['cov90']:.2f} train={tt:.0f}s", flush=True)
                del m
                torch.cuda.empty_cache()
            with open(os.path.join(OUT, "robust.json"), "w") as f:
                json.dump(rows, f, indent=1)

    # fair estimators for the main forecasters (same samples as the main study)
    fair = []
    for ds in sim.DATASETS:
        train_w, test_w = data[ds]
        for seed in CFG["seeds"]:
            for name in sim.MODELS:
                m, _ = sim.get_model(ds, name, seed, train_w, CFG["n_train"], CFG["train_steps"])
                runs = [(name, {})] + ([("masked_rand", {"order": "random"})] if name == "masked" else [])
                for lab, kw in runs:
                    met, xs = sim.evaluate(m, test_w, CFG["n_samples"], **kw)
                    cf, ef = fair_scores(xs, test_w[:, sim.L:])
                    fair.append(dict(dataset=ds, model=lab, seed=seed, crps=met["crps"], es=met["es"],
                                     crps_fair=cf, es_fair=ef))
                del m
                torch.cuda.empty_cache()
        print(f"[fair] {ds} done", flush=True)
        with open(os.path.join(OUT, "fair.json"), "w") as f:
            json.dump(fair, f, indent=1)


if __name__ == "__main__":
    main()
