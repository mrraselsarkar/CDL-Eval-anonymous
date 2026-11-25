"""Robustness checks for the real-data study (responses to review).

On the same 300 test windows per dataset as run.py:
  - Chronos with its default decoding (top-k 50) and with top-k disabled
  - Chronos-Bolt scored at its native quantile levels (0.1 to 0.9), and its 90%
    coverage under three tail models (linear, Gaussian, clamped)
  - Sundial with 512, 2,880 and 48 context steps
  - the pretrained models with 48 context steps, the context of the small models
  - the small forecasters again, for unbiased ("fair") scores
For every sample-based forecaster we report V-statistic and unbiased CRPS and
energy score, the quantile loss at the nine Chronos-Bolt levels, and PIT coverage
of the central 80% and 90% intervals.

Usage:  python robust.py      writes results/robust.json
"""

import json
import os
import sys
import time

import numpy as np
import torch

import run

sim = run.sim
DEV = run.DEV
LEVELS = torch.tensor(run.BOLT_Q, dtype=torch.float32)
Z = {"0.05": -1.6448536, "0.95": 1.6448536, "0.1": -1.2815516, "0.9": 1.2815516}


def fair(x, y):
    S = x.shape[1]
    t1 = (x - y[:, None]).abs().mean(1)
    srt = x.sort(1).values
    w = (2 * torch.arange(1, S + 1, device=x.device) - S - 1).float()
    crps = t1 - (srt * w[None, :, None]).sum(1) / (S * (S - 1))
    e1 = (x - y[:, None]).norm(dim=2).mean(1)
    e2 = torch.cdist(x, x).sum((1, 2)) / (S * (S - 1))
    return float(crps.mean()), float((e1 - 0.5 * e2).mean())


def qloss(q, y):
    """Mean quantile (pinball) loss x2 over the nine levels; q (N, H, 9), y (N, H)."""
    lv = LEVELS.to(q.device)
    d = y[..., None] - q
    return float((2 * torch.maximum(lv * d, (lv - 1) * d)).mean())


def pit_cov(x, y, level, gen):
    below = (x < y[:, None]).sum(1).float()
    u = (below + torch.rand(below.shape, device=DEV, generator=gen)) / (x.shape[1] + 1)
    return float(((u > (1 - level) / 2) & (u < (1 + level) / 2)).float().mean())


PER_WINDOW = {}


def score_samples(xs, d, gen, key=None):
    x = (xs - d["m"][:, None]) / d["sc"][:, None]
    y = (d["fut"] - d["m"]) / d["sc"]
    met, _ = run.score(xs, d, gen)
    cf, ef = fair(x, y)
    if key is not None:                                   # per-window scores for block bootstrap
        PER_WINDOW[key + "|crps"] = sim.crps_samples(x, y).mean(1).cpu().numpy()
        PER_WINDOW[key + "|es"] = sim.energy_score(x, y).cpu().numpy()
    q = torch.quantile(x, LEVELS.to(DEV), dim=1).permute(1, 2, 0)            # (N, H, 9)
    met.update(crps_fair=cf, es_fair=ef, qloss=qloss(q, y),
               pit80=pit_cov(x, y, 0.8, gen), pit90=pit_cov(x, y, 0.9, gen))
    return met


def score_bolt(q_raw, d, gen):
    """Native scoring of Chronos-Bolt quantiles and 90% coverage under three tail models."""
    q = (torch.sort(q_raw, -1).values - d["m"][:, :, None]) / d["sc"][:, :, None]   # scaled, (N, H, 9)
    y = (d["fut"] - d["m"]) / d["sc"]
    cov80 = float(((y >= q[..., 0]) & (y <= q[..., -1])).float().mean())
    med, q10, q90 = q[..., 4], q[..., 0], q[..., -1]
    tails = {"linear": (q10 - (q[..., 1] - q10) / 0.1 * 0.05, q90 + (q90 - q[..., -2]) / 0.1 * 0.05),
             "gaussian": (med + (q10 - med) * Z["0.05"] / Z["0.1"], med + (q90 - med) * Z["0.95"] / Z["0.9"]),
             "clamped": (q10, q90)}
    out = dict(qloss=qloss(q, y), cov80_native=cov80)
    for name, (lo, hi) in tails.items():
        out[f"cov90_{name}"] = float(((y >= lo) & (y <= hi)).float().mean())
    return out


@torch.no_grad()
def chronos(pipe, ctx, **kw):
    res, b = [], 5                 # larger batches spill GPU memory on a 16 GB card and run slower
    i = 0
    while i < ctx.shape[0]:
        try:
            res.append(pipe.predict(ctx[i:i + b].cpu(), prediction_length=run.H, num_samples=run.S, **kw).to(DEV))
            i += b
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            b = max(1, b // 2)
    return torch.cat(res)


@torch.no_grad()
def sundial(model, ctx):
    res = []
    for i in range(0, ctx.shape[0], 25):
        c = ctx[i:i + 25]
        mu = c.mean(1, keepdim=True)
        sd = c.std(1, keepdim=True, unbiased=False).clamp_min(1e-2)
        p = model(input_ids=(c - mu) / sd, revin=False, num_samples=run.S, max_output_length=run.H,
                  use_cache=False, return_dict=True).logits
        res.append(p * sd[:, None] + mu[:, None])
    return torch.cat(res)


def main():
    from chronos import BaseChronosPipeline
    from transformers import AutoModelForCausalLM
    sim.ABAR = sim.cosine_abar(sim.CFG["k_train"])
    run.LF = 2880                                        # longest context; shorter ones are its suffixes
    out_path = os.path.join(run.OUT, "robust.json")
    rows = []
    for ds in run.DATASETS:
        d = run.prepare(ds)
        gen = torch.Generator(device=DEV).manual_seed(11)
        ctx = d["ctx"]

        PER_WINDOW[f"{ds}|sid"] = d["sid"]
        PER_WINDOW[f"{ds}|origin_block"] = d["block"]

        def rec(label, xs=None, extra=None, t=0.0):
            met = score_samples(xs, d, gen, key=f"{ds}|{label}") if xs is not None else {}
            if extra:
                met.update(extra)
            met.update(dataset=ds, model=label, time_s=t)
            rows.append(met)
            print(f"[robust] {ds:8s} {label:22s} " + " ".join(
                f"{k}={v:.3f}" for k, v in met.items() if isinstance(v, float) and k != "time_s"), flush=True)

        pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map=str(DEV),
                                                   torch_dtype=torch.float32)
        for label, c, kw in [("chronos_512_default", ctx[:, -512:], {}),
                             ("chronos_512_full", ctx[:, -512:], {"top_k": 4096}),
                             ("chronos_48_default", ctx[:, -48:], {})]:
            torch.manual_seed(0)
            t0 = time.perf_counter()
            xs = chronos(pipe, c, **kw)
            rec(label, xs, t=time.perf_counter() - t0)
        del pipe
        torch.cuda.empty_cache()

        pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=str(DEV),
                                                   torch_dtype=torch.float32)
        for label, c in [("bolt_512", ctx[:, -512:]), ("bolt_48", ctx[:, -48:])]:
            q = torch.cat([pipe.predict_quantiles(c[i:i + 50].cpu(), prediction_length=run.H,
                                                  quantile_levels=list(run.BOLT_Q))[0].to(DEV)
                           for i in range(0, c.shape[0], 50)])
            rec(label, run.bolt_samples(torch.sort(q, -1).values, gen, False), extra=score_bolt(q, d, gen))
        del pipe
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True).to(DEV).eval()
        for label, L in [("sundial_512", 512), ("sundial_2880", 2880), ("sundial_48", 48)]:
            torch.manual_seed(0)
            t0 = time.perf_counter()
            xs = sundial(model, ctx[:, -L:])
            rec(label, xs, t=time.perf_counter() - t0)
        del model
        torch.cuda.empty_cache()

        for seed in run.SEEDS:
            for name in run.SMALL:
                xs, el = run.run_small(name, d, seed, ds)
                rec(f"{name}_s{seed}", xs, t=el)
                if name == "masked":
                    xs, el = run.run_small(name, d, seed, ds, order="random")
                    rec(f"masked_rand_s{seed}", xs, t=el)
                torch.cuda.empty_cache()
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=1)
        np.savez(os.path.join(run.OUT, "robust_perwindow.npz"), **PER_WINDOW)
    print("done", flush=True)


if __name__ == "__main__":
    main()
