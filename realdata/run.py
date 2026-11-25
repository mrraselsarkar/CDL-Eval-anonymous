"""Real-data study: pretrained and small forecasters on three hourly energy and traffic datasets.

Datasets (autogluon/chronos_datasets, held out from the training corpus of Chronos):
  ercot    hourly electricity load, 8 regions of Texas
  traffic  hourly road occupancy, 862 sensors in San Francisco
  ausel    half-hourly electricity demand, 5 Australian states

Forecasters
  pretrained, zero-shot (512 context steps):
    chronos       Chronos-T5 small: autoregressive over value tokens, sampled
    bolt_ind      Chronos-Bolt small: direct quantile head; paths sampled independently per step
    bolt_com      Chronos-Bolt small: the same quantiles, one quantile level per path
    sundial       Sundial base: autoregressive over patches with a flow-matching head
  small, trained on each dataset (48 context steps, the six forecasters of ../simulation/sim.py)

Every window is scored after scaling by the mean and standard deviation of its last
week of context, so scores are comparable across series. Horizon: 24 steps.

Usage:  python run.py            writes results/real.json, results/pit_real.json, results/examples_real.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "simulation"))
import sim  # noqa: E402

DEV = sim.DEV
H, LS, LF = 24, sim.L, 512            # horizon, small-model context, pretrained-model context
N_TEST, S, SEEDS = 300, 100, [0, 1, 2]
OUT = os.path.join(HERE, "results")
MODEL_DIR = os.path.join(OUT, "models")
DATASETS = {"ercot": ("ercot.parquet", 24, 168), "traffic": ("monash_traffic.parquet", 24, 168),
            "ausel": ("monash_australian_electricity.parquet", 48, 336)}     # file, season, scale window
SMALL = ["ar", "direct", "ardiff", "fulldiff", "flow", "masked"]
BOLT_Q = np.linspace(0.1, 0.9, 9)


# =====================================================================
# Data
# =====================================================================

def load_series(fname):
    df = pd.read_parquet(os.path.join(HERE, "data", fname))
    out = []
    for t in df["target"]:
        s = pd.Series(np.asarray(t, dtype=float)).interpolate(limit_direction="both").to_numpy()
        out.append(s)
    return out


def prepare(name):
    fname, season, scale_win = DATASETS[name]
    series = load_series(fname)
    rng = np.random.default_rng(0)
    z, train, test, stats = [], [], [], []
    for i, s in enumerate(series):
        n = len(s)
        tr_end, te_start = int(0.7 * n), int(0.8 * n)
        mu, sd = s[:tr_end].mean(), s[:tr_end].std() + 1e-8
        stats.append((mu, sd))
        zs = (s - mu) / sd                                          # per-series scaling with training statistics
        z.append(zs)
        train.append((i, tr_end))
        test.append((i, max(te_start, LF), n - H))
    # training windows of length LS + H from the first 70% of each series
    lens = np.array([t[1] - (LS + H) for t in train], dtype=float)
    pick = rng.choice(len(train), size=sim.CFG["n_train"], p=lens / lens.sum())
    tw = np.stack([z[i][(st := rng.integers(0, train[i][1] - (LS + H))):st + LS + H] for i in pick])
    # test windows: random series and forecast origin in the last 20% of each series
    tl = np.array([t[2] - t[1] for t in test], dtype=float)
    pick = rng.choice(len(test), size=N_TEST, p=tl / tl.sum())
    ctx, fut, sid, orig = [], [], [], []
    for i in pick:
        o = rng.integers(test[i][1], test[i][2])
        ctx.append(z[i][o - LF:o])
        fut.append(z[i][o:o + H])
        sid.append(i)
        orig.append(o)
    ctx, fut = np.stack(ctx), np.stack(fut)
    # per-window scale for scoring: last week of context
    m = ctx[:, -scale_win:].mean(1, keepdims=True)
    sc = ctx[:, -scale_win:].std(1, keepdims=True) + 1e-3
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    # origin: index of the first forecast step in the series; mu_s, sd_s: the series standardization of each window
    # (original units = standardized * sd_s + mu_s). These keys were added in the second revision; the windows and
    # all other keys are unchanged.
    return dict(train=to(tw), ctx=to(ctx), fut=to(fut), m=to(m), sc=to(sc), season=season,
                n_series=len(series), sid=np.array(sid), block=np.array(orig) // scale_win,
                origin=np.array(orig, dtype=np.int64), scale_win=scale_win,
                mu_s=np.array([stats[i][0] for i in sid]), sd_s=np.array([stats[i][1] for i in sid]))


# =====================================================================
# Scoring
# =====================================================================

def variogram_score(x, y, p=0.5, chunk=100):
    out = []
    for i in range(0, x.shape[0], chunk):
        xs, ys = x[i:i + chunk], y[i:i + chunk]
        vy = (ys[:, :, None] - ys[:, None, :]).abs() ** p
        vx = ((xs[:, :, :, None] - xs[:, :, None, :]).abs() ** p).mean(1)
        out.append(((vy - vx) ** 2).mean((1, 2)))
    return torch.cat(out)


def score(xs, d, gen):
    """xs: (N, S, H) samples on the per-series scale. Returns metrics on the per-window scale."""
    x = (xs - d["m"][:, None]) / d["sc"][:, None]
    y = (d["fut"] - d["m"]) / d["sc"]
    perm = torch.argsort(torch.rand(x.shape, device=DEV, generator=gen), dim=1)
    xsh = x.gather(1, perm)                                  # shuffle samples independently per step
    q = torch.quantile(x, torch.tensor([0.05, 0.95], device=DEV), dim=1)
    below = (x < y[:, None]).sum(1).float()
    pit = ((below + torch.rand(below.shape, device=DEV, generator=gen)) / (x.shape[1] + 1)).flatten()
    met = dict(mse=float(((x.mean(1) - y) ** 2).mean()),
               crps=float(sim.crps_samples(x, y).mean()),
               es=float(sim.energy_score(x, y).mean()),
               vs=float(variogram_score(x, y).mean()),
               es_shuf=float(sim.energy_score(xsh, y).mean()),
               vs_shuf=float(variogram_score(xsh, y).mean()),
               cov90=float(((y >= q[0]) & (y <= q[1])).float().mean()))
    hist = np.histogram(pit.cpu().numpy(), bins=100, range=(0, 1))[0].tolist()
    return met, hist


# =====================================================================
# Forecasters
# =====================================================================

@torch.no_grad()
def run_small(name, d, seed, ds, **kw):
    path = os.path.join(MODEL_DIR, f"{ds}_{name}_s{seed}.pt")
    if os.path.exists(path):
        model = sim.CLASSES[name]().to(DEV)
        model.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
        model.eval()
    else:
        with torch.enable_grad():
            model = sim.train(name, d["train"], seed, sim.CFG["train_steps"])
        torch.save(model.state_dict(), path)
    ctx = d["ctx"][:, -LS:]
    mu, sd = sim.inorm(ctx)
    torch.manual_seed(1234 + seed)
    t0 = time.perf_counter()
    xs = torch.cat([model.sample((ctx[i:i + 100] - mu[i:i + 100]) / sd[i:i + 100], S, **kw)
                    for i in range(0, ctx.shape[0], 100)])
    el = time.perf_counter() - t0
    return xs * sd[:, None] + mu[:, None], el


def bolt_samples(q, gen, comonotone):
    """Sample paths from per-step quantiles q (N, H, 9) by piecewise-linear inverse CDF."""
    N, Hh, K = q.shape
    lv = torch.tensor(BOLT_Q, device=DEV, dtype=torch.float32)
    # extend linearly to levels 0 and 1 using the outer segments
    lo = q[..., :1] - (q[..., 1:2] - q[..., :1])
    hi = q[..., -1:] + (q[..., -1:] - q[..., -2:-1])
    qq = torch.cat([lo, q, hi], -1)                                              # (N, H, 11)
    ll = torch.cat([torch.zeros(1, device=DEV), lv, torch.ones(1, device=DEV)])
    if comonotone:
        u = torch.rand(N, S, 1, device=DEV, generator=gen).expand(N, S, Hh)
    else:
        u = torch.rand(N, S, Hh, device=DEV, generator=gen)
    j = torch.searchsorted(ll, u.contiguous()).clamp(1, len(ll) - 1)
    l0, l1 = ll[j - 1], ll[j]
    qe = qq[:, None].expand(N, S, Hh, len(ll))
    v0 = qe.gather(3, (j - 1)[..., None])[..., 0]
    v1 = qe.gather(3, j[..., None])[..., 0]
    return v0 + (u - l0) / (l1 - l0) * (v1 - v0)


@torch.no_grad()
def run_pretrained(d, gen):
    from chronos import BaseChronosPipeline
    from transformers import AutoModelForCausalLM
    ctx = d["ctx"]
    out = {}
    # Chronos-T5: autoregressive token sampling
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map=str(DEV), torch_dtype=torch.float32)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    xs = torch.cat([pipe.predict(ctx[i:i + 5].cpu(), prediction_length=H, num_samples=S).to(DEV)
                    for i in range(0, ctx.shape[0], 5)])           # small batches: 100 samples each
    out["chronos"] = (xs, time.perf_counter() - t0)
    del pipe
    # Chronos-Bolt: direct quantile head
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=str(DEV), torch_dtype=torch.float32)
    t0 = time.perf_counter()
    q = torch.cat([pipe.predict_quantiles(ctx[i:i + 50].cpu(), prediction_length=H,
                                          quantile_levels=list(BOLT_Q))[0].to(DEV)
                   for i in range(0, ctx.shape[0], 50)])
    el = time.perf_counter() - t0
    q = torch.sort(q, -1).values
    out["bolt_ind"] = (bolt_samples(q, gen, False), el)
    out["bolt_com"] = (bolt_samples(q, gen, True), el)
    del pipe
    # Sundial: autoregressive over patches with a flow-matching head (one patch covers the horizon)
    model = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True).to(DEV).eval()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    res = []
    for i in range(0, ctx.shape[0], 25):
        c = ctx[i:i + 25]
        mu = c.mean(1, keepdim=True)
        sd = c.std(1, keepdim=True, unbiased=False).clamp_min(1e-2)
        p = model(input_ids=(c - mu) / sd, revin=False, num_samples=S, max_output_length=H,
                  use_cache=False, return_dict=True).logits
        res.append(p * sd[:, None] + mu[:, None])
    out["sundial"] = (torch.cat(res), time.perf_counter() - t0)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    global N_TEST, S, SEEDS, OUT, MODEL_DIR
    if "--smoke" in sys.argv:
        N_TEST, S, SEEDS = 20, 10, [0]
        sim.CFG.update(n_train=512, train_steps=30)
        OUT = os.path.join(HERE, "results_smoke")
        MODEL_DIR = os.path.join(OUT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    sim.ABAR = sim.cosine_abar(sim.CFG["k_train"])
    torch.backends.cuda.matmul.allow_tf32 = True
    rows, pits, examples = [], [], {}
    t_start = time.perf_counter()
    for ds in DATASETS:
        d = prepare(ds)
        gen = torch.Generator(device=DEV).manual_seed(7)
        ctx, y = d["ctx"], d["fut"]
        naive = ctx[:, -d["season"]:][:, :H]
        nmse = float((((naive - y) / d["sc"]) ** 2).mean())
        print(f"== {ds}: {d['n_series']} series, {d['train'].shape[0]} training windows, "
              f"{ctx.shape[0]} test windows; seasonal naive MSE {nmse:.3f}", flush=True)
        examples[ds] = {"context": ((ctx[:6, -96:] - d["m"][:6]) / d["sc"][:6]).tolist(),
                        "truth": ((y[:6] - d["m"][:6]) / d["sc"][:6]).tolist()}

        def record(label, xs, el, seed, kind):
            met, hist = score(xs, d, gen)
            met.update(dataset=ds, model=label, seed=seed, kind=kind, time_s=el, naive_mse=nmse)
            rows.append(met)
            pits.append(dict(dataset=ds, model=label, seed=seed, hist=hist))
            if seed in (0, None) and label not in examples[ds]:
                examples[ds][label] = ((xs[:6, :30] - d["m"][:6, None]) / d["sc"][:6, None]).tolist()
            print(f"[real] {ds:8s} {label:12s} seed={seed} mse={met['mse']:.3f} crps={met['crps']:.3f} "
                  f"es={met['es']:.3f} vs={met['vs']:.4f} vs_shuf={met['vs_shuf']:.4f} "
                  f"cov90={met['cov90']:.2f} t={el:.1f}s", flush=True)

        for label, (xs, el) in run_pretrained(d, gen).items():
            record(label, xs, el, None, "pretrained")
        for seed in SEEDS:
            for name in SMALL:
                xs, el = run_small(name, d, seed, ds)
                record(name, xs, el, seed, "small")
                if name == "masked":
                    xs, el = run_small(name, d, seed, ds, order="random")
                    record("masked_rand", xs, el, seed, "small")
                torch.cuda.empty_cache()
        for fname, obj in [("real.json", rows), ("pit_real.json", pits), ("examples_real.json", examples)]:
            with open(os.path.join(OUT, fname), "w") as f:
                json.dump(obj, f, indent=1 if fname == "real.json" else None)
    print(f"done in {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
