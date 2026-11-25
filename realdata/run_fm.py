"""Further pretrained foundation models on the test windows of run.py.

Adds four zero-shot foundation models to the real-data study, on the same 300 test windows per
dataset, the same per-window scaling and the same scores as run.py and robust.py:
  aurora        Aurora: flow matching over the forecast tokens, non-autoregressive, sampled
  toto          Toto-Open-Base-1.0: autoregressive over patches, Student-t mixture head, sampled
  timesfm_*     TimesFM 2.5 (200M): autoregressive over patches, point and quantile heads
  chronos2_*    Chronos-2: encoder with a direct multi-step quantile head
Quantile-only models are turned into sample paths as Chronos-Bolt is in run.py, from the nine
levels 0.1 to 0.9: independently at each step (_ind) or with one level per path (_com).
Robustness: Chronos-2 from its 21 native levels (chronos2_native_ind).

Toto pins older library versions and runs in its own environment, so the script can be called
once per environment with a subset of models; results for the models run are merged into
  results/real_fm.json, results/pit_real_fm.json, results/real_fm_perwindow.npz

Usage:  python run_fm.py [aurora timesfm chronos2 toto]      (default: all but toto)
"""

import json
import os
import sys
import time

import numpy as np
import torch

import run
import robust

sim, DEV, H, S = run.sim, run.DEV, run.H, run.S
LF = 512
OUT = run.OUT
LEVELS9 = list(run.BOLT_Q)
INTERVAL = 3600


# =====================================================================
# Sample paths from quantiles
# =====================================================================

def quantile_samples(q, levels, gen, comonotone):
    """Sample paths from per-step quantiles q (N, H, K) at the given levels, by piecewise-linear
    inverse CDF extended linearly to the levels 0 and 1 (run.bolt_samples for any set of levels)."""
    N, Hh, K = q.shape
    lv = torch.tensor(levels, device=DEV, dtype=torch.float32)
    lo = q[..., :1] - (q[..., 1:2] - q[..., :1]) * lv[0] / (lv[1] - lv[0])
    hi = q[..., -1:] + (q[..., -1:] - q[..., -2:-1]) * (1 - lv[-1]) / (lv[-1] - lv[-2])
    qq = torch.cat([lo, q, hi], -1)
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


# =====================================================================
# Forecasters: each returns ("samples", (N, S, H)) or ("quantiles", (N, H, K), levels)
# =====================================================================

@torch.no_grad()
def f_aurora(ctx):
    from aurora import load_model
    model = load_model().to(DEV).eval()
    out = {}
    # native token length 48 (one token covers the horizon); 24 resamples the patch projections
    for label, tl in [("aurora", 48), ("aurora_tok24", 24)]:
        torch.manual_seed(0)
        res = []
        for i in range(0, ctx.shape[0], 25):
            x = model.generate(inputs=ctx[i:i + 25], max_output_length=H, num_samples=S, inference_token_len=tl)
            res.append(x[..., :H].float())                          # generate() standardizes by the context
        out[label] = ("samples", torch.cat(res))
    return out


@torch.no_grad()
def f_timesfm(ctx):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
    model.compile(timesfm.ForecastConfig(max_context=LF, max_horizon=H, normalize_inputs=True,
                                         use_continuous_quantile_head=True, fix_quantile_crossing=True))
    c = ctx.cpu().numpy()
    _, q = model.forecast(horizon=H, inputs=[r for r in c])          # (N, H, 10): mean, then 0.1 ... 0.9
    q = torch.tensor(np.asarray(q)[..., 1:], dtype=torch.float32, device=DEV)
    return {"timesfm": ("quantiles", q, LEVELS9)}


@torch.no_grad()
def f_chronos2(ctx):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=str(DEV))
    native = list(pipe.quantiles)
    out = {}
    for label, levels in [("chronos2", LEVELS9), ("chronos2_native", native)]:
        qs = []
        for i in range(0, ctx.shape[0], 50):
            q, _ = pipe.predict_quantiles(ctx[i:i + 50, None].cpu(), prediction_length=H, quantile_levels=levels)
            qs.append(torch.stack([t[0] for t in q]).to(DEV))        # (b, H, K)
        out[label] = ("quantiles", torch.cat(qs).float(), levels)
    return out


@torch.no_grad()
def f_toto(ctx):
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto
    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(DEV).eval()
    fc = TotoForecaster(toto.model)
    res = []
    for i in range(0, ctx.shape[0], 10):
        c = ctx[i:i + 10, None]                                       # (b, 1 variate, L)
        b, _, L = c.shape
        step = torch.arange(L, device=DEV, dtype=torch.int64) * INTERVAL
        ts = MaskedTimeseries(series=c, padding_mask=torch.ones_like(c, dtype=torch.bool),
                              id_mask=torch.zeros_like(c, dtype=torch.int64),
                              timestamp_seconds=step.expand(b, 1, L).contiguous(),
                              time_interval_seconds=torch.full((b, 1), INTERVAL, device=DEV, dtype=torch.int64))
        f = fc.forecast(ts, prediction_length=H, num_samples=S, samples_per_batch=S)
        res.append(f.samples[:, 0].permute(0, 2, 1))                   # (b, S, H)
    return {"toto": ("samples", torch.cat(res).float())}


RUNNERS = {"aurora": f_aurora, "timesfm": f_timesfm, "chronos2": f_chronos2, "toto": f_toto}


# =====================================================================
# Scoring and bookkeeping
# =====================================================================

def native_quantile_scores(q, levels, d):
    """Quantile loss at the levels 0.1 ... 0.9 and coverage of the central 80% interval, read from
    the quantiles themselves (no sampling)."""
    qs = (torch.sort(q, -1).values - d["m"][:, :, None]) / d["sc"][:, :, None]
    y = (d["fut"] - d["m"]) / d["sc"]
    idx = [min(range(len(levels)), key=lambda k: abs(levels[k] - v)) for v in LEVELS9]
    q9 = qs[..., idx]
    return dict(qloss=robust.qloss(q9, y), cov80_native=float(((y >= q9[..., 0]) & (y <= q9[..., -1])).float().mean()))


def load_json(name, default):
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return default


def main():
    which = [a for a in sys.argv[1:] if not a.startswith("-")] or ["aurora", "timesfm", "chronos2"]
    sim.ABAR = sim.cosine_abar(sim.CFG["k_train"])
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = load_json("real_fm.json", [])
    pits = load_json("pit_real_fm.json", [])
    pw_path = os.path.join(OUT, "real_fm_perwindow.npz")
    pw = dict(np.load(pw_path)) if os.path.exists(pw_path) else {}
    for ds in run.DATASETS:
        d = run.prepare(ds)
        ctx = d["ctx"][:, -LF:]
        global INTERVAL
        INTERVAL = 86400 // d["season"]                               # seconds per step (daily season)
        pw[f"{ds}|sid"], pw[f"{ds}|origin_block"] = d["sid"], d["block"]
        for name in which:
            torch.manual_seed(0)
            t0 = time.perf_counter()
            outs = RUNNERS[name](ctx)
            el = time.perf_counter() - t0
            torch.cuda.empty_cache()
            for label, o in outs.items():
                variants = {label: o[1]} if o[0] == "samples" else {}
                extra = {}
                if o[0] == "quantiles":
                    q, levels = o[1], o[2]
                    extra = native_quantile_scores(q, levels, d)
                    qs = torch.sort(q, -1).values
                    g = torch.Generator(device=DEV).manual_seed(7)
                    variants[label + "_ind"] = quantile_samples(qs, levels, g, False)
                    if label != "chronos2_native":
                        variants[label + "_com"] = quantile_samples(qs, levels, g, True)
                for lab, xs in variants.items():
                    gen = torch.Generator(device=DEV).manual_seed(7)
                    met, hist = run.score(xs, d, gen)
                    met.update({k: v for k, v in robust.score_samples(xs, d, gen, key=f"{ds}|{lab}").items()
                                if k in ("crps_fair", "es_fair", "pit80", "pit90")})
                    if "qloss" not in extra:
                        met["qloss"] = robust.score_samples(xs, d, gen)["qloss"]
                    met.update(extra)
                    met.update(dataset=ds, model=lab, seed=None, kind="pretrained", time_s=el,
                               naive_mse=float((((ctx[:, -d["season"]:][:, :H] - d["fut"]) / d["sc"]) ** 2).mean()))
                    rows = [r for r in rows if not (r["dataset"] == ds and r["model"] == lab)] + [met]
                    pits = [p for p in pits if not (p["dataset"] == ds and p["model"] == lab)] + \
                           [dict(dataset=ds, model=lab, seed=None, hist=hist)]
                    print(f"[fm] {ds:8s} {lab:20s} mse={met['mse']:.3f} crps={met['crps']:.3f} es={met['es']:.3f} "
                          f"vs={met['vs']:.4f} vs_shuf={met['vs_shuf']:.4f} cov90={met['cov90']:.2f} "
                          f"pit90={met['pit90']:.2f} t={el:.1f}s", flush=True)
        pw.update(robust.PER_WINDOW)
        with open(os.path.join(OUT, "real_fm.json"), "w") as f:
            json.dump(rows, f, indent=1)
        with open(os.path.join(OUT, "pit_real_fm.json"), "w") as f:
            json.dump(pits, f)
        np.savez(pw_path, **pw)
    print("done", flush=True)


if __name__ == "__main__":
    main()
