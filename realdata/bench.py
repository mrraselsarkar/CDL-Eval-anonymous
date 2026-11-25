"""Benchmark of foundation models with per-window scores (revised protocol).

Differences from run.py and run_fm.py:
  - any horizon (24, 192, 720) and two data sources: the three held-out datasets of run.py, and
    Australian state demand from AEMO for January to September 2026, which post-dates every model;
  - quantile models are scored at their native levels; every model is scored by the quantile loss and
    the coverage of the central 80% interval at the levels 0.1 to 0.9, which all models provide;
  - per-window scores (CRPS, energy and variogram score, shuffled variogram score, quantile loss,
    coverage) are stored for block-bootstrap intervals;
  - sample-based models are run with several sampling seeds;
  - a probabilistic seasonal naive forecaster is included;
  - the number of sequential network calls of every model is recorded.

Usage:
  python bench.py --data heldout --horizon 24 --models chronos bolt chronos2 timesfm sundial aurora naive
  python bench.py --data heldout --horizon 24 --models toto             (in the Toto environment)
Results are merged into results/bench_<tag>.json and results/bench_<tag>_perwindow.npz (tag: --tag).

Second revision (all additions; the command lines above give the same windows and results as before):
  --data aemo_mid | ausel_season | ercot2026 | heldout_calib   further window sets (see prepare)
  --store rev2          also write the full forecasts, one file per run (format in rev2.py); the summary files then go
                        to results/rev2/ instead of results/ unless --outdir is given
  --context L           context length of the foundation models (default 512; up to 4,096)
  --flow_steps K        sampling steps of the flow heads of Sundial and Aurora (default: their 50 Euler steps)
  --flow_sampler sde    Euler-Maruyama sampler with the same marginals as the flow (ode: the same without noise)
  --max_windows k, --paths S   tiny smoke tests on the first k windows with S paths
"""

import argparse
import glob
import json
import math
import os
import time
import types

import numpy as np
import pandas as pd
import torch

import run

sim, DEV, S = run.sim, run.DEV, run.S
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LF = 512                                   # context of the foundation models
LONG = 2880                                # context kept for the naive forecaster
Q9 = np.round(np.linspace(0.1, 0.9, 9), 2).tolist()


# =====================================================================
# Data
# =====================================================================

def load_aemo():
    """Half-hourly demand of five Australian states, December 2025 to September 2026."""
    out = []
    for region in ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]:
        files = sorted(glob.glob(os.path.join(HERE, "data", "aemo_2026", f"PRICE_AND_DEMAND_*_{region}.csv")))
        df = pd.concat([pd.read_csv(f) for f in files])
        df["t"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
        s = df.set_index("t")["TOTALDEMAND"].sort_index()
        s = s[~s.index.duplicated()].resample("30min", label="right", closed="right").mean().interpolate()
        out.append(s)
    return out


def prepare(data, H, n_test, ctx_len=None):
    """Test windows. For the held-out datasets these are the windows of run.prepare (identical to the
    main study for H = 24). For AEMO, origins are drawn in 2026 after 60 days of history.

    Added in the second revision (the windows above are unchanged): the keys origin (index of the first forecast
    step), origin_ts (its time stamp, int64 ns), week (ISO year * 100 + week of origin_ts, common to all series),
    mu_s and sd_s (standardization of the series of each window: original units = standardized * sd_s + mu_s);
    ctx_len > 2,880 extends the context (left-padded with NaN where the series is shorter); and the window sets
      aemo_mid       AEMO 2015-11 to 2019-12, origins in January to September of 2016-2019 (season-matched to 2026)
      ausel_season   Australian electricity 2002-2015, origins in the last 20% and in January to September
      ercot2026      ERCOT weather zones, origins from January 2026 to the latest complete month
      heldout_calib  150 calibration windows per held-out dataset from the 60-80% part of each series (datasets
                     ercot_calib, traffic_calib, ausel_calib), for the spread correction."""
    if data == "heldout":
        run.H, run.N_TEST, run.LF = H, n_test, LONG
        out = {}
        for ds in run.DATASETS:
            d = run.prepare(ds)
            d.pop("train")
            out[ds] = _heldout_extras(ds, d, ctx_len)
        return out
    if data in NEW_DATA:
        return {data: NEW_DATA[data](H, n_test, ctx_len)}
    if data == "heldout_calib":
        return {f"{ds}_calib": _prepare_series(ds, H, n_test, CALIB_SEED, 0.6, 0.8, None, ctx_len)
                for ds in run.DATASETS}
    series = load_aemo()
    rng = np.random.default_rng(0)
    first = pd.Timestamp("2026-01-01")
    ctx, fut, sid, orig = [], [], [], []
    mus, sds = [], []
    for _ in range(n_test):
        i = int(rng.integers(len(series)))
        s = series[i]
        lo = max(LONG, int(np.searchsorted(s.index, first)))
        o = int(rng.integers(lo, len(s) - H))
        v = s.to_numpy(dtype=float)
        mu, sd = v[:lo].mean(), v[:lo].std() + 1e-8               # scale with the data before the first origin
        ctx.append((v[o - LONG:o] - mu) / sd)
        fut.append((v[o:o + H] - mu) / sd)
        sid.append(i)
        orig.append(o)
        mus.append(mu)
        sds.append(sd)
    ctx, fut = np.stack(ctx), np.stack(fut)
    m = ctx[:, -336:].mean(1, keepdims=True)
    sc = ctx[:, -336:].std(1, keepdims=True) + 1e-3
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    d = dict(ctx=to(ctx), fut=to(fut), m=to(m), sc=to(sc), season=48, n_series=len(series),
             sid=np.array(sid), block=np.array(orig) // 336,
             origin=np.array(orig, dtype=np.int64), scale_win=336, mu_s=np.array(mus), sd_s=np.array(sds))
    if ctx_len and ctx_len > LONG:
        d["ctx"] = to(_context([(s.to_numpy(dtype=float) - mu) / sd for s, mu, sd in
                                zip(series, *_series_stats(d))], d["sid"], d["origin"], ctx_len))
    return {"aemo": _time_keys(d, [s.index for s in series])}


# ---------------------------------------------------------------------
# Window sets of the second revision
# ---------------------------------------------------------------------

CALIB_SEED = 1                              # calibration windows (heldout_calib)
SEEDS_NEW = {"aemo_mid": 11, "ausel_season": 12, "ercot2026": 13}


def _series_stats(d):
    """Per-series (mu, sd) lists from the per-window standardization of a window dict."""
    n = d["n_series"]
    mu, sd = np.full(n, np.nan), np.full(n, np.nan)
    mu[d["sid"]], sd[d["sid"]] = d["mu_s"], d["sd_s"]
    return mu, sd


def _time_keys(d, times):
    """Adds origin_ts (time stamp of the first forecast step, int64 ns) and week (ISO year * 100 + ISO week)."""
    ts = pd.DatetimeIndex([times[i][o] for i, o in zip(d["sid"], d["origin"])]).as_unit("ns")
    d["origin_ts"] = ts.asi8.astype(np.int64)
    iso = ts.isocalendar()
    d["week"] = (iso["year"].to_numpy() * 100 + iso["week"].to_numpy()).astype(np.int64)
    return d


def _context(values, sid, origin, L):
    """Contexts of L steps before each origin, left-padded with NaN where the series is shorter."""
    out = np.full((len(sid), L), np.nan)
    for k, (i, o) in enumerate(zip(sid, origin)):
        c = values[i][max(0, o - L):o]
        out[k, L - len(c):] = c
    return out


def _windows(values, times, picks, H, season, scale_win, mu, sd, ctx_len=None):
    """Window dict in the format of run.prepare from standardized series and (series, origin) pairs."""
    L = max(LONG, ctx_len or 0)
    sid = np.array([i for i, _ in picks], dtype=np.int64)
    orig = np.array([o for _, o in picks], dtype=np.int64)
    ctx = _context(values, sid, orig, L)
    fut = np.stack([values[i][o:o + H] for i, o in picks])
    m = ctx[:, -scale_win:].mean(1, keepdims=True)
    sc = ctx[:, -scale_win:].std(1, keepdims=True) + 1e-3
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    d = dict(ctx=to(ctx), fut=to(fut), m=to(m), sc=to(sc), season=season, n_series=len(values), sid=sid,
             block=orig // scale_win, origin=orig, scale_win=scale_win,
             mu_s=np.asarray(mu, dtype=float)[sid], sd_s=np.asarray(sd, dtype=float)[sid])
    return _time_keys(d, times)


def _heldout_extras(ds, d, ctx_len):
    """Time keys (and a longer context) for the windows of run.prepare."""
    import fresh_data
    fname = run.DATASETS[ds][0]
    times = fresh_data.parquet_timestamps(fname)
    if ctx_len and ctx_len > LONG:
        raw = run.load_series(fname)
        z = [(s - s[:int(0.7 * len(s))].mean()) / (s[:int(0.7 * len(s))].std() + 1e-8) for s in raw]
        d["ctx"] = torch.tensor(_context(z, d["sid"], d["origin"], ctx_len), dtype=torch.float32, device=DEV)
    return _time_keys(d, times)


def _prepare_series(ds, H, n_test, seed, lo_frac, hi_frac, months, ctx_len):
    """Windows from a held-out dataset, standardized as in run.prepare (first 70% of each series): origins in
    [lo_frac, hi_frac) of each series (hi_frac = 1: up to the end, as the test windows; otherwise the forecast ends
    before hi_frac), optionally only in the given calendar months; a series is drawn with probability proportional
    to its number of admissible origins, and the origin uniformly among them."""
    import fresh_data
    fname, season, scale_win = run.DATASETS[ds]
    raw = run.load_series(fname)
    times = fresh_data.parquet_timestamps(fname)
    vals, mus, sds, valid = [], [], [], []
    for s, t in zip(raw, times):
        n = len(s)
        mu, sd = s[:int(0.7 * n)].mean(), s[:int(0.7 * n)].std() + 1e-8
        vals.append((s - mu) / sd)
        mus.append(mu)
        sds.append(sd)
        a = max(int(lo_frac * n), LONG)
        b = n - H if hi_frac >= 1 else int(hi_frac * n) - H
        idx = np.arange(a, max(a, b))
        if months is not None and len(idx):
            idx = idx[np.isin(t[idx].month, months)]
        valid.append(idx)
    rng = np.random.default_rng(seed)
    w = np.array([len(v) for v in valid], dtype=float)
    pick = rng.choice(len(valid), size=n_test, p=w / w.sum())
    picks = [(int(i), int(valid[i][rng.integers(len(valid[i]))])) for i in pick]
    return _windows(vals, times, picks, H, season, scale_win, mus, sds, ctx_len)


def _prepare_fresh(series, first, H, n_test, seed, season, scale_win, ctx_len, months=None, years=None):
    """Windows from recent data, as for AEMO 2026: each series standardized with its values before `first` (at least
    the 2,880 steps of context), a series drawn uniformly, and the origin uniformly from `first` on (optionally only
    in the given months and years)."""
    vals, mus, sds, valid = [], [], [], []
    for s in series:
        v = s.to_numpy(dtype=float)
        lo = max(LONG, int(np.searchsorted(s.index, first)))
        mu, sd = v[:lo].mean(), v[:lo].std() + 1e-8
        vals.append((v - mu) / sd)
        mus.append(mu)
        sds.append(sd)
        idx = np.arange(lo, len(v) - H)
        t = s.index[idx]
        keep = np.ones(len(idx), dtype=bool)
        if months is not None:
            keep &= np.isin(t.month, months)
        if years is not None:
            keep &= np.isin(t.year, years)
        valid.append(idx[keep])
    rng = np.random.default_rng(seed)
    picks = []
    for _ in range(n_test):
        i = int(rng.integers(len(series)))
        picks.append((i, int(valid[i][rng.integers(len(valid[i]))])))
    return _windows(vals, [s.index for s in series], picks, H, season, scale_win, mus, sds, ctx_len)


def prepare_aemo_mid(H, n_test, ctx_len=None):
    import fresh_data
    return _prepare_fresh(fresh_data.load_aemo_mid(), pd.Timestamp("2016-01-01"), H, n_test, SEEDS_NEW["aemo_mid"],
                          48, 336, ctx_len, months=list(range(1, 10)), years=[2016, 2017, 2018, 2019])


def prepare_ausel_season(H, n_test, ctx_len=None):
    return _prepare_series("ausel", H, n_test, SEEDS_NEW["ausel_season"], 0.8, 1.0, list(range(1, 10)), ctx_len)


def prepare_ercot2026(H, n_test, ctx_len=None):
    import fresh_data
    return _prepare_fresh(fresh_data.load_ercot2026(), pd.Timestamp("2026-01-01"), H, n_test, SEEDS_NEW["ercot2026"],
                          24, 168, ctx_len)


NEW_DATA = {"aemo_mid": prepare_aemo_mid, "ausel_season": prepare_ausel_season, "ercot2026": prepare_ercot2026}


def subset(d, k):
    """The first k windows of a window dict (smoke tests)."""
    n = d["ctx"].shape[0]
    return {key: (v[:k] if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == n else v)
            for key, v in d.items()}


# ---------------------------------------------------------------------
# Flow heads of Sundial and Aurora: number of steps and sampler
# ---------------------------------------------------------------------

FLOW = {"steps": None, "sampler": "euler", "eta": 1.0}


def _integrate(head, z, mu0):
    """Sampler of a flow head whose network predicts the clean target x1 from x_t = t x1 + (1 - t) x0, with
    x0 ~ N(mu0, I). With x0 estimated as (x_t - t x1_hat) / (1 - t), the velocity is v = x1_hat - x0_hat and the score
    is -(x0_hat - mu0) / (1 - t). "ode" integrates v with Euler steps; "sde" integrates
    dx = [v - eta^2 (x0_hat - mu0)] dt + eta sqrt(2 (1 - t)) dW (Euler-Maruyama, diffusion g^2 = 2 eta^2 (1 - t)),
    which has the same marginals; no noise is added in the last step."""
    eta = FLOW["eta"] if FLOW["sampler"] == "sde" else 0.0
    K = head.num_sampling_steps
    dt = 1.0 / K
    x = mu0 + torch.randn(z.shape[0], head.in_channels, device=z.device)
    for i in range(K):
        t = i / K
        x1 = head.net(x, torch.full((x.shape[0],), t * 1000.0, device=x.device), z)
        x0 = (x - t * x1) / (1 - t)
        x = x + (x1 - x0 - eta ** 2 * (x0 - mu0)) * dt
        if eta > 0 and i < K - 1:
            x = x + eta * math.sqrt(2 * (1 - t) * dt) * torch.randn_like(x)
    return x


def _sundial_sample(self, z, num_samples=1):
    z = z.repeat(num_samples, 1)
    x = _integrate(self, z, torch.zeros(z.shape[0], self.in_channels, device=z.device))
    return x.reshape(num_samples, -1, self.in_channels).transpose(0, 1)


def _aurora_sample(self, z, prototype=None, num_samples=1, inference_token_len=48):
    import torch.nn.functional as F
    from aurora.util_functions import resample
    z = z.repeat(num_samples, 1)
    mu0 = prototype.repeat(num_samples, 1) if prototype is not None else \
        torch.zeros(z.shape[0], self.in_channels, device=z.device)
    x = _integrate(self, z, mu0)
    if not self.training:
        x = F.linear(x, resample(torch.eye(self.in_channels).to(x.device), inference_token_len).T)
        return x.reshape(num_samples, -1, inference_token_len).transpose(0, 1)
    return x.reshape(num_samples, -1, self.in_channels).transpose(0, 1)


def _flow_head(head, kind):
    """Applies FLOW to a flow head; returns its number of sampling steps (50 by default)."""
    if FLOW["steps"]:
        head.num_sampling_steps = int(FLOW["steps"])
    if FLOW["sampler"] != "euler":
        head.sample = types.MethodType(_sundial_sample if kind == "sundial" else _aurora_sample, head)
    return head.num_sampling_steps


# =====================================================================
# Forecasters. Each returns (kind, array, levels, sequential calls per forecast)
# =====================================================================

@torch.no_grad()
def f_chronos(ctx, H, seed, size="small"):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(f"amazon/chronos-t5-{size}", device_map=str(DEV),
                                               torch_dtype=torch.float32)
    torch.manual_seed(seed)
    res, b, i = [], 5, 0
    while i < ctx.shape[0]:
        try:
            res.append(pipe.predict(ctx[i:i + b].cpu(), prediction_length=H, num_samples=S,
                                    limit_prediction_length=False).to(DEV).float())
            i += b
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            b = max(1, b // 2)
    return "samples", torch.cat(res), None, H


def f_chronos_base(ctx, H, seed):
    return f_chronos(ctx, H, seed, "base")


@torch.no_grad()
def f_bolt(ctx, H, seed, size="small"):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(f"amazon/chronos-bolt-{size}", device_map=str(DEV),
                                               torch_dtype=torch.float32)
    q = torch.cat([pipe.predict_quantiles(ctx[i:i + 50].cpu(), prediction_length=H, quantile_levels=Q9,
                                          limit_prediction_length=False)[0].to(DEV).float()
                   for i in range(0, ctx.shape[0], 50)])
    return "quantiles", q, Q9, math.ceil(H / 64)


def f_bolt_base(ctx, H, seed):
    return f_bolt(ctx, H, seed, "base")


@torch.no_grad()
def f_chronos2(ctx, H, seed):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=str(DEV))
    levels = [round(float(x), 4) for x in pipe.quantiles]
    qs = []
    for i in range(0, ctx.shape[0], 50):
        q, _ = pipe.predict_quantiles(ctx[i:i + 50, None].cpu(), prediction_length=H, quantile_levels=levels)
        qs.append(torch.stack([t[0] for t in q]).to(DEV).float())
    cfg = pipe.model.chronos_config
    return "quantiles", torch.cat(qs), levels, math.ceil(H / (cfg.output_patch_size * cfg.max_output_patches))


@torch.no_grad()
def f_timesfm(ctx, H, seed):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
    model.compile(timesfm.ForecastConfig(max_context=LF, max_horizon=max(H, 128), normalize_inputs=True,
                                         per_core_batch_size=32, use_continuous_quantile_head=True,
                                         fix_quantile_crossing=True))
    _, q = model.forecast(horizon=H, inputs=[r for r in ctx.cpu().numpy()])
    q = torch.tensor(np.asarray(q)[:, :H, 1:], dtype=torch.float32, device=DEV)
    return "quantiles", q, Q9, math.ceil(H / 128)


@torch.no_grad()
def f_toto(ctx, H, seed):
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto
    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(DEV).eval()
    fc = TotoForecaster(toto.model)
    torch.manual_seed(seed)
    res = []
    for i in range(0, ctx.shape[0], 10):
        c = ctx[i:i + 10, None]
        b, _, L = c.shape
        ts = MaskedTimeseries(series=c, padding_mask=torch.ones_like(c, dtype=torch.bool),
                              id_mask=torch.zeros_like(c, dtype=torch.int64),
                              timestamp_seconds=torch.zeros_like(c, dtype=torch.int64),   # not used by the model
                              time_interval_seconds=torch.full((b, 1), 3600, device=DEV, dtype=torch.int64))
        f = fc.forecast(ts, prediction_length=H, num_samples=S, samples_per_batch=S)
        res.append(f.samples[:, 0].permute(0, 2, 1).float())
    return "samples", torch.cat(res), None, math.ceil(H / 64)


@torch.no_grad()
def f_sundial(ctx, H, seed):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True).to(DEV).eval()
    steps = _flow_head(model.flow_loss, "sundial")
    torch.manual_seed(seed)
    res = []
    for i in range(0, ctx.shape[0], 25):
        c = ctx[i:i + 25]
        mu = c.mean(1, keepdim=True)
        sd = c.std(1, keepdim=True, unbiased=False).clamp_min(1e-2)
        p = model(input_ids=(c - mu) / sd, revin=False, num_samples=S, max_output_length=H,
                  use_cache=False, return_dict=True).logits
        res.append(p[..., :H].float() * sd[:, None] + mu[:, None])
    return "samples", torch.cat(res), None, math.ceil(H / 720) * (steps + 1)


@torch.no_grad()
def f_aurora(ctx, H, seed):
    from aurora import load_model
    model = load_model().to(DEV).eval()
    steps = _flow_head(model.flow_match, "aurora")
    torch.manual_seed(seed)
    res = [model.generate(inputs=ctx[i:i + 25], max_output_length=H, num_samples=S,
                          inference_token_len=48)[..., :H].float() for i in range(0, ctx.shape[0], 25)]
    return "samples", torch.cat(res), None, steps + 1


@torch.no_grad()
def f_naive(ctx_long, H, seed, season):
    """Seasonal naive with paths of its own past errors: the forecast repeats the last season, and each
    sample path adds the error path that the same rule made from a random earlier origin in the context."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    N, L = ctx_long.shape
    reps = math.ceil(H / season)
    base = ctx_long[:, -season:].repeat(1, reps)[:, :H]
    lo, hi = season, L - H                                         # origins whose future lies in the context
    o = torch.randint(lo, hi, (N, S), device=DEV, generator=g)
    idx = o[..., None] + torch.arange(H, device=DEV)               # (N, S, H) positions of the future
    lag = o[..., None] - season + (torch.arange(H, device=DEV) % season)
    x = ctx_long[:, None, :].expand(N, S, L)
    err = x.gather(2, idx) - x.gather(2, lag)
    return "samples", base[:, None] + err, None, 0


RUNNERS = {"chronos": f_chronos, "chronos_base": f_chronos_base, "bolt": f_bolt, "bolt_base": f_bolt_base,
           "chronos2": f_chronos2, "timesfm": f_timesfm, "toto": f_toto, "sundial": f_sundial,
           "aurora": f_aurora, "naive": f_naive}
SAMPLED = {"chronos", "chronos_base", "toto", "sundial", "aurora", "naive"}


# =====================================================================
# Scoring
# =====================================================================

def quantile_samples(q, levels, gen):
    """Paths drawn independently at each step from per-step quantiles (piecewise-linear inverse CDF,
    extended linearly to the levels 0 and 1)."""
    N, Hh, K = q.shape
    lv = torch.tensor(levels, device=DEV, dtype=torch.float32)
    lo = q[..., :1] - (q[..., 1:2] - q[..., :1]) * lv[0] / (lv[1] - lv[0])
    hi = q[..., -1:] + (q[..., -1:] - q[..., -2:-1]) * (1 - lv[-1]) / (lv[-1] - lv[-2])
    qq = torch.cat([lo, q, hi], -1)
    ll = torch.cat([torch.zeros(1, device=DEV), lv, torch.ones(1, device=DEV)])
    u = torch.rand(N, S, Hh, device=DEV, generator=gen)
    j = torch.searchsorted(ll, u.contiguous()).clamp(1, len(ll) - 1)
    qe = qq[:, None].expand(N, S, Hh, len(ll))
    v0, v1 = qe.gather(3, (j - 1)[..., None])[..., 0], qe.gather(3, j[..., None])[..., 0]
    return v0 + (u - ll[j - 1]) / (ll[j] - ll[j - 1]) * (v1 - v0)


def window_scores(kind, arr, levels, d, gen):
    """Per-window scores on the scale of the last week of context."""
    m, sc = d["m"], d["sc"]
    y = (d["fut"] - m) / sc
    if kind == "quantiles":
        qn = (torch.sort(arr, -1).values - m[:, :, None]) / sc[:, :, None]
        idx = [min(range(len(levels)), key=lambda k: abs(levels[k] - v)) for v in Q9]
        q9 = qn[..., idx]
        x = quantile_samples(qn, levels, gen)
    else:
        x = (arr - m[:, None]) / sc[:, None]
        q9 = torch.quantile(x, torch.tensor(Q9, device=DEV), dim=1).permute(1, 2, 0)
    lv = torch.tensor(Q9, device=DEV)
    diff = y[..., None] - q9
    ql = (2 * torch.maximum(lv * diff, (lv - 1) * diff)).mean((1, 2))
    if kind == "quantiles":
        cov80 = ((y >= q9[..., 0]) & (y <= q9[..., -1])).float().mean(1)
    else:
        below = (x < y[:, None]).sum(1).float()
        u = (below + torch.rand(below.shape, device=DEV, generator=gen)) / (x.shape[1] + 1)
        cov80 = ((u > 0.1) & (u < 0.9)).float().mean(1)
    below = (x < y[:, None]).sum(1).float()
    u = (below + torch.rand(below.shape, device=DEV, generator=gen)) / (x.shape[1] + 1)
    cov90 = ((u > 0.05) & (u < 0.95)).float().mean(1)
    perm = torch.argsort(torch.rand(x.shape, device=DEV, generator=gen), dim=1)
    xs = x.gather(1, perm)
    crps_h = sim.crps_samples(x, y)                                  # (N, H)
    # dependence that does not depend on spread: lag-one rank correlation of the forecast paths within each
    # window (rho_model); the PIT values u are kept to compute the observed rank correlation across windows
    rx = x.argsort(1).argsort(1).float()
    rx = (rx - rx.mean(1, keepdim=True)) / rx.std(1, keepdim=True).clamp_min(1e-6)
    rho_model = ((rx[:, :, :-1] * rx[:, :, 1:]).sum(1) / (x.shape[1] - 1)).mean(1)
    chunk = max(1, int(2e8 // (x.shape[1] * x.shape[2] ** 2)))      # windows per chunk: memory grows with H^2
    out = dict(crps=crps_h.mean(1), es=sim.energy_score(x, y), vs=run.variogram_score(x, y, chunk=chunk),
               vs_shuf=run.variogram_score(xs, y, chunk=chunk), ql=ql, cov80=cov80, cov90=cov90,
               rho_model=rho_model, pit=u)
    lead = dict(crps=crps_h.mean(0), ql=(2 * torch.maximum(lv * diff, (lv - 1) * diff)).mean((0, 2)))
    return {k: v.float().cpu().numpy() for k, v in out.items()}, {k: v.float().cpu().numpy() for k, v in lead.items()}


# =====================================================================
# Main
# =====================================================================

def setting_name(a):
    """Name of a non-default setting (part of the rev2 file name), or None for the protocol of the article."""
    parts = []
    if a.context and a.context != 512:
        parts.append(f"ctx{a.context}")
    if a.flow_sampler != "euler":
        parts.append(f"{a.flow_sampler}{a.flow_steps or 50}" + (f"eta{a.flow_eta:g}" if a.flow_sampler == "sde"
                                                                 and a.flow_eta != 1.0 else ""))
    elif a.flow_steps:
        parts.append(f"euler{a.flow_steps}")
    return "_".join(parts) or None


def main():
    global LF, S
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="heldout", choices=["heldout", "aemo"] + list(NEW_DATA) + ["heldout_calib"])
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tag", default="main", help="one result file per environment, so that runs do not collide")
    # second revision
    ap.add_argument("--store", default="none", choices=["none", "rev2"],
                    help="rev2: also write the full forecasts of every run (rev2.py)")
    ap.add_argument("--outdir", default=None,
                    help="directory of the result files (default: results/, or results/rev2 with --store rev2)")
    ap.add_argument("--context", type=int, default=None, help="context length of the foundation models (512)")
    ap.add_argument("--flow_steps", type=int, default=None, help="sampling steps of the Sundial and Aurora heads (50)")
    ap.add_argument("--flow_sampler", default="euler", choices=["euler", "ode", "sde"])
    ap.add_argument("--flow_eta", type=float, default=1.0)
    ap.add_argument("--setting", default=None, help="name of the setting in the rev2 file names (default: derived)")
    ap.add_argument("--max_windows", type=int, default=None, help="smoke tests: only the first k windows")
    ap.add_argument("--paths", type=int, default=None, help="smoke tests: number of sample paths (100)")
    a = ap.parse_args()
    if a.context:
        LF = a.context
    if a.paths:
        S = a.paths
    FLOW.update(steps=a.flow_steps, sampler=a.flow_sampler, eta=a.flow_eta)
    setting = a.setting or setting_name(a)
    out_dir = a.outdir or (os.path.join(OUT, "rev2") if a.store == "rev2" else OUT)
    os.makedirs(out_dir, exist_ok=True)
    sim.ABAR = sim.cosine_abar(sim.CFG["k_train"])
    torch.backends.cuda.matmul.allow_tf32 = True
    jpath = os.path.join(out_dir, f"bench_{a.tag}.json")
    npath = os.path.join(out_dir, f"bench_{a.tag}_perwindow.npz")
    rows = json.load(open(jpath)) if os.path.exists(jpath) else []
    pw = dict(np.load(npath)) if os.path.exists(npath) else {}
    sets = prepare(a.data, a.horizon, a.n_test, ctx_len=a.context)
    for ds, d in sets.items():
        if a.max_windows:
            d = subset(d, a.max_windows)
        pw[f"{ds}|{a.horizon}|sid"], pw[f"{ds}|{a.horizon}|block"] = d["sid"], d["block"]
        ctx = d["ctx"][:, -LF:]
        for name in a.models:
            seeds = a.seeds if name in SAMPLED else [0]
            for seed in seeds:
                t0 = time.perf_counter()
                if name == "naive":
                    kind, arr, levels, calls = f_naive(d["ctx"], a.horizon, seed, d["season"])
                else:
                    kind, arr, levels, calls = RUNNERS[name](ctx, a.horizon, seed)
                el = time.perf_counter() - t0
                torch.cuda.empty_cache()
                gen = torch.Generator(device=DEV).manual_seed(7 + seed)
                win, lead = window_scores(kind, arr, levels, d, gen)
                label = name if setting is None else f"{name}@{setting}"
                key = f"{ds}|{a.horizon}|{label}|{seed}"
                for k, v in win.items():
                    pw[f"{key}|{k}"] = v
                row = {k: float(v.mean()) for k, v in win.items() if k != "pit"}
                ru = win["pit"].argsort(0).argsort(0).astype(float)          # observed lag-one rank correlation
                ru = (ru - ru.mean(0)) / ru.std(0)
                row["rho_obs"] = float((ru[:, :-1] * ru[:, 1:]).mean(0).mean())
                row.update(dataset=ds, horizon=a.horizon, model=name, seed=seed, kind=kind, calls=calls,
                           time_s=el, n_test=int(d["ctx"].shape[0]),
                           lead_crps=lead["crps"].tolist(), lead_ql=lead["ql"].tolist())
                if setting is not None:
                    row.update(setting=setting, context=LF)
                rows = [r for r in rows if not (r["dataset"] == ds and r["horizon"] == a.horizon and
                                                r["model"] == name and r["seed"] == seed and
                                                r.get("setting") == setting)] + [row]
                if a.store == "rev2":
                    import rev2
                    meta = dict(context=int(d["ctx"].shape[1]) if name == "naive" else int(LF), calls=calls,
                                time_s=el, tag=a.tag, data=a.data, flow_steps=a.flow_steps,
                                flow_sampler=a.flow_sampler, flow_eta=a.flow_eta,
                                season=int(d["season"]), steps_per_week=int(d.get("scale_win", 0)),
                                n_series=int(d["n_series"]), levels_native=kind == "quantiles")
                    rev2.write_forecast(ds, a.horizon, name, seed, kind, arr, levels, d,
                                        pit=win["pit"] if kind == "samples" else None, setting=setting, meta=meta,
                                        out_dir=out_dir)
                print(f"[bench] {ds:8s} H={a.horizon:<4d} {label:13s} s{seed} crps={row['crps']:.3f} "
                      f"ql={row['ql']:.3f} es={row['es']:.3f} gain={100 * (row['vs_shuf'] / row['vs'] - 1):+.1f}% "
                      f"cov80={row['cov80']:.2f} cov90={row['cov90']:.2f} t={el:.0f}s", flush=True)
                with open(jpath, "w") as f:
                    json.dump(rows, f)
                np.savez(npath, **pw)
    print("done", flush=True)


if __name__ == "__main__":
    main()
