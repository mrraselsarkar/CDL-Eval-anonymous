"""Experiments of the second revision that need more than bench.py. Results are rev2 files (rev2.py).

  chunked   Chronos-2 at H = 720 on the 150 long-horizon windows of the held-out datasets, rolled out in chunks:
            --chunk 64 --feedback median   (64-step chunks, the median fed back as context)
            --chunk 24 --feedback median
            --chunk 64 --feedback paths    (each of the 100 paths rolled out separately: one draw per step from the
                                            native quantiles is appended to the context of its path; 100 series per
                                            call)
            The one-pass baseline is bench.py --horizon 720 --n_test 150 --models chronos2 --store rev2.   [GPU]
  copula    Paths for quantile models (Chronos-2, TimesFM 2.5) from their stored quantiles: 100 draws per step,
            independently, then (i) reordered by the Schaake shuffle with the ranks of the most recent 100 historical
            H-step trajectories of the same series that start at the same time of day (setting "schaake"), and
            (ii) joined by a Gaussian copula with the correlation of the normal scores of those trajectories
            (setting "gausscop").                                                                           [CPU]
  spread    Spread correction of sample paths: deviations from the per-step median are scaled by one factor c per
            model and dataset, chosen so that the PIT coverage of the central 80% interval on calibration windows is
            80%. Calibration windows: <dataset>_calib (150 windows from the 60-80% part of each held-out series, run
            by bench.py --data heldout_calib); for datasets without calibration windows (aemo, aemo_mid, ercot2026)
            leave-one-month-out cross-fitting on the test windows. Setting "spread".                          [CPU]

Usage:
  python revision_runs.py chunked --chunk 64 --feedback median
  python revision_runs.py copula  [--datasets ercot traffic ausel aemo] [--models chronos2 timesfm]
  python revision_runs.py spread  [--datasets ...] [--models ...]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

import rev2

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_MODELS = ["naive", "chronos", "chronos_base", "toto", "sundial", "aurora"]
QUANTILE_MODELS = ["chronos2", "timesfm"]
DATASETS24 = ["ercot", "traffic", "ausel", "aemo", "aemo_mid", "ausel_season", "ercot2026"]
CALIB_OF = {"ercot": "ercot_calib", "traffic": "traffic_calib", "ausel": "ausel_calib", "ausel_season": "ausel_calib"}
SEASON = {"ercot": 24, "traffic": 24, "ausel": 48, "aemo": 48, "aemo_mid": 48, "ausel_season": 48, "ercot2026": 24}


# =====================================================================
# Series history (original units) for the copulas
# =====================================================================

_SERIES = {}


def series_of(dataset):
    """Series of a dataset (pandas Series with time stamps), in the series order of bench.prepare."""
    import fresh_data
    base = dataset.replace("_calib", "")
    if base not in _SERIES:
        if base in ("ercot", "traffic", "ausel", "ausel_season"):
            _SERIES[base] = fresh_data.load_parquet(fresh_data.DATASET_FILES["ausel" if base == "ausel_season" else base])
        elif base == "aemo":
            _SERIES[base] = fresh_data.load_aemo2026()
        elif base == "aemo_mid":
            _SERIES[base] = fresh_data.load_aemo_mid()
        elif base == "ercot2026":
            _SERIES[base] = fresh_data.load_ercot2026()
        else:
            raise KeyError(dataset)
    return _SERIES[base]


def origin_index(series, sid, origin_ns):
    """Index of the first forecast step of every window in its series."""
    out = np.empty(len(sid), dtype=np.int64)
    cache = {}
    for k, (i, t) in enumerate(zip(sid, origin_ns)):
        if i not in cache:
            cache[i] = series[i].index.as_unit("ns").asi8
        j = int(np.searchsorted(cache[i], t))
        assert j < len(cache[i]) and cache[i][j] == t, "origin not found in series"
        out[k] = j
    return out


def history_trajectories(values, o, H, season, n=100):
    """The most recent n trajectories of H steps that start at the same time of day as the forecast (o - k * season)
    and end before the origin. Returns (m, H) with m <= n."""
    k0 = math.ceil(H / season)
    starts = [o - k * season for k in range(k0, k0 + n) if o - k * season >= 0]
    return np.stack([values[s:s + H] for s in starts]) if starts else np.empty((0, H))


# =====================================================================
# Copulas for quantile models
# =====================================================================

def copula_paths(q, levels, traj, S, rng, kind):
    """Paths (S, H) for one window from quantiles q (K, H) and historical trajectories traj (m, H)."""
    from scipy.stats import norm
    H = q.shape[1]
    m = traj.shape[0]
    if kind == "schaake":
        x = rev2.sample_quantiles(q[None], levels, rng.random((1, S, H)))[0]          # independent draws (S, H)
        x = np.sort(x, 0)
        member = np.arange(S) % m                                                       # template, repeated if m < S
        tmpl = traj[member] + 1e-9 * rng.random((S, H))                                 # random tie-breaking
        r = tmpl.argsort(0).argsort(0)                                                  # rank of each member per step
        return np.take_along_axis(x, r, 0)
    # Gaussian copula: correlation of the normal scores of the trajectories
    from scipy.stats import rankdata
    z = norm.ppf(rankdata(traj, axis=0) / (m + 1))
    R = np.corrcoef(z.T)
    R = np.where(np.isfinite(R), R, 0.0)
    np.fill_diagonal(R, 1.0)
    R = 0.999 * R + 0.001 * np.eye(H)                                                   # positive definite
    L = np.linalg.cholesky(R)
    u = norm.cdf(rng.standard_normal((S, H)) @ L.T)
    return rev2.sample_quantiles(q[None], levels, np.clip(u, 1e-6, 1 - 1e-6)[None])[0]


def run_copula(datasets, models, S=100, out_dir=None, n_hist=100):
    for ds in datasets:
        for model in models:
            for path, key in rev2.listing(ds, 24, model, "", out_dir):
                r = rev2.load(path)
                series = series_of(ds)
                vals = [s.to_numpy(dtype=float) for s in series]
                o = origin_index(series, r["sid"], r["origin"])
                H = r["y"].shape[1]
                season = r["meta"].get("season") or SEASON[ds]
                for kind in ["schaake", "gausscop"]:
                    rng = np.random.default_rng(1000 + key["seed"])
                    paths, n_traj = [], []
                    for n in range(len(o)):
                        traj = history_trajectories(vals[r["sid"][n]], o[n], H, season, n_hist)
                        n_traj.append(len(traj))
                        paths.append(copula_paths(r["q"][n], r["levels"], traj, S, rng, kind))
                    paths = np.stack(paths).astype(np.float32)
                    pit = rev2.randomized_pit(paths, r["y"], rng)
                    meta = dict(r["meta"], setting=kind, kind="copula", n_paths=S, base="native quantiles",
                                copula=kind, n_hist_min=int(min(n_traj)), n_hist_median=float(np.median(n_traj)))
                    rev2.save(rev2.fname(ds, H, model, key["seed"], kind, out_dir), y=r["y"], y_orig=r["y_orig"],
                              m=r["m"], sd=r["sd"], levels=r["levels"], q=r["q"], pit=pit, sid=r["sid"],
                              origin=r["origin"], block=r["block"], week=r["week"], meta=meta, paths=paths)
                    print(f"[copula] {ds} {model} s{key['seed']} {kind}: {len(o)} windows, "
                          f"trajectories min {min(n_traj)} median {np.median(n_traj):.0f}", flush=True)


# =====================================================================
# Spread correction of sample paths
# =====================================================================

def pit_coverage(dev, target, V, c, lo=0.1, hi=0.9):
    """PIT coverage of the central interval after scaling the deviations from the median by c."""
    below = (dev < (target / c)[:, None]).sum(1)
    u = (below + V) / (dev.shape[1] + 1)
    return float(((u > lo) & (u < hi)).mean())


def fit_factor(paths, y, rng, goal=0.8):
    """Smallest factor c with PIT coverage >= goal (the coverage does not decrease in c)."""
    x = np.asarray(paths, np.float32)
    med = np.median(x, 1, keepdims=True)
    dev, target = x - med, np.asarray(y, np.float32) - med[:, 0]
    V = rng.random(target.shape)
    lo, hi = math.log(0.05), math.log(50.0)
    if pit_coverage(dev, target, V, math.exp(hi)) < goal:
        return math.exp(hi)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if pit_coverage(dev, target, V, math.exp(mid)) >= goal:
            hi = mid
        else:
            lo = mid
    return math.exp(hi)


def apply_factor(paths, c):
    x = np.asarray(paths, np.float32)
    med = np.median(x, 1, keepdims=True)
    return med + np.asarray(c, np.float32).reshape(-1, 1, 1) * (x - med)


def run_spread(datasets, models, out_dir=None):
    for ds in datasets:
        for model in models:
            runs = rev2.listing(ds, 24, model, "", out_dir)
            if not runs:
                continue
            cal_ds = CALIB_OF.get(ds)
            c_cal = None
            if cal_ds is not None:
                cal = rev2.load_run(cal_ds, 24, model, "", seeds=[0], out_dir=out_dir)
                if not cal:
                    print(f"[spread] {ds} {model}: no calibration run {cal_ds}, skipped", flush=True)
                    continue
                c_cal = fit_factor(cal[0]["paths"], cal[0]["y"], np.random.default_rng(5))
            for path, key in runs:
                r = rev2.load(path)
                if "paths" not in r:
                    continue
                N = r["y"].shape[0]
                if c_cal is not None:
                    c = np.full(N, c_cal)
                    how = dict(calibration=cal_ds, c=c_cal)
                else:                                     # leave-one-month-out cross-fitting on the test windows
                    month = pd.DatetimeIndex(r["origin"]).to_period("M").astype(str).to_numpy()
                    c, per = np.empty(N), {}
                    for mo in np.unique(month):
                        out = month != mo
                        per[mo] = fit_factor(r["paths"][out], r["y"][out], np.random.default_rng(5)) \
                            if out.any() else 1.0                 # a single month (smoke tests): no correction
                        c[month == mo] = per[mo]
                    how = dict(calibration="leave-one-month-out", c_by_month=per)
                x = apply_factor(r["paths"], c)
                rng = np.random.default_rng(2000 + key["seed"])
                pit = rev2.randomized_pit(x, r["y"], rng)
                meta = dict(r["meta"], setting="spread", spread=how)
                rev2.save(rev2.fname(ds, 24, model, key["seed"], "spread", out_dir), y=r["y"], y_orig=r["y_orig"],
                          m=r["m"], sd=r["sd"], levels=rev2.SAMPLE_LEVELS, q=rev2.sample_level_quantiles(x), pit=pit,
                          sid=r["sid"], origin=r["origin"], block=r["block"], week=r["week"], meta=meta, paths=x)
                cov0 = float(((r["pit"] > 0.1) & (r["pit"] < 0.9)).mean())
                cov1 = float(((pit > 0.1) & (pit < 0.9)).mean())
                print(f"[spread] {ds} {model} s{key['seed']}: c = {np.median(c):.3f} ({how['calibration']}); "
                      f"coverage {cov0:.3f} -> {cov1:.3f}", flush=True)


# =====================================================================
# Chunked rollout of Chronos-2 (GPU)
# =====================================================================

def run_chunked(chunk, feedback, H=720, n_test=150, S=100, seed=0, max_windows=None, out_dir=None,
                windows_per_call=1, data="heldout"):
    import torch
    import bench
    from chronos import BaseChronosPipeline
    DEV = bench.DEV
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=str(DEV))
    levels = [round(float(x), 4) for x in pipe.quantiles]
    i50 = levels.index(0.5)
    lv = torch.tensor(levels, device=DEV, dtype=torch.float32)

    def predict(ctx, h):                                                  # (B, L) -> (B, h, K)
        out = []
        for i in range(0, ctx.shape[0], 500):
            q, _ = pipe.predict_quantiles(ctx[i:i + 500, None].cpu(), prediction_length=h, quantile_levels=levels)
            out.append(torch.stack([t[0] for t in q]).to(DEV).float())
        return torch.sort(torch.cat(out), -1).values

    def draw(q, gen):                                                     # one draw per step, (B, h, K) -> (B, h)
        lo = q[..., :1] - (q[..., 1:2] - q[..., :1]) * lv[0] / (lv[1] - lv[0])
        hi = q[..., -1:] + (q[..., -1:] - q[..., -2:-1]) * (1 - lv[-1]) / (lv[-1] - lv[-2])
        qq = torch.cat([lo, q, hi], -1)
        ll = torch.cat([torch.zeros(1, device=DEV), lv, torch.ones(1, device=DEV)])
        u = torch.rand(q.shape[:2], device=DEV, generator=gen)
        j = torch.searchsorted(ll, u.contiguous()).clamp(1, len(ll) - 1)
        v0, v1 = qq.gather(2, (j - 1)[..., None])[..., 0], qq.gather(2, j[..., None])[..., 0]
        return v0 + (u - ll[j - 1]) / (ll[j] - ll[j - 1]) * (v1 - v0)

    setting = f"chunk{chunk}{feedback}"
    for ds, d in bench.prepare(data, H, n_test).items():
        if max_windows:
            d = bench.subset(d, max_windows)
        ctx = d["ctx"][:, -bench.LF:]
        N = ctx.shape[0]
        t0 = time.perf_counter()
        with torch.no_grad():
            if feedback == "median":
                hist, qs = ctx.clone(), []
                for start in range(0, H, chunk):
                    q = predict(hist[:, -bench.LF:], min(chunk, H - start))
                    qs.append(q)
                    hist = torch.cat([hist, q[..., i50]], 1)
                kind, arr = "quantiles", torch.cat(qs, 1)
            else:
                gen = torch.Generator(device=DEV).manual_seed(seed)
                res = []
                for n0 in range(0, N, windows_per_call):
                    c = ctx[n0:n0 + windows_per_call]
                    hist = c.repeat_interleave(S, 0)                     # (B * S, L): each path its own series
                    parts = []
                    for start in range(0, H, chunk):
                        x = draw(predict(hist[:, -bench.LF:], min(chunk, H - start)), gen)
                        parts.append(x)
                        hist = torch.cat([hist, x], 1)
                    res.append(torch.cat(parts, 1).reshape(c.shape[0], S, H))
                kind, arr = "samples", torch.cat(res)
        el = time.perf_counter() - t0
        gen = torch.Generator(device=DEV).manual_seed(7 + seed)
        win, _ = bench.window_scores(kind, arr, levels, d, gen)
        meta = dict(context=bench.LF, calls=math.ceil(H / chunk), time_s=el, chunk=chunk, feedback=feedback,
                    season=int(d["season"]), steps_per_week=int(d["scale_win"]), n_series=int(d["n_series"]),
                    levels_native=kind == "quantiles")
        rev2.write_forecast(ds, H, "chronos2", seed, kind, arr, levels, d,
                            pit=win["pit"] if kind == "samples" else None, setting=setting, meta=meta, out_dir=out_dir)
        print(f"[chunked] {ds:8s} H={H} {setting}: ql={win['ql'].mean():.3f} cov80={win['cov80'].mean():.3f} "
              f"crps={win['crps'].mean():.3f} t={el:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["chunked", "copula", "spread"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS24)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--feedback", choices=["median", "paths"], default="median")
    ap.add_argument("--horizon", type=int, default=720)
    ap.add_argument("--n_test", type=int, default=150)
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--max_windows", type=int, default=None, help="smoke tests")
    ap.add_argument("--windows_per_call", type=int, default=1, help="chunked paths: windows (x100 series) per call")
    ap.add_argument("--outdir", default=None, help="directory of the rev2 files (default results/rev2)")
    a = ap.parse_args()
    if a.cmd == "chunked":
        run_chunked(a.chunk, a.feedback, a.horizon, a.n_test, a.paths, max_windows=a.max_windows, out_dir=a.outdir,
                    windows_per_call=a.windows_per_call)
    elif a.cmd == "copula":
        run_copula(a.datasets, a.models or QUANTILE_MODELS, a.paths, a.outdir)
    else:
        run_spread(a.datasets, a.models or SAMPLE_MODELS, a.outdir)
    print("done", flush=True)


if __name__ == "__main__":
    main()
