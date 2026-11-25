"""Full forecasts of the second revision ("rev2" files): writer, reader, PIT and shared metrics.

One file per run:  results/rev2/<dataset>__H<H>__<model>__s<seed>[__<setting>].npz  with

  y        (N, H) float32   outcomes on the window scale of Eq. (3): (y - m) / sd, m and sd from the last week of context
  y_orig   (N, H) float32   outcomes in original units (MW, occupancy); y_orig = y * sd + m
  m, sd    (N,)   float64   window scale in original units
  paths    (N, S, H) float16  sample paths on the Eq. (3) scale (sample-based forecasts only)
  levels   (K,)   float32   quantile levels: native levels of quantile models, SAMPLE_LEVELS for sample paths
  q        (N, K, H) float32  quantiles on the Eq. (3) scale
  pit      (N, H) float32   randomized PIT, Eq. (5), for sample paths; for quantile models the PIT of the piecewise-linear
                            quantile function extended linearly to the levels 0 and 1 (as for the CRPS paths of the paper)
  sid, origin, block, week (N,)  series index, time stamp of the first forecast step (int64 ns), block id
                            (origin index // steps per week, as in bench.py), calendar week of the origin (ISO year*100+week)
  meta     JSON string: dataset, H, model, seed, context, setting, n_paths, kind, ...

The writer and reader need only numpy (the writer also accepts torch tensors), so the module can be imported in every
environment of the benchmark, including the Toto environment. scipy is imported only by the rank functions.
"""

import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV2 = os.path.join(HERE, "results", "rev2")
SAMPLE_LEVELS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
Q9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NAME_RE = re.compile(r"^(?P<dataset>[^_]+(?:_[^_]+)*)__H(?P<H>\d+)__(?P<model>.+?)__s(?P<seed>\d+)(?:__(?P<setting>.+))?\.npz$")


# =====================================================================
# File names
# =====================================================================

def fname(dataset, H, model, seed, setting=None, out_dir=None):
    base = f"{dataset}__H{int(H)}__{model}__s{int(seed)}" + (f"__{setting}" if setting else "") + ".npz"
    return os.path.join(out_dir or REV2, base)


def parse_name(path):
    """(dataset, H, model, seed, setting) of a rev2 file name, or None."""
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return m["dataset"], int(m["H"]), m["model"], int(m["seed"]), m["setting"]


def listing(dataset=None, H=None, model=None, setting="", out_dir=None):
    """Rev2 files that match. setting="" matches files without a setting, None matches any setting."""
    out = []
    for p in sorted(glob.glob(os.path.join(out_dir or REV2, "*.npz"))):
        k = parse_name(p)
        if k is None:
            continue
        ds, h, mo, seed, st = k
        if (dataset is None or ds == dataset) and (H is None or h == H) and (model is None or mo == model) and \
                (setting is None or (st or "") == setting):
            out.append((p, dict(dataset=ds, H=h, model=mo, seed=seed, setting=st)))
    return out


# =====================================================================
# Quantile functions and PIT
# =====================================================================

def extend_quantiles(q, levels):
    """Knots of the piecewise-linear quantile function, extended linearly to the levels 0 and 1 with the slope of the
    outer segments (as bench.quantile_samples). q: (..., K) sorted. Returns (values (..., K+2), levels (K+2,))."""
    lv = np.asarray(levels, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    lo = q[..., :1] - (q[..., 1:2] - q[..., :1]) * lv[0] / (lv[1] - lv[0])
    hi = q[..., -1:] + (q[..., -1:] - q[..., -2:-1]) * (1 - lv[-1]) / (lv[-1] - lv[-2])
    return np.concatenate([lo, q, hi], -1), np.concatenate([[0.0], lv, [1.0]])


def quantile_pit(q, levels, y):
    """PIT F(y) of the piecewise-linear CDF. q: (N, K, H) sorted along K, y: (N, H). Values outside the extended
    range get 0 or 1."""
    qq, ll = extend_quantiles(np.moveaxis(np.asarray(q), 1, -1), levels)        # (N, H, K+2)
    yy = np.asarray(y, dtype=np.float64)[..., None]
    n_knots = qq.shape[-1]
    j = (qq <= yy).sum(-1)                                                       # knots at or below y
    jc = np.clip(j, 1, n_knots - 1)
    q0 = np.take_along_axis(qq, (jc - 1)[..., None], -1)[..., 0]
    q1 = np.take_along_axis(qq, jc[..., None], -1)[..., 0]
    l0, l1 = ll[jc - 1], ll[jc]
    gap = q1 - q0
    f = np.where(gap > 0, l0 + (yy[..., 0] - q0) / np.where(gap > 0, gap, 1.0) * (l1 - l0), l1)
    f = np.where(j == 0, 0.0, np.where(j >= n_knots, 1.0, f))
    return np.clip(f, 0.0, 1.0).astype(np.float32)


def sample_quantiles(q, levels, u):
    """Inverse of the piecewise-linear CDF. q: (N, K, H) sorted, u: (N, S, H) in (0, 1). Returns (N, S, H)."""
    qq, ll = extend_quantiles(np.moveaxis(np.asarray(q), 1, -1), levels)        # (N, H, K+2)
    j = np.clip(np.searchsorted(ll, u), 1, len(ll) - 1)                          # (N, S, H)
    qe = np.broadcast_to(qq[:, None], (qq.shape[0], u.shape[1]) + qq.shape[1:])  # (N, S, H, K+2)
    v0 = np.take_along_axis(qe, (j - 1)[..., None], -1)[..., 0]
    v1 = np.take_along_axis(qe, j[..., None], -1)[..., 0]
    return v0 + (u - ll[j - 1]) / (ll[j] - ll[j - 1]) * (v1 - v0)


def randomized_pit(paths, y, rng):
    """Eq. (5): (#{s: x_s < y} + V) / (S + 1). paths (N, S, H), y (N, H)."""
    paths = np.asarray(paths, dtype=np.float32)
    below = (paths < np.asarray(y, dtype=np.float32)[:, None]).sum(1)
    return ((below + rng.random(below.shape)) / (paths.shape[1] + 1)).astype(np.float32)


def sample_level_quantiles(paths, levels=SAMPLE_LEVELS):
    """Sample quantiles (N, K, H) with linear interpolation (torch.quantile's default)."""
    return np.moveaxis(np.quantile(np.asarray(paths, dtype=np.float32), levels, axis=1), 0, 1).astype(np.float32)


# =====================================================================
# Writer and reader
# =====================================================================

def _np(a):
    if a is None:
        return None
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(a)


def iso_week(origin_ns):
    """ISO year * 100 + ISO week of int64 ns time stamps."""
    import pandas as pd
    iso = pd.DatetimeIndex(np.asarray(origin_ns, dtype="int64")).isocalendar()
    return (iso["year"].to_numpy() * 100 + iso["week"].to_numpy()).astype(np.int64)


def save(path, y, y_orig, m, sd, levels, q, pit, sid, origin, block, week, meta, paths=None):
    """Writes one rev2 file (all arrays on the Eq. (3) scale except y_orig, m and sd)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = dict(y=np.asarray(y, np.float32), y_orig=np.asarray(y_orig, np.float32),
               m=np.asarray(m, np.float64).reshape(-1), sd=np.asarray(sd, np.float64).reshape(-1),
               levels=np.asarray(levels, np.float32), q=np.asarray(q, np.float32), pit=np.asarray(pit, np.float32),
               sid=np.asarray(sid, np.int64), origin=np.asarray(origin, np.int64), block=np.asarray(block, np.int64),
               week=np.asarray(week, np.int64), meta=np.array(json.dumps(meta)))
    if paths is not None:
        arr["paths"] = np.asarray(paths, np.float16)
    N, H = arr["y"].shape
    assert arr["q"].shape == (N, len(arr["levels"]), H), arr["q"].shape
    assert arr["pit"].shape == (N, H) and arr["m"].shape == (N,)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arr)
    os.replace(tmp, path)
    return path


def from_forecast(kind, arr, levels, d, pit=None, rng=None):
    """Rev2 arrays from a forecast of bench.py.

    kind, arr, levels: output of a bench runner ("samples": (N, S, H) paths, "quantiles": (N, H, K)) on the scale of
    d["ctx"]; d: window dict of bench.prepare (needs fut, m, sc, mu_s, sd_s, sid, origin_ts, block, week); pit: the
    randomized PIT of bench.window_scores for sample paths (computed here with rng if None)."""
    fut, m, sc = _np(d["fut"]), _np(d["m"]), _np(d["sc"])                         # (N, H), (N, 1), (N, 1)
    mu_s, sd_s = np.asarray(d["mu_s"], np.float64), np.asarray(d["sd_s"], np.float64)
    y = (fut - m) / sc
    out = dict(y=y, y_orig=fut.astype(np.float64) * sd_s[:, None] + mu_s[:, None],
               m=m[:, 0].astype(np.float64) * sd_s + mu_s, sd=sc[:, 0].astype(np.float64) * sd_s,
               sid=d["sid"], origin=d["origin_ts"], block=d["block"], week=d["week"])
    a = _np(arr)
    if kind == "samples":
        x = (a - m[:, None]) / sc[:, None]
        out.update(paths=x, levels=np.array(SAMPLE_LEVELS, np.float32), q=sample_level_quantiles(x))
        out["pit"] = _np(pit) if pit is not None else randomized_pit(x, y, rng or np.random.default_rng(0))
    else:
        qn = (np.sort(a, -1) - m[:, :, None]) / sc[:, :, None]                    # (N, H, K)
        qn = np.moveaxis(qn, -1, 1)                                               # (N, K, H)
        out.update(levels=np.asarray(levels, np.float32), q=qn, pit=quantile_pit(qn, levels, y))
    return out


def write_forecast(dataset, H, model, seed, kind, arr, levels, d, pit=None, setting=None, meta=None, out_dir=None):
    a = from_forecast(kind, arr, levels, d, pit)
    mt = dict(dataset=dataset, H=int(H), model=model, seed=int(seed), setting=setting, kind=kind,
              n_paths=int(a["paths"].shape[1]) if "paths" in a else 0,
              scale="y = (y_orig - m) / sd; m, sd: mean and sd (+1e-3 on the standardized scale) of the last week of context",
              origin="time stamp of the first forecast step")
    mt.update(meta or {})
    return save(fname(dataset, H, model, seed, setting, out_dir), meta=mt, **a)


def load(path, paths=True):
    """Dict of arrays of a rev2 file; meta parsed. paths=False skips the (large) paths array."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if paths or k != "paths"}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def load_run(dataset, H, model, setting="", seeds=None, out_dir=None, paths=True):
    """All seeds of one run as a list of dicts (sorted by seed)."""
    res = []
    for p, k in listing(dataset, H, model, setting, out_dir):
        if seeds is None or k["seed"] in seeds:
            res.append(load(p, paths))
    return sorted(res, key=lambda r: r["meta"]["seed"])


# =====================================================================
# Scores
# =====================================================================

def level_index(levels, wanted, tol=1e-4):
    """Indices of the wanted levels in `levels`, or None if one is missing."""
    lv = np.asarray(levels, dtype=np.float64)
    idx = []
    for w in wanted:
        j = int(np.argmin(np.abs(lv - w)))
        if abs(lv[j] - w) > tol:
            return None
        idx.append(j)
    return idx


def quantile_loss(q, levels, y, at=Q9):
    """Per-window QL of Eq. (4) at the levels `at` (as bench.window_scores). q (N, K, H), y (N, H). Returns (N,)."""
    idx = level_index(levels, at)
    qa = np.asarray(q, np.float64)[:, idx]                                     # (N, k, H)
    lv = np.asarray(at, np.float64)[None, :, None]
    diff = np.asarray(y, np.float64)[:, None] - qa
    return (2 * np.maximum(lv * diff, (lv - 1) * diff)).mean((1, 2))


def quantile_loss_lead(q, levels, y, at=Q9):
    """QL per window and lead time, (N, H)."""
    idx = level_index(levels, at)
    qa = np.asarray(q, np.float64)[:, idx]
    lv = np.asarray(at, np.float64)[None, :, None]
    diff = np.asarray(y, np.float64)[:, None] - qa
    return (2 * np.maximum(lv * diff, (lv - 1) * diff)).mean(1)


def crps_paths(paths, y):
    """CRPS per window and step from paths (Eq. 6), (N, H)."""
    x = np.sort(np.asarray(paths, np.float64), 1)
    S = x.shape[1]
    t1 = np.abs(x - np.asarray(y, np.float64)[:, None]).mean(1)
    w = (2 * np.arange(1, S + 1) - S - 1)[None, :, None]
    return t1 - 0.5 * 2.0 / (S * S) * (x * w).sum(1)


def pit_below_count(pit, S):
    """Number of paths below the outcome, recovered from the randomized PIT of S paths."""
    return np.clip(np.floor(np.asarray(pit, np.float64) * (S + 1) + 1e-9), 0, S).astype(np.int64)


def _ranks(a, axis):
    from scipy.stats import rankdata
    return rankdata(a, axis=axis)


def spearman_cols(r1, r2):
    """Pearson correlation of rank columns, along axis 0. r1, r2: (n, H') already ranked. Returns (H',)."""
    a = r1 - r1.mean(0)
    b = r2 - r2.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.where(den > 0, (a * b).sum(0) / np.where(den > 0, den, 1), np.nan)


def rho_model_windows(paths, lags):
    """Per-window Spearman correlation of steps h and h+lag across the paths, averaged over h (Eq. 10 per window).
    Ties (float16 storage) get average ranks. Returns {lag: (N,)}."""
    r = _ranks(np.asarray(paths, np.float32), axis=1)                             # (N, S, H)
    r = r - r.mean(1, keepdims=True)
    nrm = np.sqrt((r * r).sum(1))                                               # (N, H)
    out = {}
    for lag in lags:
        if lag >= r.shape[2]:
            continue
        num = (r[:, :, :-lag] * r[:, :, lag:]).sum(1)
        den = nrm[:, :-lag] * nrm[:, lag:]
        out[lag] = np.nanmean(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), 1)
    return out


def rho_obs(pit, lags, ranks=None, keep=None):
    """Eq. (11) at several lags: Spearman correlation across windows of the PIT values of steps h and h+lag,
    averaged over h. keep: optional (N, H) boolean mask; a pair is used only if both values are kept (the ranks are
    then computed per pair of steps from the kept windows). Returns {lag: value}."""
    u = np.asarray(pit, np.float64)
    out = {}
    if keep is None:
        r = _ranks(u, axis=0) if ranks is None else ranks
        for lag in lags:
            if lag < u.shape[1]:
                out[lag] = float(np.nanmean(spearman_cols(r[:, :-lag], r[:, lag:])))
        return out
    from scipy.stats import spearmanr
    for lag in lags:
        if lag >= u.shape[1]:
            continue
        vals = []
        for h in range(u.shape[1] - lag):
            k = keep[:, h] & keep[:, h + lag]
            if k.sum() > 10:
                vals.append(spearmanr(u[k, h], u[k, h + lag])[0])
        out[lag] = float(np.nanmean(vals)) if vals else float("nan")
    return out


# =====================================================================
# Cluster bootstrap
# =====================================================================

def cluster_index(ids):
    """List of index arrays, one per cluster id (e.g. week)."""
    ids = np.asarray(ids)
    order = np.argsort(ids, kind="stable")
    uniq, start = np.unique(ids[order], return_index=True)
    return np.split(order, start[1:])


def cluster_draws(ids, B, rng):
    """B bootstrap index arrays that resample clusters with replacement."""
    groups = cluster_index(ids)
    k = len(groups)
    for _ in range(B):
        pick = rng.integers(0, k, k)
        yield np.concatenate([groups[i] for i in pick])


def interval(values, level=0.95):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    a = (1 - level) / 2
    return [float(np.quantile(v, a)), float(np.quantile(v, 1 - a))]
