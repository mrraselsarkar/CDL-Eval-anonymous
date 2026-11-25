"""Calendar weeks of the test windows and the cluster bootstrap used by the report scripts.

The series of one dataset share weather and calendar: the eight ERCOT regions, the five Australian states and the
862 Bay Area sensors are observed on the same dates. Test windows whose forecast origins fall in the same calendar
week are therefore dependent, also across series. The report scripts (bench_report.py, rank_stats.py,
robust_report.py) resample calendar weeks of the forecast origin, across all series of a dataset, instead of
blocks of one series and one week.

The stored per-window results contain the series index (sid) and a block index (origin // 168 or 336) but not the
origin date. `python cluster_stats.py` regenerates the test windows with the deterministic preparation code of
run.prepare (held-out data, as called by bench.prepare) and bench.prepare (AEMO 2026), checks that the
regenerated series and block indices equal the stored arrays of every result file, reads the timestamp of each
origin, and writes results/window_weeks.npz:

  <ds>|<H>|sid, <ds>|<H>|block   regenerated indices (equal to the stored ones)
  <ds>|<H>|origin                timestamp of the last context value (the forecast origin), ISO format
  <ds>|<H>|week                  calendar week of the origin (weeks run Monday to Sunday), as the number of days
                                 from 1970-01-01 to its Monday divided by 7

`python cluster_stats.py compare` prints the 95% intervals of the report tables with the old blocks (series x week)
and with calendar weeks, and their widths. It runs on the CPU.
"""

import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # never touch the GPU: CPU work only (an empty value is dropped on Windows)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WEEKS = os.path.join(RES, "window_weeks.npz")
SETTINGS = [("heldout", 24, 300), ("heldout", 192, 150), ("heldout", 720, 150), ("aemo", 24, 300)]
LONG = 2880                                                 # bench.LONG: context kept for the naive forecaster


# =====================================================================================================
# Regeneration of the test windows
# =====================================================================================================

def _heldout_origins(run, ds, H, n_test):
    """Origins of the held-out test windows, drawn exactly as run.prepare draws them when called from
    bench.prepare (run.H = H, run.N_TEST = n_test, run.LF = 2880). run.prepare does not return the origins, so its
    random draws are repeated here; the training windows are drawn first and consume the same random numbers."""
    import pandas as pd
    fname, season, scale_win = run.DATASETS[ds]
    df = pd.read_parquet(os.path.join(HERE, "data", fname))
    series = run.load_series(fname)
    rng = np.random.default_rng(0)
    train, test = [], []
    for i, s in enumerate(series):
        n = len(s)
        tr_end, te_start = int(0.7 * n), int(0.8 * n)
        train.append((i, tr_end))
        test.append((i, max(te_start, LONG), n - H))
    lens = np.array([t[1] - (run.LS + H) for t in train], dtype=float)
    pick = rng.choice(len(train), size=run.sim.CFG["n_train"], p=lens / lens.sum())
    for i in pick:                                          # training windows: same draws as run.prepare
        rng.integers(0, train[i][1] - (run.LS + H))
    tl = np.array([t[2] - t[1] for t in test], dtype=float)
    pick = rng.choice(len(test), size=n_test, p=tl / tl.sum())
    sid, orig, stamp = [], [], []
    for i in pick:
        o = int(rng.integers(test[i][1], test[i][2]))
        sid.append(int(i))
        orig.append(o)
        stamp.append(pd.Timestamp(df["timestamp"].iloc[i][o - 1]))
    return np.array(sid), np.array(orig), stamp, scale_win, series


def _aemo_origins(bench, H, n_test):
    """Origins of the AEMO 2026 windows, drawn as in bench.prepare."""
    import pandas as pd
    series = bench.load_aemo()
    rng = np.random.default_rng(0)
    first = pd.Timestamp("2026-01-01")
    sid, orig, stamp = [], [], []
    for _ in range(n_test):
        i = int(rng.integers(len(series)))
        s = series[i]
        lo = max(LONG, int(np.searchsorted(s.index, first)))
        o = int(rng.integers(lo, len(s) - H))
        sid.append(i)
        orig.append(o)
        stamp.append(s.index[o - 1])
    return np.array(sid), np.array(orig), stamp, 336, series


def week_number(stamps):
    """Calendar week (Monday to Sunday) of each timestamp: days from 1970-01-01 to its Monday, divided by 7."""
    import pandas as pd
    t = pd.DatetimeIndex(stamps).normalize()
    monday = t - pd.to_timedelta(t.dayofweek, unit="D")
    return ((monday - pd.Timestamp("1970-01-05")).days // 7).to_numpy().astype(np.int64)   # 1970-01-05: a Monday


def stored_ids():
    """sid and block arrays of every stored result file, keyed by '<ds>|<H>'; all files must agree."""
    out = {}
    files = sorted(glob.glob(os.path.join(RES, "bench_*_perwindow.npz")))
    for f in files:
        z = np.load(f)
        for k in z.files:
            if k.endswith("|sid") or k.endswith("|block"):
                key, what = k.rsplit("|", 1)
                prev = out.setdefault(key, {}).get(what)
                assert prev is None or np.array_equal(prev, z[k]), f"{k} differs between result files"
                out[key][what] = z[k]
    for f in ["robust_perwindow.npz", "real_fm_perwindow.npz"]:      # first-round files: H = 24, same windows
        z = np.load(os.path.join(RES, f))
        for k in z.files:
            if k.endswith("|sid") or k.endswith("|origin_block"):
                ds, what = k.split("|")
                what = "block" if what == "origin_block" else what
                assert np.array_equal(out[f"{ds}|24"][what], z[k]), f"{f}: {k} differs from the bench files"
    return out


def regenerate():
    import torch
    import run
    import bench
    assert not torch.cuda.is_available(), "this script must not use the GPU"
    stored = stored_ids()
    out, summary, origs = {}, {}, {}
    for data, H, n_test in SETTINGS:
        names = list(run.DATASETS) if data == "heldout" else ["aemo"]
        for ds in names:
            key = f"{ds}|{H}"
            if key not in stored:
                continue
            if data == "heldout":
                sid, orig, stamp, scale_win, series = _heldout_origins(run, ds, H, n_test)
            else:
                sid, orig, stamp, scale_win, series = _aemo_origins(bench, H, n_test)
            block = orig // scale_win
            ok = np.array_equal(sid, stored[key]["sid"]) and np.array_equal(block, stored[key]["block"])
            assert ok, f"{key}: regenerated windows differ from the stored ones"
            week = week_number(stamp)
            origs[key] = orig
            out[f"{key}|sid"], out[f"{key}|block"], out[f"{key}|week"] = sid, block, week
            out[f"{key}|origin"] = np.array([str(t) for t in stamp])
            nb = len(np.unique(sid.astype(np.int64) * 100000 + block))
            summary[key] = dict(windows=len(sid), series=int(len(np.unique(sid))), series_week_blocks=nb,
                                calendar_weeks=int(len(np.unique(week))),
                                first_origin=str(min(stamp)), last_origin=str(max(stamp)))
            print(f"{key:12s} windows={len(sid)} sid/block match stored arrays: {ok}; blocks (series x week) {nb}, "
                  f"calendar weeks {summary[key]['calendar_weeks']}, origins {min(stamp)} to {max(stamp)}", flush=True)
    # the regenerated 24-step held-out windows must also have the contexts that run.prepare returns
    run.H, run.N_TEST, run.LF = 24, 300, LONG
    for ds in run.DATASETS:
        d = run.prepare(ds)
        fname, _, _ = run.DATASETS[ds]
        series = run.load_series(fname)
        n = [len(s) for s in series]
        z = [(s - s[:int(0.7 * len(s))].mean()) / (s[:int(0.7 * len(s))].std() + 1e-8) for s in series]
        ctx = np.stack([z[i][o - LONG:o] for i, o in zip(out[f"{ds}|24|sid"], origs[f"{ds}|24"])])
        err = float(np.abs(ctx - d["ctx"].cpu().numpy()).max())
        assert err < 1e-4, f"{ds}: regenerated contexts differ from run.prepare ({err})"
        print(f"{ds}: contexts of the regenerated windows equal those of run.prepare (max diff {err:.1e}); "
              f"series lengths {min(n)}-{max(n)}", flush=True)
    np.savez(WEEKS, **out)
    with open(os.path.join(RES, "window_weeks.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", WEEKS)


# =====================================================================================================
# Cluster bootstrap
# =====================================================================================================

_WEEKS = None


def weeks(pw, ds, h):
    """Calendar-week cluster of each test window of dataset ds at horizon h. Checks that the windows of the
    per-window results pw (keys '<ds>|<h>|sid' and '|block', or '<ds>|sid' and '|origin_block') are those of
    results/window_weeks.npz."""
    global _WEEKS
    if _WEEKS is None:
        if not os.path.exists(WEEKS):
            sys.exit("results/window_weeks.npz is missing: run  python cluster_stats.py  first")
        _WEEKS = dict(np.load(WEEKS))
    key = f"{ds}|{h}"
    sid = pw[f"{key}|sid"] if f"{key}|sid" in pw else pw[f"{ds}|sid"]
    block = pw[f"{key}|block"] if f"{key}|block" in pw else pw[f"{ds}|origin_block"]
    assert np.array_equal(sid, _WEEKS[f"{key}|sid"]) and np.array_equal(block, _WEEKS[f"{key}|block"]), \
        f"{key}: the per-window results do not belong to the windows of window_weeks.npz"
    return _WEEKS[f"{key}|week"]


def series_week_blocks(pw, ds, h):
    """The earlier clusters: one series and one week (kept for the comparison of interval widths)."""
    key = f"{ds}|{h}"
    sid = pw[f"{key}|sid"] if f"{key}|sid" in pw else pw[f"{ds}|sid"]
    block = pw[f"{key}|block"] if f"{key}|block" in pw else pw[f"{ds}|origin_block"]
    return sid.astype(np.int64) * 100000 + block.astype(np.int64)


def cluster_sums(cluster, *xs):
    """Per-cluster sums of each per-window array."""
    u, inv = np.unique(cluster, return_inverse=True)
    return [np.bincount(inv, x, len(u)) for x in xs]


def ratio_draws(cluster, xa, xb, B, rng):
    """Bootstrap draws of mean(xa) / mean(xb), resampling clusters with replacement."""
    sa, sb = cluster_sums(cluster, xa, xb)
    d = rng.integers(0, len(sa), (B, len(sa)))
    return sa[d].sum(1) / sb[d].sum(1)


# =====================================================================================================
# Old against new intervals
# =====================================================================================================

def compare():
    """95% intervals of Tables S3, S4 and S9 and Fig. 4(b) with the old and the new clusters (same resamples and
    random streams as the report scripts)."""
    import bench_report as br
    rows, pw = br.load()
    res = {}
    for kind in ["series_week", "week"]:
        br.CLUSTER = kind
        res[kind] = br.intervals(rows, pw)
    br.CLUSTER = "week"
    print(f"{'interval':58s} {'series x week':>24s} {'calendar week':>24s}  width ratio")
    ratios = {}
    for k, (est, lo, hi) in res["week"].items():
        e0, l0, h0 = res["series_week"][k]
        ratios.setdefault(k[0], []).append((hi - lo) / (h0 - l0))
        print(f"{' '.join(map(str, k)):58s} {est:+7.3f} [{l0:+7.3f},{h0:+7.3f}] {est:+7.3f} [{lo:+7.3f},{hi:+7.3f}]"
              f"  {(hi - lo) / (h0 - l0):5.2f}")
    for t, r in ratios.items():
        print(f"{t}: median width ratio {np.median(r):.2f} (range {min(r):.2f} to {max(r):.2f}, {len(r)} intervals)")
    return res


if __name__ == "__main__":
    if "compare" in sys.argv[1:]:
        compare()
    else:
        regenerate()
