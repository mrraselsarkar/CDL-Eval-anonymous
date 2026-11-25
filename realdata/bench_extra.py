"""Additional baselines of the second revision, on the test windows of bench.py.

  ets              statsmodels ETSModel: additive error, additive damped trend, additive seasonality (period 24
                   for hourly and 48 for half-hourly data), initial states estimated; fitted to the 512 context
                   steps of each window; 100 paths simulated from the end of the context. If a fit fails or
                   gives non-finite paths, the window falls back to (A,N,A) and then to (A,N,N); the meta field
                   of the result file records how often. CPU only.
  deepar           GluonTS (PyTorch) DeepAR with a Student-t output, trained per dataset on the first 70% of
                   each series (original units, mean scaling), context length 512, GluonTS default lags and time
                   features, 2 LSTM layers of 40 units, 50 epochs x 100 batches of 64 windows, Adam with learning
                   rate 1e-3 decayed to 0 by a cosine schedule, seed 0; 100 sample paths from the 2,880 steps of
                   history that bench.py keeps for each window. (Version v2: with GluonTS's constant learning rate the
                   final weights had a one-step bias of ~1 predictive sd that compounded over the horizon.)
  lagllama         Lag-Llama (official checkpoint time-series-foundation-models/Lag-Llama), zero-shot, context
                   length 32 (its pretraining context, the recommended starting point), no RoPE scaling.
  lagllama_ctx512  the same model with context length 512 and linear RoPE scaling (factor (512 + H) / 32), as
                   its README recommends for contexts longer than 32; written as model lagllama with setting ctx512
                   (file suffix __ctx512, per-window key lagllama@ctx512), the convention of bench.py.
                   Both read the last context + 1,092 steps (their largest lag) of the window history
                   (--lag_history 512: only the 512 context steps; setting suffix hist512).
  small512_ar      AR forecaster with Gaussian head and full-window flow-matching forecaster of
  small512_flow    ../simulation/sim.py, trained as run.py trains them (16,384 windows from the first 70% of each
                   series, 3,000 steps of batch 256, three seeds) with 512 context steps instead of 48.

Trained models forecast the AEMO data (and the other datasets of the same source) with the model trained on the
held-out dataset of the same source (aemo, aemo_mid, ausel_season: ausel; ercot2026: ercot), because every
AEMO value lies in or near the test period.

Outputs
  results/rev2/<dataset>__H<H>__<model>__s<seed>[__<setting>].npz   written by rev2.write_forecast (rev2.py)
  results/bench_extra_perwindow.npz                     per-window scores, key style of bench_main_perwindow.npz
  results/bench_extra.json                              summary rows, format of bench.py
Scores are computed with bench.window_scores on windows from bench.prepare; the sid and block arrays are
checked against the stored per-window files of bench.py.

Environment: requirements/baselines.txt (torch 2.7.0+cu128, gluonts 0.14.4, statsmodels 0.15), and the Lag-Llama
repository in ../third_party/lag-llama or in the folder named by the environment variable LAGLLAMA_DIR.

Usage
  python bench_extra.py --cpu --models ets --data ercot traffic ausel aemo --horizon 24
  python bench_extra.py --cpu --models ets --data ercot traffic ausel --horizon 192 --n_test 150 --seeds 0
  python bench_extra.py --models deepar lagllama lagllama_ctx512 --data ercot traffic ausel aemo --horizon 24
  python bench_extra.py --models small512_ar small512_flow --data ercot traffic ausel aemo --horizon 24
  python bench_extra.py --cpu --smoke --limit 4 --models ... --data ... --out <dir>   tiny budgets, other folder
  python bench_extra.py --report                                                    table of the stored rows
"""

import sys
import os

if "--log" in sys.argv[:-1]:                  # all output of Python (and of its warnings and loggers) to a file
    _log_file = open(sys.argv[sys.argv.index("--log") + 1], "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = _log_file
    import faulthandler
    faulthandler.enable(file=_log_file)
if "--cpu" in sys.argv:                       # must precede the import of torch (directly or through bench)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# The Lag-Llama checkpoint and the Lightning checkpoints of GluonTS store GluonTS objects in their
# hyperparameters; torch >= 2.6 refuses to unpickle them unless weights_only=False.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LAGLLAMA_DIR = os.environ.get("LAGLLAMA_DIR", os.path.join(HERE, "..", "third_party", "lag-llama"))
LAGLLAMA_REPO = "time-series-foundation-models/Lag-Llama"
LEVELS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
S = 100                                   # paths per window
LCTX = 512                                # context steps (bench.LF)
HELDOUT = ["ercot", "traffic", "ausel"]
TRAIN_SRC = {"ercot": "ercot", "traffic": "traffic", "ausel": "ausel", "aemo": "ausel", "aemo_mid": "ausel",
             "ausel_season": "ausel", "ercot2026": "ercot"}
MODELS = ["ets", "deepar", "lagllama", "lagllama_ctx512", "small512_ar", "small512_flow"]
# command-line name -> (model name, setting) of the result files; settings follow bench.py ("ctx512": context 512)
OUT_NAME = {"ets": ("ets", None), "deepar": ("deepar", None), "lagllama": ("lagllama", None),
            "lagllama_ctx512": ("lagllama", "ctx512"), "small512_ar": ("small512_ar", None),
            "small512_flow": ("small512_flow", None)}
# lr_schedule "cosine": the learning rate decays from lr to 0 over the 5,000 training steps. With the constant rate of
# GluonTS (its ReduceLROnPlateau never fired within 50 epochs) the final weights carried a one-step bias of about one
# predictive sd (+1.8% on ERCOT) that compounds through the lagged inputs over the horizon (+30% at lead 24).
# version: suffix of the stored predictors, so that predictors of the constant-rate version are never reused.
DEEPAR = dict(context_length=512, num_layers=2, hidden_size=40, dropout_rate=0.1, lr=1e-3, lr_schedule="cosine",
              batch_size=64, num_batches_per_epoch=100, max_epochs=50, shuffle_buffer_length=1024, train_frac=0.7,
              version="v2")
ETS_SPECS = [("AAdA", dict(trend="add", damped_trend=True, seasonal="add")),
             ("ANA", dict(trend=None, seasonal="add")),
             ("ANN", dict(trend=None, seasonal=None))]


def log(*a):
    print("[extra]", *a, flush=True)


# =====================================================================
# bench.py: windows and scores (imported, never re-implemented)
# =====================================================================

_BENCH = None


def bench():
    """bench.py, imported lazily; another agent may be editing it, so a failed import is retried."""
    global _BENCH
    if _BENCH is None:
        for k in range(8):
            try:
                import bench as b
                _BENCH = b
                break
            except Exception as e:                                   # noqa: BLE001
                if k == 7:
                    raise
                log(f"import bench failed ({type(e).__name__}: {e}); retrying in 3 minutes")
                time.sleep(180)
    return _BENCH


_SETS = {}


def get_windows(ds, H, n_test):
    """Test windows of bench.prepare: the three held-out datasets come from prepare('heldout', ...), every other
    dataset from prepare(<name>, ...)."""
    b = bench()
    grp = "heldout" if ds in HELDOUT else ds
    key = (grp, H, n_test)
    if key not in _SETS:
        last = None
        for k in range(4):
            try:
                _SETS[key] = b.prepare(grp, H, n_test)
                break
            except Exception as e:                                   # noqa: BLE001
                last = e
                log(f"bench.prepare({grp!r}, {H}, {n_test}) failed ({type(e).__name__}: {e}); retry in 3 minutes")
                time.sleep(180)
                import importlib
                b = importlib.reload(b)
        else:
            raise last
    sets = _SETS[key]
    if ds not in sets:
        raise KeyError(f"bench.prepare({grp!r}) returned {list(sets)}, not {ds}")
    return sets[ds]


def verify_windows(ds, H, d):
    """sid and block must equal those stored by bench.py (and by any rev2 file of another model)."""
    sid, blk = np.asarray(d["sid"]), np.asarray(d["block"])
    checked = []
    for f in sorted(glob.glob(os.path.join(RES, "bench_*_perwindow.npz"))):
        if os.path.basename(f).startswith("bench_extra"):
            continue
        z = np.load(f)
        if f"{ds}|{H}|sid" in z.files:
            a, bb = z[f"{ds}|{H}|sid"], z[f"{ds}|{H}|block"]
            if a.shape != sid.shape or not (np.array_equal(a, sid) and np.array_equal(bb, blk)):
                raise RuntimeError(f"{ds} H={H}: windows differ from {os.path.basename(f)}")
            checked.append(os.path.basename(f))
    mine = {"ets", "deepar", "lagllama", "lagllama_ctx512", "small512_ar", "small512_flow"}
    for f in sorted(glob.glob(os.path.join(RES, "rev2", f"{ds}__H{H}__*__s*.npz"))):
        if os.path.basename(f).split("__")[2] in mine:
            continue
        try:
            z = np.load(f)
            if "sid" in z.files and "block" in z.files and z["sid"].shape == sid.shape:
                if not (np.array_equal(z["sid"], sid) and np.array_equal(z["block"], blk)):
                    raise RuntimeError(f"{ds} H={H}: windows differ from rev2/{os.path.basename(f)}")
                checked.append("rev2/" + os.path.basename(f))
        except RuntimeError:
            raise
        except Exception:                                            # noqa: BLE001  (file being written)
            pass
    return checked


# =====================================================================
# Series in original units and the origin of each window
# =====================================================================

def load_series(ds):
    """Series of a dataset in original units with time stamps, in the series order of bench.prepare."""
    import pandas as pd
    b = bench()
    run = b.run
    if ds in ("ercot", "traffic", "ausel", "ausel_season"):
        fname = run.DATASETS["ausel" if ds == "ausel_season" else ds][0]
        vals = run.load_series(fname)                               # the values the windows are cut from
        ts = pd.read_parquet(os.path.join(HERE, "data", fname), columns=["timestamp"])["timestamp"]
        return [pd.Series(v, index=pd.DatetimeIndex(pd.to_datetime(t))) for v, t in zip(vals, ts)]
    if ds == "aemo":
        return b.load_aemo()
    for name in (f"load_{ds}", f"load_series_{ds}"):
        if hasattr(b, name):
            return getattr(b, name)()
    import fresh_data
    if ds == "aemo_mid":
        return fresh_data.load_aemo_mid()
    if ds == "ercot2026":
        return fresh_data.load_ercot2026()
    raise KeyError(f"no series loader for {ds}")


def _match(v, ctx, fut, cands, tail=48):
    """Origin o (first forecast step) among cands with ctx = (v[o-L:o] - mu) / sd; returns (o, mu, sd, err)."""
    L, H = len(ctx), len(fut)
    cands = cands[(cands >= L) & (cands + H <= len(v))]
    if len(cands) == 0:
        return None
    while tail < L and np.ptp(ctx[-tail:]) < 1e-9:                  # constant end of context (stuck sensor)
        tail = min(L, tail * 4)
    x = ctx[-tail:]
    xc = x - x.mean()
    den = float(xc @ xc)
    if den <= 0:
        return None
    win = np.lib.stride_tricks.sliding_window_view(v, tail)           # win[j] = v[j:j+tail]
    err = np.empty(len(cands))
    for s0 in range(0, len(cands), 4096):                             # chunks bound the memory of the full scan
        c = cands[s0:s0 + 4096]
        A = win[c - tail]                                             # (k, tail): v[c-tail:c]
        slope = (A - A.mean(1, keepdims=True)) @ xc / den
        icpt = A.mean(1) - slope * x.mean()
        e = np.abs(A - (icpt[:, None] + slope[:, None] * x)).max(1) / np.maximum(np.abs(slope), 1e-12)
        e[~(slope > 0)] = np.inf
        err[s0:s0 + 4096] = e
    best = None
    for j in np.argsort(err)[:5]:
        if not np.isfinite(err[j]):
            break
        o = int(cands[j])
        a = v[o - L:o]
        sd, mu = np.polyfit(ctx, a, 1)
        e_ctx = np.abs(a - (mu + sd * ctx)).max() / sd
        e_fut = np.abs((v[o:o + H] - mu) / sd - fut).max()
        e = max(e_ctx, e_fut)
        if best is None or e < best[3]:
            best = (o, mu, sd, e)
    return best


def attach_origins(d, ds, H, tol=2e-3):
    """Adds to d the series in original units and, for each window, the origin index (first forecast step), the
    affine map between bench's standardized values and the original units, the outcomes in original units, the time
    stamp and ISO week of the origin. Where bench.prepare already gives origin, mu_s, sd_s, origin_ts and week, they
    are kept and checked against the values recovered here from the raw series."""
    series = load_series(ds)
    ctx = d["ctx"].double().cpu().numpy()
    if np.isnan(ctx).any():
        raise ValueError("context with missing values")
    fut = d["fut"].double().cpu().numpy()
    N = ctx.shape[0]
    sw = int(d.get("scale_win", 336 if d["season"] == 48 else 168))
    sid, blk = np.asarray(d["sid"]), np.asarray(d["block"])
    org, mu, sd, err = np.zeros(N, np.int64), np.zeros(N), np.zeros(N), np.zeros(N)
    given = d.get("origin", d.get("origin_idx", None))
    for n in range(N):
        v = series[int(sid[n])].to_numpy(dtype=float)
        if given is not None:
            cands = np.array([int(given[n])])
        else:
            cands = np.arange(int(blk[n]) * sw, (int(blk[n]) + 1) * sw)
        best = _match(v, ctx[n], fut[n], cands)
        if best is None or best[3] > tol:                              # block definition differs: full scan
            best = _match(v, ctx[n], fut[n], np.arange(ctx.shape[1], len(v) - H + 1))
        if best is None or best[3] > tol:
            raise RuntimeError(f"{ds}: window {n} not found in series {sid[n]} (error {best and best[3]})")
        org[n], mu[n], sd[n], err[n] = best
    y_orig = np.stack([series[int(s)].to_numpy(dtype=float)[o:o + H] for s, o in zip(sid, org)])
    import pandas as pd
    ts = pd.DatetimeIndex([series[int(s)].index[o] for s, o in zip(sid, org)]).as_unit("ns").asi8.astype(np.int64)
    iso = pd.DatetimeIndex(ts).isocalendar()
    week = (iso["year"].to_numpy().astype(np.int64) * 100 + iso["week"].to_numpy().astype(np.int64))
    # one scale per series: check that the windows of a series agree
    spread = max((np.ptp(sd[sid == s]) / sd[sid == s].mean()) for s in np.unique(sid))
    chk = {}
    if given is not None and not np.array_equal(np.asarray(given), org):
        raise RuntimeError(f"{ds}: origin of bench.prepare differs from the recovered origin")
    for key, mine in (("mu_s", mu), ("sd_s", sd)):
        if key in d:
            chk[key] = float(np.max(np.abs(np.asarray(d[key], float) - mine) / sd))
            if chk[key] > 1e-3:
                raise RuntimeError(f"{ds}: {key} of bench.prepare differs from the recovered scale ({chk[key]:.1e})")
        else:
            d[key] = mine
    for key, mine in (("origin_ts", ts), ("week", week)):
        if key in d:
            if not np.array_equal(np.asarray(d[key], np.int64), mine):
                raise RuntimeError(f"{ds}: {key} of bench.prepare differs from the series time stamps")
        else:
            d[key] = mine
    d.update(origin_idx=org, y_orig_raw=y_orig, series=series, match_err=float(err.max()),
             scale_spread=float(spread), scale_check=chk)
    return d


def subset(d, k):
    """First k windows (smoke tests)."""
    out = {}
    for key, v in d.items():
        if key == "series" or not hasattr(v, "shape") or getattr(v, "ndim", 0) == 0 or v.shape[0] != d["ctx"].shape[0]:
            out[key] = v
        else:
            out[key] = v[:k]
    return out


def to_z(d, x_orig):
    """Original units -> standardized scale of bench (per window)."""
    return (x_orig - d["mu_s"][:, None, None]) / d["sd_s"][:, None, None]


def freq_of(d):
    return "H" if d["season"] == 24 else "30min"


# =====================================================================
# ETS
# =====================================================================

def _ets_worker(args):
    """Fits ETS to one window (Eq. (3) scale) and simulates S paths for each seed; falls back to simpler models."""
    i, x, H, period, seeds, n_paths = args
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    t0 = time.perf_counter()
    errs = []
    for name, kw in ETS_SPECS:
        kw = dict(kw)
        if kw.get("seasonal"):
            kw["seasonal_periods"] = period
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ETSModel(x, error="add", initialization_method="estimated", **kw).fit(
                    disp=False, maxiter=5000, maxfun=100000)                # L-BFGS-B, numerical gradient
                out = np.stack([np.asarray(res.simulate(H, anchor="end", repetitions=n_paths,
                                                        rng=np.random.default_rng([seed, i]))).reshape(H, n_paths).T
                                for seed in seeds])
            if not np.all(np.isfinite(out)) or np.abs(out).max() > 1e3:
                raise FloatingPointError("non-finite or exploding paths")
            rv = res.mle_retvals
            conv = bool(rv.get("converged", True)) if isinstance(rv, dict) else True
            par = {k: float(v) for k, v in zip(res.model.param_names, np.asarray(res.params))
                   if k.startswith(("smoothing", "damping"))}
            return dict(i=i, spec=name, converged=conv, paths=out.astype(np.float32), time=time.perf_counter() - t0,
                        errors=errs, params=par)
        except Exception as e:                                       # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__}: {str(e)[:150]}")
    raise RuntimeError(f"window {i}: every ETS model failed: {errs}")


def run_ets(d, H, seeds, workers):
    import multiprocessing as mp
    import torch
    ctx = d["ctx"][:, -LCTX:]
    x = ((ctx - d["m"]) / d["sc"]).double().cpu().numpy()
    jobs = [(i, x[i], H, int(d["season"]), list(seeds), S) for i in range(x.shape[0])]
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"                                          # inherited by the workers
    t0 = time.perf_counter()
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = sorted(pool.imap_unordered(_ets_worker, jobs, chunksize=2), key=lambda r: r["i"])
    else:
        res = [_ets_worker(j) for j in jobs]
    el = time.perf_counter() - t0
    paths = np.stack([r["paths"] for r in res], 1)                  # (n_seeds, N, S, H), Eq. (3) scale
    specs = [r["spec"] for r in res]
    phi = [r["params"].get("damping_trend") for r in res if "damping_trend" in r["params"]]
    info = dict(spec_counts={n: specs.count(n) for n, _ in ETS_SPECS}, n_fallback=sum(s != "AAdA" for s in specs),
                n_not_converged=sum(not r["converged"] for r in res),
                fallback_windows=[int(r["i"]) for r in res if r["spec"] != "AAdA"],
                fallback_errors=[r["errors"] for r in res if r["errors"]][:20],
                fit_time_cpu_s=float(sum(r["time"] for r in res)), wall_s=el, workers=workers,
                phi_median=float(np.median(phi)) if phi else None,
                phi_at_lower_bound=int(sum(p <= 0.8 + 1e-6 for p in phi)) if phi else None,
                settings="ETSModel(error='add', trend='add', damped_trend=True, seasonal='add', seasonal_periods="
                         f"{int(d['season'])}, initialization_method='estimated').fit(maxiter=5000, maxfun=100000); "
                         "simulate(H, anchor='end', repetitions=100); fitted on the 512 context steps on the "
                         "Eq. (3) scale; fallbacks (A,N,A) then (A,N,N)")
    m, sc = d["m"], d["sc"]
    out = []
    for k, seed in enumerate(seeds):
        p = torch.tensor(paths[k], dtype=torch.float32, device=m.device)
        out.append((seed, p * sc[:, :, None] + m[:, :, None], dict(info), 0, el))
    return out


# =====================================================================
# DeepAR (GluonTS, PyTorch)
# =====================================================================

def _gluonts_quiet():
    import logging
    warnings.filterwarnings("ignore")
    for n in ("lightning", "lightning.pytorch", "pytorch_lightning", "gluonts"):
        logging.getLogger(n).setLevel(logging.ERROR)


def _cosine_lr(lr0, total_steps):
    """Lightning callback: learning rate lr0 * (1 + cos(pi * step / total_steps)) / 2, set before every training batch
    (it overrides the ReduceLROnPlateau scheduler of GluonTS)."""
    import math
    import lightning.pytorch as pl

    class CosineLR(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            f = 0.5 * (1.0 + math.cos(math.pi * min(trainer.global_step, total_steps) / total_steps))
            for opt in trainer.optimizers:
                for g in opt.param_groups:
                    g["lr"] = lr0 * f

    return CosineLR()


def deepar_predictor(src, H, freq, args, device):
    """Trains DeepAR on the first 70% of each series of dataset `src` (or loads the stored predictor)."""
    import pandas as pd
    import torch
    from pathlib import Path
    from gluonts.dataset.common import ListDataset
    from gluonts.torch.distributions import StudentTOutput
    from gluonts.torch.model.deepar import DeepAREstimator
    from gluonts.torch.model.predictor import PyTorchPredictor
    from gluonts.transform import ExpectedNumInstanceSampler
    _gluonts_quiet()
    cfg = dict(DEEPAR)
    if args.smoke:
        cfg.update(max_epochs=1, num_batches_per_epoch=3)
    tag = f"deepar_{src}_H{H}_s{args.train_seed}_{cfg['version']}" + ("_smoke" if args.smoke else "")
    path = Path(os.path.join(args.out, "rev2", "models_extra", tag))
    info = dict(train_source=src, freq=freq, **cfg, train_seed=args.train_seed, distr="StudentT",
                predictor_dir=str(path))
    if (path / "predictor.json").exists():
        pred = PyTorchPredictor.deserialize(path, device=device)
        meta_f = path / "extra_train_info.json"
        if meta_f.exists():
            info.update(json.load(open(meta_f)))
        info["loaded"] = True
        return pred, info
    series = load_series(src)
    train = []
    for s in series:
        n = int(cfg["train_frac"] * len(s))
        train.append({"start": pd.Period(s.index[0], freq=freq), "target": s.to_numpy(dtype=np.float32)[:n]})
    train_ds = ListDataset(train, freq=freq)
    from lightning.pytorch import seed_everything
    seed_everything(args.train_seed, workers=True)
    np.random.seed(args.train_seed)
    root = os.path.join(args.out, "rev2", "models_extra", tag + "_lightning")
    os.makedirs(root, exist_ok=True)
    if cfg["lr_schedule"] != "cosine":
        raise ValueError(f"lr_schedule {cfg['lr_schedule']!r}")
    callbacks = [_cosine_lr(cfg["lr"], cfg["max_epochs"] * cfg["num_batches_per_epoch"])]
    est = DeepAREstimator(
        freq=freq, prediction_length=H, context_length=cfg["context_length"], num_layers=cfg["num_layers"],
        hidden_size=cfg["hidden_size"], dropout_rate=cfg["dropout_rate"], lr=cfg["lr"],
        distr_output=StudentTOutput(), num_parallel_samples=S, batch_size=cfg["batch_size"],
        num_batches_per_epoch=cfg["num_batches_per_epoch"],
        train_sampler=ExpectedNumInstanceSampler(num_instances=max(1.0, 640.0 / len(series)), min_future=H),
        trainer_kwargs=dict(max_epochs=cfg["max_epochs"], accelerator="gpu" if device == "cuda" else "cpu",
                            devices=1, enable_progress_bar=False, logger=False, default_root_dir=root,
                            enable_model_summary=False, callbacks=callbacks))
    t0 = time.perf_counter()
    pred = est.train(train_ds, cache_data=True, shuffle_buffer_length=cfg["shuffle_buffer_length"])
    tt = time.perf_counter() - t0
    path.mkdir(parents=True, exist_ok=True)
    pred.serialize(path)
    lags = getattr(getattr(pred.prediction_net, "model", None), "lags_seq", None)
    if lags is None:
        from gluonts.time_feature import get_lags_for_frequency
        lags = get_lags_for_frequency(freq)
    lags = [int(v) for v in lags]
    extra = dict(train_time_s=tt, lags_seq=lags, max_lag=max(lags),
                 past_length=int(getattr(getattr(pred.prediction_net, "model", None), "_past_length", -1)),
                 n_series=len(series), train_steps_per_series=[int(cfg["train_frac"] * len(s)) for s in series],
                 n_params=int(sum(p.numel() for p in pred.prediction_net.parameters())))
    json.dump(extra, open(path / "extra_train_info.json", "w"))
    info.update(extra, loaded=False)
    log(f"DeepAR {tag}: trained in {tt:.0f}s, max lag {extra['max_lag']}")
    return pred, info


def run_deepar(d, ds, H, seeds, args):
    import pandas as pd
    import torch
    from gluonts.dataset.common import ListDataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    freq = freq_of(d)
    pred, info = deepar_predictor(TRAIN_SRC[ds], H, freq, args, dev)
    hist = d["ctx"].shape[1]                                          # 2,880 steps kept by bench.prepare
    entries = []
    for n in range(d["ctx"].shape[0]):
        s, o = d["series"][int(d["sid"][n])], int(d["origin_idx"][n])
        entries.append({"start": pd.Period(s.index[o - hist], freq=freq),
                        "target": s.to_numpy(dtype=np.float32)[o - hist:o]})
    test_ds = ListDataset(entries, freq=freq)
    info["history_steps"] = hist
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.perf_counter()
        fc = list(pred.predict(test_ds))
        el = time.perf_counter() - t0
        x = np.stack([f.samples for f in fc]).astype(np.float64)    # (N, S, H) original units
        p = torch.tensor(to_z(d, x), dtype=torch.float32, device=d["ctx"].device)
        yield seed, p, dict(info), H, el


# =====================================================================
# Lag-Llama
# =====================================================================

def run_lagllama(d, ds, H, seeds, args, ctx_len):
    import pandas as pd
    import torch
    if LAGLLAMA_DIR not in sys.path:
        sys.path.append(LAGLLAMA_DIR)                                # appended: realdata/run.py must win over its run.py
    _gluonts_quiet()
    from gluonts.dataset.common import ListDataset
    from huggingface_hub import hf_hub_download
    from lag_llama.gluon.estimator import LagLlamaEstimator
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = hf_hub_download(LAGLLAMA_REPO, "lag-llama.ckpt")
    hp = torch.load(ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
    a, data_map = hp["model_kwargs"], hp["data_id_to_name_map"]
    rope = None
    if ctx_len > a["context_length"]:
        rope = {"type": "linear", "factor": float(max(1.0, (ctx_len + H) / a["context_length"]))}
    def build(bs):
        e = LagLlamaEstimator(ckpt_path=ckpt, prediction_length=H, context_length=ctx_len,
                              input_size=a["input_size"], n_layer=a["n_layer"], n_embd_per_head=a["n_embd_per_head"],
                              n_head=a["n_head"], scaling=a["scaling"], time_feat=a["time_feat"], rope_scaling=rope,
                              batch_size=bs, num_parallel_samples=S, device=torch.device(dev))
        return e, e.create_predictor(e.create_transformation(), e.create_lightning_module())   # use_kv_cache=False

    bs = int(min(64, max(2, 9000 // (ctx_len + H))))                  # windows per batch (x 100 paths)
    est, predictor = build(bs)
    max_lag = int(max(est.lags_seq))
    hist = ctx_len + max_lag if args.lag_history == "full" else LCTX
    hist = min(hist, d["ctx"].shape[1])
    freq = freq_of(d)
    entries = []
    for n in range(d["ctx"].shape[0]):
        s, o = d["series"][int(d["sid"][n])], int(d["origin_idx"][n])
        entries.append({"start": pd.Period(s.index[o - hist], freq=freq),
                        "target": s.to_numpy(dtype=np.float32)[o - hist:o]})
    test_ds = ListDataset(entries, freq=freq)
    from importlib.metadata import version
    info = dict(checkpoint=f"{LAGLLAMA_REPO}/lag-llama.ckpt", checkpoint_path=ckpt, context_length=ctx_len,
                pretrain_context_length=int(a["context_length"]), rope_scaling=rope, history_steps=hist,
                max_lag=max_lag, n_lags=len(est.lags_seq), scaling=a["scaling"], time_feat=bool(a["time_feat"]),
                use_kv_cache=False, batch_size=bs, gluonts=version("gluonts"),
                lag_history=args.lag_history,
                pretraining_datasets=[v for k, v in data_map.items() if k >= 0],
                held_out_from_pretraining=[v for k, v in data_map.items() if k < 0])
    for seed in seeds:
        while True:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.perf_counter()
            try:
                fc = list(predictor.predict(test_ds))
                break
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                del predictor
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                log(f"Lag-Llama: out of GPU memory, batch of {bs} windows")
                est, predictor = build(bs)
                info["batch_size"] = bs
        el = time.perf_counter() - t0
        x = np.stack([f.samples for f in fc]).astype(np.float64)
        p = torch.tensor(to_z(d, x), dtype=torch.float32, device=d["ctx"].device)
        yield seed, p, dict(info), H, el


# =====================================================================
# Small forecasters of ../simulation/sim.py with 512 context steps
# =====================================================================

def small_model(name, src, seed, args):
    """AR (Gaussian head) or full-window flow matching, trained as in run.py with 512 context steps."""
    import torch
    b = bench()
    run, sim = b.run, b.sim
    sim.L = LCTX                                                      # the classes read sim.L when built and trained
    tag = f"small512_{name}_{src}_s{seed}" + ("_smoke" if args.smoke else "")
    path = os.path.join(args.out, "rev2", "models_extra", tag + ".pt")
    info = dict(train_source=src, context=LCTX, n_train=sim.CFG["n_train"], train_steps=sim.CFG["train_steps"],
                batch=sim.CFG["batch"], seed=seed, checkpoint=path)
    if os.path.exists(path):
        model = sim.CLASSES[name]().to(sim.DEV)
        model.load_state_dict(torch.load(path, map_location=sim.DEV, weights_only=True))
        model.eval()
        info["loaded"] = True
        return model, info
    # training windows of run.prepare with LS = 512 (its test windows are not used: bench.prepare gives them)
    old = (run.LS, run.H, run.N_TEST, run.LF)
    run.LS, run.H = LCTX, sim.H
    try:
        train = run.prepare(src)["train"]
    finally:
        run.LS, run.H, run.N_TEST, run.LF = old
    t0 = time.perf_counter()
    with torch.enable_grad():
        model = sim.train(name, train, seed, sim.CFG["train_steps"])
    tt = time.perf_counter() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    info.update(train_time_s=tt, train_windows=list(train.shape), loaded=False,
                n_params=int(sum(p.numel() for p in model.parameters())))
    log(f"{tag}: trained in {tt:.0f}s on {tuple(train.shape)} windows")
    return model, info


def run_small(d, ds, H, seeds, args, name):
    import torch
    sim = bench().sim
    if H != sim.H:
        raise ValueError(f"the small forecasters have a fixed horizon of {sim.H} steps")
    if args.smoke:
        sim.CFG.update(n_train=512, train_steps=30)
    for seed in seeds:
        model, info = small_model(name, TRAIN_SRC[ds], seed, args)
        ctx = d["ctx"][:, -LCTX:].to(sim.DEV)
        mu, sd = sim.inorm(ctx)
        torch.manual_seed(1234 + seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            xs = torch.cat([model.sample((ctx[i:i + 100] - mu[i:i + 100]) / sd[i:i + 100], S)
                            for i in range(0, ctx.shape[0], 100)])
        el = time.perf_counter() - t0
        calls = int(model.calls())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield seed, (xs * sd[:, None] + mu[:, None]).to(d["ctx"].device), info, calls, el


# =====================================================================
# Output: rev2 files, per-window scores, summary rows
# =====================================================================

class Lock:
    def __init__(self, path, stale=1800):
        self.path, self.stale = path, stale

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                if time.time() - t0 > 3600:
                    raise TimeoutError(self.path)
                time.sleep(0.5)

    def __exit__(self, *a):
        try:
            os.remove(self.path)
        except OSError:
            pass


def _savez(path, arrays):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _own_rev2_writer(path, d, paths_z, pit, meta):
    """Same keys as rev2.write_forecast; used only if rev2.py cannot be imported."""
    import torch
    m, sc = d["m"], d["sc"]
    y = ((d["fut"] - m) / sc).float()
    x = ((paths_z - m[:, :, None]) / sc[:, :, None]).float()
    q = torch.quantile(x, torch.tensor(LEVELS, device=x.device), dim=1).permute(1, 0, 2)
    mu_s, sd_s = np.asarray(d["mu_s"], np.float64), np.asarray(d["sd_s"], np.float64)
    arrays = dict(y=y.cpu().numpy(), y_orig=(d["fut"].double().cpu().numpy() * sd_s[:, None] + mu_s[:, None]),
                  m=m[:, 0].double().cpu().numpy() * sd_s + mu_s, sd=sc[:, 0].double().cpu().numpy() * sd_s,
                  paths=x.cpu().numpy().astype(np.float16), levels=np.array(LEVELS, dtype=np.float32),
                  q=q.cpu().numpy().astype(np.float32), pit=np.asarray(pit, dtype=np.float32),
                  sid=np.asarray(d["sid"], dtype=np.int64), origin=np.asarray(d["origin_ts"], dtype=np.int64),
                  block=np.asarray(d["block"], dtype=np.int64), week=np.asarray(d["week"], dtype=np.int64),
                  meta=np.array(json.dumps(meta)))
    arrays["y_orig"] = arrays["y_orig"].astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _savez(path, arrays)
    return path


def save_run(out_dir, ds, H, model, setting, seed, d, paths_z, win, lead, info, calls, el, checked):
    label = model if setting is None else f"{model}@{setting}"
    n = int(d["fut"].shape[0])
    from importlib.metadata import version
    meta = dict(dataset=ds, H=int(H), model=model, seed=int(seed), setting=setting, kind="samples", n_paths=S,
                context=LCTX, calls=int(calls), time_s=float(el), n_windows=n, season=int(d["season"]),
                steps_per_week=int(d.get("scale_win", 0)), n_series=int(d["n_series"]),
                script="realdata/bench_extra.py", created=time.strftime("%Y-%m-%d %H:%M:%S"),
                scale="y = (y_orig - m) / sd; m, sd: mean and sd (+1e-3 on the standardized scale) of the last week "
                      "of context",
                origin="time stamp of the first forecast step",
                pit="(#{s: path < y} + V) / (S + 1), V ~ U(0, 1), from bench.window_scores",
                windows_checked_against=checked, origin_match_error=d["match_err"],
                series_scale_spread=d["scale_spread"], scale_check=d.get("scale_check"),
                train_source=TRAIN_SRC.get(ds) if model in ("deepar", "small512_ar", "small512_flow") else None,
                versions={k: version(k) for k in ("torch", "numpy", "gluonts", "statsmodels")},
                device=str(paths_z.device), details=info)
    meta = json.loads(json.dumps(meta, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    rev2_dir = os.path.join(out_dir, "rev2")
    try:
        import rev2
        fn = rev2.write_forecast(ds, H, model, seed, "samples", paths_z, None, d, pit=win["pit"], setting=setting,
                                 meta=meta, out_dir=rev2_dir)
    except ImportError:
        base = f"{ds}__H{int(H)}__{model}__s{int(seed)}" + (f"__{setting}" if setting else "") + ".npz"
        fn = _own_rev2_writer(os.path.join(rev2_dir, base), d, paths_z, win["pit"], meta)
    # outcomes in original units of the file against the raw series (checks the scale and the origin)
    with np.load(fn) as z:
        rel = float(np.max(np.abs(z["y_orig"].astype(np.float64) - d["y_orig_raw"]) /
                           np.maximum(np.abs(d["y_orig_raw"]), np.asarray(d["sd_s"], np.float64)[:, None])))
        if rel > 1e-4:
            raise RuntimeError(f"{fn}: y_orig differs from the raw series (relative {rel:.1e})")
    # per-window scores and summary row (format of bench.py)
    row = {k: float(np.mean(v)) for k, v in win.items() if k != "pit"}
    ru = np.asarray(win["pit"]).argsort(0).argsort(0).astype(float)
    ru = (ru - ru.mean(0)) / ru.std(0)
    row["rho_obs"] = float((ru[:, :-1] * ru[:, 1:]).mean(0).mean())
    row.update(dataset=ds, horizon=H, model=model, seed=int(seed), kind="samples", calls=int(calls),
               time_s=float(el), n_test=n, lead_crps=np.asarray(lead["crps"]).tolist(),
               lead_ql=np.asarray(lead["ql"]).tolist(), source="bench_extra.py")
    if setting is not None:
        row.update(setting=setting, context=int(info.get("context_length", LCTX)))
    jpath = os.path.join(out_dir, "bench_extra.json")
    npath = os.path.join(out_dir, "bench_extra_perwindow.npz")
    with Lock(os.path.join(out_dir, "bench_extra.lock")):
        rows = json.load(open(jpath)) if os.path.exists(jpath) else []
        rows = [r for r in rows if not (r["dataset"] == ds and r["horizon"] == H and r["model"] == model and
                                        r["seed"] == seed and r.get("setting") == setting)] + [row]
        pw = dict(np.load(npath)) if os.path.exists(npath) else {}
        pw[f"{ds}|{H}|sid"], pw[f"{ds}|{H}|block"] = np.asarray(d["sid"]), np.asarray(d["block"])
        for k, v in win.items():
            pw[f"{ds}|{H}|{label}|{seed}|{k}"] = np.asarray(v)
        tmp = jpath + ".part"
        with open(tmp, "w") as f:
            json.dump(rows, f)
        os.replace(tmp, jpath)
        _savez(npath, pw)
    return row, fn


# =====================================================================
# Main
# =====================================================================

def forecast(model, d, ds, H, seeds, args):
    if model == "ets":
        return run_ets(d, H, seeds, args.workers)
    if model == "deepar":
        return run_deepar(d, ds, H, seeds, args)
    if model == "lagllama":
        return run_lagllama(d, ds, H, seeds, args, 32)
    if model == "lagllama_ctx512":
        return run_lagllama(d, ds, H, seeds, args, 512)
    if model in ("small512_ar", "small512_flow"):
        return run_small(d, ds, H, seeds, args, model.split("_")[1])
    raise ValueError(model)


def report(out_dir):
    rows = json.load(open(os.path.join(out_dir, "bench_extra.json")))
    lab = lambda r: r["model"] + (f"@{r['setting']}" if r.get("setting") else "")
    order = ["ercot", "traffic", "ausel", "aemo", "ausel_season", "aemo_mid", "ercot2026"]
    keys = sorted({(r["dataset"], r["horizon"], lab(r)) for r in rows},
                  key=lambda k: (k[1], order.index(k[0]) if k[0] in order else 99, k[0], k[2]))
    print(f"{'dataset':13s} {'H':>4s} {'model':18s} {'n':>2s} {'QL':>6s} {'cov80':>6s} {'cov90':>6s} {'CRPS':>6s} "
          f"{'ES':>7s} {'rho_m':>6s} {'rho_o':>6s} {'time_s':>7s}")
    for ds, H, m in keys:
        rs = [r for r in rows if (r["dataset"], r["horizon"], lab(r)) == (ds, H, m)]
        g = lambda k: float(np.mean([r[k] for r in rs]))
        print(f"{ds:13s} {H:4d} {m:18s} {len(rs):2d} {g('ql'):6.3f} {100 * g('cov80'):6.1f} {100 * g('cov90'):6.1f} "
              f"{g('crps'):6.3f} {g('es'):7.3f} {g('rho_model'):6.2f} {g('rho_obs'):6.2f} {g('time_s'):7.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", choices=MODELS)
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="sampling seeds (training seeds for small512_*); default 0 1 2 at H = 24, else 0")
    ap.add_argument("--train_seed", type=int, default=0, help="training seed of DeepAR")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 6), help="ETS processes")
    ap.add_argument("--lag_history", default="full", choices=["full", "512"],
                    help="Lag-Llama input: context + largest lag (full) or the 512 context steps")
    ap.add_argument("--limit", type=int, default=None, help="only the first k windows (smoke tests)")
    ap.add_argument("--smoke", action="store_true", help="tiny training budgets (use with --out and --limit)")
    ap.add_argument("--cpu", action="store_true", help="hide the GPU (CUDA_VISIBLE_DEVICES=-1)")
    ap.add_argument("--out", default=RES, help="folder for rev2/, bench_extra.json and the per-window file")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--log", default=None, help="append all output to this file")
    ap.add_argument("--require_gpu", action="store_true", help="stop if CUDA is not available")
    ap.add_argument("--skip_existing", action="store_true", help="skip runs whose rev2 file exists")
    a = ap.parse_args()
    if a.report:
        report(a.out)
        return
    if not a.models or not a.data:
        ap.error("--models and --data are required")
    if a.smoke and os.path.abspath(a.out) == os.path.abspath(RES):
        ap.error("--smoke writes to another folder: give --out")
    if a.lag_history != "full" and any(m.startswith("lagllama") for m in a.models):
        log("note: --lag_history 512 writes under the same model names")
    import torch
    log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv[1:])}")
    log(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, out: {a.out}")
    if a.require_gpu and not torch.cuda.is_available():
        log("CUDA is not available (--require_gpu): stopping")
        sys.exit(2)
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}, arch list {torch.cuda.get_arch_list()}")
    seeds = a.seeds if a.seeds is not None else ([0, 1, 2] if a.horizon == 24 else [0])
    b = bench()
    b.sim.ABAR = b.sim.cosine_abar(b.sim.CFG["k_train"])
    failures = []
    for ds in a.data:
        try:
            t0 = time.perf_counter()
            d = get_windows(ds, a.horizon, a.n_test)
            checked = verify_windows(ds, a.horizon, d)
            d = attach_origins(d, ds, a.horizon)
            log(f"{ds} H={a.horizon}: {d['ctx'].shape[0]} windows, checked against {checked or 'nothing stored'}; "
                f"origin match error {d['match_err']:.1e}, series scale spread {d['scale_spread']:.1e} "
                f"({time.perf_counter() - t0:.0f}s)")
            if a.limit:
                d = subset(d, a.limit)
        except Exception as e:                                       # noqa: BLE001
            traceback.print_exc()
            failures.append(f"{ds}: windows: {type(e).__name__}: {e}")
            continue
        for model in a.models:
            name, setting = OUT_NAME[model]
            if name == "lagllama" and a.lag_history != "full":
                setting = (setting + "_" if setting else "") + "hist512"
            label = name if setting is None else f"{name}@{setting}"
            todo = seeds
            if a.skip_existing:
                fn_of = lambda s: os.path.join(a.out, "rev2", f"{ds}__H{a.horizon}__{name}__s{s}"
                                               + (f"__{setting}" if setting else "") + ".npz")
                todo = [s for s in seeds if not os.path.exists(fn_of(s))]
                if len(todo) < len(seeds):
                    log(f"{ds} H={a.horizon} {label}: seeds {sorted(set(seeds) - set(todo))} exist, skipped")
                if not todo:
                    continue
            try:
                for seed, paths_z, info, calls, el in forecast(model, d, ds, a.horizon, todo, a):
                    gen = torch.Generator(device=d["ctx"].device).manual_seed(7 + seed)
                    res = b.window_scores("samples", paths_z, None, d, gen)
                    win, lead = res[0], res[1]
                    row, fn = save_run(a.out, ds, a.horizon, name, setting, seed, d, paths_z, win, lead, info, calls,
                                       el, checked)
                    print(f"[extra] {ds:8s} H={a.horizon:<4d} {label:16s} s{seed} crps={row['crps']:.3f} "
                          f"ql={row['ql']:.3f} es={row['es']:.3f} cov80={row['cov80']:.2f} cov90={row['cov90']:.2f} "
                          f"rho_model={row['rho_model']:.2f} rho_obs={row['rho_obs']:.2f} t={el:.0f}s "
                          f"-> {os.path.relpath(fn, HERE)}", flush=True)
                    if model == "ets" and seed == todo[0]:
                        log(f"ETS {ds} H={a.horizon}: models used {info['spec_counts']}, not converged "
                            f"{info['n_not_converged']}, median phi {info['phi_median']}, wall {info['wall_s']:.0f}s "
                            f"({info['workers']} processes)")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:                                   # noqa: BLE001
                traceback.print_exc()
                failures.append(f"{ds} {model}: {type(e).__name__}: {e}")
    if failures:
        log("FAILED: " + " | ".join(failures))
        sys.exit(1)
    log("done")


if __name__ == "__main__":
    main()
