"""Tables, figures and numbers of the second revision, from the rev2 files (CPU only).

Reads results/rev2/*.npz (rev2.py) and results/rev2/data_checks.json (fresh_data.py) and writes
  ../outputs/tables/rev2_*.tex     LaTeX tables (new files only)
  ../outputs/figures/rev2_*.pdf           figures (style of ../figstyle.py)
  results/rev2/report.json        every number of the tables and figures
Sections whose input files are missing are skipped, so the script can be run on partial results.

  1 calibration    one-sided coverage of the central 80%, 90% and 98% intervals, coverage by lead time, PIT histograms
  2 dependence     rho_model and rho_obs at lags 1, 2, 6, 12 (24, 48 where H allows) with week-cluster bootstrap
                   intervals; tie-aware rho_obs (pairs with a PIT outside the sample range excluded; mid-PIT) and the
                   share of PIT values outside the sample range
  3 aggregates     calibration of the total and the maximum over the horizon (randomized PIT of the observed value
                   among the path values; coverage of the central 80% interval)
  4 leakage        Delta of Eq. (13) relative to Chronos-Bolt small and to the median of the models without overlap,
                   for several pairs of old and new data, with week-cluster bootstrap intervals
  5 decision       reserves in MW from the 0.9 (0.95, 0.99) quantile above the median, and a daily-peak reserve
  6 euler          Sundial and Aurora with 5 to 200 sampling steps (and the stochastic sampler)
  7 chunked        Chronos-2 rolled out in chunks at H = 720: QL and coverage by forecast day
  8 context        TimesFM 2.5 and Chronos-2 with 2,048 and 4,096 context steps
  9 postprocessing spread correction and copulas

Usage:  python revision_report.py [--rev2dir DIR] [--texdir DIR] [--figdir DIR] [--boot 2000]
"""

import argparse
import json
import os
import re
import sys
from functools import lru_cache

import numpy as np

import rev2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

MODELS = ["naive", "chronos", "chronos_base", "toto", "timesfm", "bolt", "bolt_base", "chronos2", "sundial", "aurora"]
PATHS = {"naive", "chronos", "chronos_base", "toto", "sundial", "aurora"}
QMODELS = ["timesfm", "bolt", "bolt_base", "chronos2"]
class _Names(dict):
    def __missing__(self, key):                       # models of other scripts (e.g. extra baselines): own name
        return str(key).replace("_", "\\_")


SHORT = _Names({"naive": "Naive", "chronos": "Chronos small", "chronos_base": "Chronos base", "toto": "Toto",
                "timesfm": "TimesFM 2.5", "bolt": "Chronos-Bolt small", "bolt_base": "Chronos-Bolt base",
                "chronos2": "Chronos-2", "sundial": "Sundial", "aurora": "Aurora"})
DSN = {"ercot": "ERCOT", "traffic": "Traffic", "ausel": "Australia 2002--2015", "aemo": "AEMO 2026",
       "aemo_mid": "AEMO 2016--2019", "ausel_season": "Australia 2002--2015, Jan.--Sep.", "ercot2026": "ERCOT 2026"}
MAIN4 = ["ercot", "traffic", "ausel", "aemo"]
ALL24 = MAIN4 + ["aemo_mid", "ausel_season", "ercot2026"]
LAGS = [1, 2, 6, 12, 24, 48]
VARIANT = {"": "", "schaake": " + Schaake", "gausscop": " + Gauss. copula", "spread": " + spread corr.",
           "indep": " (indep. draws)"}
# models whose pretraining data have no known overlap with the older data of a source (Table III: E = excluded)
NO_OVERLAP = {"ausel": ["chronos", "chronos_base", "bolt", "bolt_base"],
              "ercot": ["chronos", "chronos_base", "timesfm", "bolt", "bolt_base"]}
PAIRS = [("ausel", "aemo"), ("ausel_season", "aemo"), ("ausel_season", "aemo_mid"), ("aemo_mid", "aemo"),
         ("ercot", "ercot2026"), ("ercot|valid", "ercot2026|valid")]
LEAKN = {"ausel": "Aus.\\ 02--15", "ausel_season": "Aus.\\ 02--15 Jan.--Sep.", "aemo": "AEMO 26",
         "aemo_mid": "AEMO 16--19", "ercot": "ERCOT 04--21", "ercot2026": "ERCOT 26"}
SOURCE = {"ausel": "ausel", "ausel_season": "ausel", "aemo_mid": "ausel", "aemo": "ausel", "ercot": "ercot",
          "ercot2026": "ercot"}
STEP_HOURS = {"ercot": 1.0, "ercot2026": 1.0, "aemo": 0.5, "aemo_mid": 0.5}
OUT = {}                                               # all numbers, written to report.json


# =====================================================================
# Access to the rev2 files
# =====================================================================

class Store:
    def __init__(self, d):
        self.dir = d
        self.idx = rev2.listing(out_dir=d, setting=None)

    def keys(self, ds=None, H=None, model=None, setting=""):
        return [(p, k) for p, k in self.idx if (ds is None or k["dataset"] == ds) and (H is None or k["H"] == H)
                and (model is None or k["model"] == model) and (setting is None or (k["setting"] or "") == setting)]

    def runs(self, ds, H, model, setting=""):
        return [_load(p) for p, _ in sorted(self.keys(ds, H, model, setting), key=lambda x: x[1]["seed"])]

    def models(self, ds, H, setting="", extra=False):
        """Models with files, in the order of MODELS; models of other scripts (extra baselines written in the same
        format) only with extra=True."""
        have = {k["model"] for _, k in self.keys(ds, H, None, setting)}
        return [m for m in MODELS if m in have] + (sorted(have - set(MODELS)) if extra else [])


@lru_cache(maxsize=16)
def _load(path):
    r = rev2.load(path)
    if "paths" in r:
        r["paths"] = r["paths"].astype(np.float32)
    return r


def with_paths(r, S=100, seed=0):
    """Paths of a run; for quantile models independent draws from the quantile function (as for the CRPS)."""
    if "paths" in r:
        return r["paths"]
    rng = np.random.default_rng(seed)
    return rev2.sample_quantiles(r["q"], r["levels"], rng.random((r["y"].shape[0], S, r["y"].shape[1]))).astype(np.float32)


def is_quantile(r):
    return "paths" not in r


def covered(r, lo, hi):
    u = r["pit"]
    return (u >= lo) & (u <= hi) if is_quantile(r) else (u > lo) & (u < hi)


def fmt(v, nd=3):
    return "--" if v is None or v != v else f"{v:,.{nd}f}"


def pct(v):
    return "--" if v is None or v != v else f"{100 * v:.0f}"


def ci(lo, hi, nd=2):
    return "" if lo != lo else f" [{lo:.{nd}f}, {hi:.{nd}f}]"


def write_tex(texdir, name, lines):
    os.makedirs(texdir, exist_ok=True)
    with open(os.path.join(texdir, f"rev2_{name}.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by realdata/revision_report.py\n" + "\n".join(lines) + "\n")


def signs(row):
    """A table row with minus signs instead of hyphens before numbers, and without negative zeros."""
    row = re.sub(r"(?<![\w$\\{}-])-(0(?:\.0+)?)(?![\d.])", r"\1", row)
    return re.sub(r"(?<![\w$\\{}-])-(?=\d)", "$-$", row)


def table(caption, label, colspec, head, body, note, wide=True, size=r"\scriptsize"):
    body = [signs(b) for b in body]
    env = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    # fitwidth (defined in supplementary.tex) shrinks the table only where it is wider than the text (the
    # supplement is narrower than a two-column page)
    return ([rf"\begin{{{env}}}[!t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size,
             r"\setlength{\tabcolsep}{3pt}", r"\begin{fitwidth}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"] + head + [r"\midrule"]
            + body + [r"\bottomrule", r"\end{tabular}", r"\end{fitwidth}", r"", r"\vspace{2pt}",
                      rf"\parbox{{{width}}}{{\footnotesize {note}}}", rf"\end{{{env}}}"])


# =====================================================================
# Scores
# =====================================================================

def energy_score(x, y, chunk=20):
    out = []
    for i in range(0, x.shape[0], chunk):
        xs, ys = x[i:i + chunk].astype(np.float64), y[i:i + chunk].astype(np.float64)
        t1 = np.linalg.norm(xs - ys[:, None], axis=2).mean(1)
        d = np.sqrt(((xs[:, :, None] - xs[:, None]) ** 2).sum(-1))
        out.append(t1 - 0.5 * d.mean((1, 2)))
    return np.concatenate(out)


def variogram_score(x, y, p=0.5, chunk=20):
    out = []
    for i in range(0, x.shape[0], chunk):
        xs, ys = x[i:i + chunk].astype(np.float64), y[i:i + chunk].astype(np.float64)
        vy = np.abs(ys[:, :, None] - ys[:, None]) ** p
        vx = (np.abs(xs[:, :, :, None] - xs[:, :, None]) ** p).mean(1)
        out.append(((vy - vx) ** 2).mean((1, 2)))
    return np.concatenate(out)


def basic(r):
    """Per-window QL, coverage, CRPS (independent draws for quantile models)."""
    x = with_paths(r)
    return dict(ql=rev2.quantile_loss(r["q"], r["levels"], r["y"]), cov80=covered(r, 0.1, 0.9).mean(1),
                crps=rev2.crps_paths(x, r["y"]).mean(1))


def seed_mean(runs, fn):
    vals = [fn(r) for r in runs]
    return {k: np.mean([v[k] for v in vals], 0) for k in vals[0]} if vals else {}


# =====================================================================
# 1 Calibration
# =====================================================================

def section_calibration(st, texdir, figdir):
    res = {}
    for ds in ALL24:
        for m in st.models(ds, 24):
            runs = st.runs(ds, 24, m)
            out = {}
            for name, (lo, hi) in {"80": (0.1, 0.9), "90": (0.05, 0.95), "98": (0.01, 0.99)}.items():
                if is_quantile(runs[0]) and rev2.level_index(runs[0]["levels"], [lo, hi]) is None:
                    continue
                out[name] = dict(below=float(np.mean([(r["pit"] < lo).mean() for r in runs])),
                                 above=float(np.mean([(r["pit"] > hi).mean() for r in runs])),
                                 cov=float(np.mean([covered(r, lo, hi).mean() for r in runs])))
            out["lead80"] = np.mean([covered(r, 0.1, 0.9).mean(0) for r in runs], 0).tolist()
            out["hist10"] = np.mean([np.histogram(r["pit"], bins=10, range=(0, 1))[0] / r["pit"].size
                                     for r in runs], 0).tolist()
            res[f"{ds}|{m}"] = out
    OUT["calibration"] = res
    if not res:
        return
    dss = [d for d in ALL24 if any(k.startswith(d + "|") for k in res)]
    body = []
    for m in MODELS:
        cells = []
        for ds in dss:
            c = res.get(f"{ds}|{m}", {}).get("80")
            cells.append("--" if c is None else f"{pct(c['below'])} / {pct(c['above'])}")
        if any(c != "--" for c in cells):
            body.append(SHORT[m] + " & " + " & ".join(cells) + r" \\")
    write_tex(texdir, "onesided80", table(
        "One-Sided Coverage of the Central 80\\% Interval, 24-Step Forecasts", "tab:rev2onesided",
        "@{}l" + " c" * len(dss) + "@{}", ["Model & " + " & ".join(DSN[d] for d in dss) + r" \\"], body,
        "Share (\\%) of outcomes below the 0.1 quantile / above the 0.9 quantile (10 / 10 for a calibrated "
        "forecast). The shares come from the randomized PIT of the sample paths, Eq.~(\\ref{eq:pit}), and from the PIT of the piecewise-linear "
        "quantile function of quantile models. Averaged over the sampling seeds."))
    body = []
    for m in MODELS:
        cells = []
        for ds in MAIN4:
            for lvl in ["90", "98"]:
                c = res.get(f"{ds}|{m}", {}).get(lvl)
                cells.append("--" if c is None else f"{100 * c['below']:.1f} / {100 * c['above']:.1f}")
        if any(c != "--" for c in cells):
            body.append(SHORT[m] + " & " + " & ".join(cells) + r" \\")
    head = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{DSN[d]}}}" for d in MAIN4) + r" \\",
            "Model" + " & 90\\% & 98\\%" * len(MAIN4) + r" \\"]
    write_tex(texdir, "onesided_tails", table(
        "One-Sided Coverage of the Central 90\\% and 98\\% Intervals", "tab:rev2tails",
        "@{}l" + " c" * 2 * len(MAIN4) + "@{}", head, body,
        "Share (\\%) below the lower / above the upper end (5 / 5 and 1 / 1 for a calibrated forecast). Quantile "
        "models are listed solely where the native levels include the ends of the interval (Chronos-Bolt and "
        "TimesFM~2.5 provide 0.1 to 0.9)."))
    import figstyle
    import matplotlib.pyplot as plt
    figstyle.apply()
    ms = [m for m in MODELS if any(f"{d}|{m}" in res for d in MAIN4)]
    dsf = [d for d in MAIN4 if any(f"{d}|{m}" in res for m in ms)]
    fig, ax = plt.subplots(len(dsf), len(ms), figsize=(7.16, 1.0 * len(dsf) + 0.3), sharex=True, sharey=True,
                           squeeze=False)
    for i, ds in enumerate(dsf):
        for j, m in enumerate(ms):
            a = ax[i, j]
            h = res.get(f"{ds}|{m}", {}).get("hist10")
            if h is not None:
                a.bar(np.arange(10) / 10 + 0.05, np.array(h) * 10, 0.1, color=COLOR.get(m, figstyle.GREY),
                      edgecolor=figstyle.INK, linewidth=0.3)
            a.axhline(1, color=figstyle.INK, lw=0.6, ls="--")
            a.set_xlim(0, 1)
            a.set_ylim(0, 4)
            a.set_xticks([0, 0.5, 1])
            a.set_xticklabels(["0", "", "1"])          # full labels of adjacent panels would touch
            a.tick_params(labelsize=5.5)
            if i == 0:
                a.set_title(SHORT[m].replace("Chronos-Bolt ", "Chronos-Bolt\n").replace("Chronos ", "Chronos\n"),
                            fontsize=6)
            if j == 0:
                a.set_ylabel(DSF[ds], fontsize=6)
    fig.supxlabel("PIT", fontsize=7)
    fig.savefig(os.path.join(figdir, "rev2_pit_hist.pdf"))
    plt.close(fig)
    fig, ax = plt.subplots(1, len(dsf), figsize=(7.16, 2.3), sharey=True, squeeze=False)
    for i, ds in enumerate(dsf):
        a = ax[0, i]
        for m in ms:
            c = res.get(f"{ds}|{m}", {}).get("lead80")
            if c is not None:
                a.plot(np.arange(1, len(c) + 1), 100 * np.array(c), color=COLOR.get(m), ls=LS.get(m, "-"), lw=1,
                       label=SHORT[m])
        a.axhline(80, color=figstyle.INK, lw=0.6, ls=":")
        a.set_title(DSF[ds])
        a.set_xlabel("Lead time (steps)")
    ax[0, 0].set_ylabel("Coverage of 80% interval (%)")
    fig_legend(fig, ax)
    fig.savefig(os.path.join(figdir, "rev2_coverage_lead.pdf"))
    plt.close(fig)


# =====================================================================
# 2 Dependence
# =====================================================================

def dep_variants(st, ds, H=24):
    """(label, model, setting) of every forecast with paths of its own (and copula / spread variants)."""
    out = []
    for m in st.models(ds, H):
        if m in PATHS:
            out.append((m, m, ""))
    for setting in ["schaake", "gausscop", "spread"]:
        for m in st.models(ds, H, setting):
            out.append((f"{m}|{setting}", m, setting))
    return out


def rho_stats(runs, lags, B, rng):
    """rho_model and rho_obs per lag, averaged over seeds, with week-cluster bootstrap intervals; tie-aware
    variants of rho_obs and the share of PIT values outside the sample range."""
    from scipy.stats import rankdata
    H = runs[0]["y"].shape[1]
    lags = [l for l in lags if l < H]
    week = runs[0]["week"]
    rm = [rev2.rho_model_windows(r["paths"], lags) for r in runs]                    # per seed {lag: (N,)}
    rmw = {l: np.mean([x[l] for x in rm], 0) for l in lags}
    ro = [rev2.rho_obs(r["pit"], lags) for r in runs]
    res = {l: dict(model=float(np.nanmean(rmw[l])), obs=float(np.mean([x[l] for x in ro]))) for l in lags}
    draws = {l: ([], []) for l in lags}
    for idx in rev2.cluster_draws(week, B, rng):
        vals = [rev2.rho_obs(r["pit"][idx], lags, ranks=rankdata(r["pit"][idx], axis=0)) for r in runs]
        for l in lags:
            draws[l][0].append(np.nanmean(rmw[l][idx]))
            draws[l][1].append(np.mean([v[l] for v in vals]))
    for l in lags:
        dm, do = np.array(draws[l][0]), np.array(draws[l][1])
        res[l].update(model_ci=rev2.interval(dm), obs_ci=rev2.interval(do), diff=res[l]["model"] - res[l]["obs"],
                      diff_ci=rev2.interval(dm - do))
    # ties: PIT values outside the range of the paths (no path below or none above the outcome)
    S = runs[0]["paths"].shape[1]
    ties = {}
    for r in runs:
        k = rev2.pit_below_count(r["pit"], S)
        inside = (k > 0) & (k < S)
        mid = (k + 0.5) / (S + 1)
        ties.setdefault("outside", []).append(float(1 - inside.mean()))
        for l, v in rev2.rho_obs(r["pit"], lags, keep=inside).items():
            ties.setdefault(f"excl{l}", []).append(v)
        for l, v in rev2.rho_obs(mid, lags).items():
            ties.setdefault(f"mid{l}", []).append(v)
    for key, v in ties.items():
        ties[key] = float(np.nanmean(v))
    return res, ties


def write_dep_table(texdir, tname, cap, label, lag_cols, body, B):
    head = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{Lag {l}}}" for l in lag_cols) +
            r" & \multicolumn{2}{c}{Ties, lag 1} \\",
            "Forecast" + r" & $\rho_{\mathrm{model}}$ / $\rho_{\mathrm{obs}}$ & Difference [95\% CI]" * len(lag_cols)
            + r" & Excl. / mid & Out (\%) \\"]
    write_tex(texdir, tname, table(
        "Rank Dependence at Several Lags, 24-Step Forecasts" + cap, label,
        "@{}l" + " c c" * len(lag_cols) + " c r@{}", head, body,
        "$\\rho_{\\mathrm{model}}$ and $\\rho_{\\mathrm{obs}}$ as in Eqs.~(\\ref{eq:rhomodel}) and~(\\ref{eq:rhoobs}) "
        "with steps $h$ and $h + \\mathrm{lag}$. Difference: $\\rho_{\\mathrm{model}} - \\rho_{\\mathrm{obs}}$ with a "
        f"95\\% interval from {B:,} bootstrap samples of calendar weeks. Ties: $\\rho_{{\\mathrm{{obs}}}}$ at lag 1 "
        "without the pairs whose PIT value lies outside the paths / with the mid-PIT. Out: share of PIT values outside "
        "the paths. Schaake, Gauss.\\ copula: quantile paths joined by the empirical or Gaussian copula of past "
        "trajectories."))


def section_dependence(st, texdir, figdir, B):
    res = {}
    rng = np.random.default_rng(11)
    for ds in ALL24:
        for label, m, setting in dep_variants(st, ds):
            runs = st.runs(ds, 24, m, setting)
            r, t = rho_stats(runs, LAGS, B, rng)
            res[f"{ds}|{label}"] = dict(lags={str(k): v for k, v in r.items()}, ties=t)
            print(f"[dep] {ds:12s} {label:18s} rho_model(1) {r[1]['model']:.3f} rho_obs(1) {r[1]['obs']:.3f} "
                  f"outside {t['outside']:.3f}", flush=True)
    OUT["dependence"] = res
    if not res:
        return
    lag_cols = [1, 6, 12]
    for tname, dsets, cap, lab in [("dependence", MAIN4, "", "stab:rev2dep"),
                                   ("dependence_new", ["aemo_mid", "ausel_season", "ercot2026"],
                                    ", Season-Matched and New Data", "stab:rev2depnew")]:
        body, last = [], None
        for ds in dsets:
            for label, m, setting in dep_variants(st, ds):
                if setting == "spread":                    # in the spread table (ranks of the paths unchanged)
                    continue
                if ds != last:
                    if last is not None:
                        body.append(r"\midrule")
                    body.append(rf"\multicolumn{{{1 + 2 * len(lag_cols) + 2}}}{{@{{}}l}}{{\textit{{{DSN[ds]}}}}} \\")
                    last = ds
                v = res[f"{ds}|{label}"]
                cells = []
                for l in lag_cols:
                    x = v["lags"].get(str(l))
                    cells += (["--", "--"] if x is None else
                              [f"{x['model']:.2f} / {x['obs']:.2f}", f"{x['diff']:+.2f}{ci(*x['diff_ci'])}"])
                t = v["ties"]
                cells += [fmt(t.get("excl1"), 2) + " / " + fmt(t.get("mid1"), 2), pct(t["outside"])]
                body.append(SHORT[m] + VARIANT[setting] + " & " + " & ".join(cells) + r" \\")
        if body:
            write_dep_table(texdir, tname, cap, lab, lag_cols, body, B)
    # longer horizons (optional runs): lags 24 and 48 need H > 48
    long_body = []
    for H in [192, 720]:
        for ds in ["ercot", "traffic", "ausel"]:
            for m in [m for m in st.models(ds, H) if m in MODELS]:
                runs = st.runs(ds, H, m)
                if "paths" not in runs[0]:
                    continue
                r, t = rho_stats(runs, [1, 24, 48], max(1, B // 4), rng)
                res[f"{ds}|H{H}|{m}"] = dict(lags={str(k): v for k, v in r.items()}, ties=t)
                cells = [f"{r[l]['model']:.2f} / {r[l]['obs']:.2f} ({r[l]['diff']:+.2f}{ci(*r[l]['diff_ci'])})"
                         for l in [1, 24, 48]]
                long_body.append(f"{DSN[ds]} & {H} & {SHORT[m]} & " + " & ".join(cells) + r" \\")
    if long_body:
        write_tex(texdir, "dependence_long", table(
            "Rank Dependence at Lags of 24 and 48 Steps, Longer Horizons", "stab:rev2deplong",
            "@{}l r l c c c@{}", [r"Dataset & $H$ & Model & Lag 1 & Lag 24 & Lag 48 \\"], long_body,
            "$\\rho_{\\mathrm{model}}$ / $\\rho_{\\mathrm{obs}}$ (difference with a 95\\% interval from "
            f"{max(1, B // 4):,} week-cluster bootstrap samples) for the naive baseline, Toto, Sundial and Aurora; "
            "150 windows and one seed."))
    import figstyle
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    figstyle.apply()
    dsf = [d for d in ALL24 if any(k.startswith(d + "|") for k in res)]
    ncol = min(4, len(dsf))
    nrow = -(-len(dsf) // ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(7.16, 2.0 * nrow + 0.4), sharey=True, squeeze=False)
    for i, ds in enumerate(dsf):
        a = ax[i // ncol, i % ncol]
        for label, m, setting in dep_variants(st, ds):
            if setting:
                continue
            v = res[f"{ds}|{label}"]["lags"]
            ls = sorted(int(k) for k in v)
            a.plot(ls, [v[str(l)]["model"] for l in ls], color=COLOR.get(m), lw=1, marker="o", ms=2, label=SHORT[m])
            a.plot(ls, [v[str(l)]["obs"] for l in ls], color=COLOR.get(m), lw=0.8, ls="--")
        a.set_xscale("log")
        a.xaxis.set_minor_locator(NullLocator())
        a.set_xticks([1, 2, 6, 12])
        a.set_xticklabels(["1", "2", "6", "12"])
        a.set_title(DSF[ds])
        if i + ncol >= len(dsf):
            a.set_xlabel("Lag (steps)")
        if i % ncol == 0:
            a.set_ylabel(r"$\rho_{\mathrm{model}}$ (solid), $\rho_{\mathrm{obs}}$ (dashed)")
    for j in range(len(dsf), nrow * ncol):
        ax[j // ncol, j % ncol].set_visible(False)
    fig.subplots_adjust(hspace=0.35)
    fig_legend(fig, ax)
    fig.savefig(os.path.join(figdir, "rev2_rho_lags.pdf"))
    plt.close(fig)


# =====================================================================
# 3 Total and maximum over the horizon
# =====================================================================

def agg_calibration(x, y, rng):
    out = {}
    for name, fn in [("total", lambda a, ax: a.sum(ax)), ("max", lambda a, ax: a.max(ax))]:
        px, py = fn(x, 2), fn(y, 1)                                       # (N, S), (N,)
        below = (px < py[:, None]).sum(1)
        u = (below + rng.random(len(py))) / (px.shape[1] + 1)
        out[name] = dict(cov80=float(((u > 0.1) & (u < 0.9)).mean()), below=float((u < 0.1).mean()),
                         above=float((u > 0.9).mean()))
    return out


def section_aggregates(st, texdir):
    res = {}
    rng = np.random.default_rng(13)
    for ds in ALL24:
        variants = dep_variants(st, ds) + [(f"{m}|indep", m, "") for m in QMODELS if m in st.models(ds, 24)]
        for label, m, setting in variants:
            vals = [agg_calibration(with_paths(r), r["y"], rng) for r in st.runs(ds, 24, m, setting)]
            res[f"{ds}|{label}"] = {k: {kk: float(np.mean([v[k][kk] for v in vals])) for kk in vals[0][k]}
                                    for k in vals[0]}
    OUT["aggregates"] = res
    if not res:
        return
    dss = [d for d in MAIN4 if any(k.startswith(d + "|") for k in res)]
    labels = []
    for k in res:
        lab = k.split("|", 1)[1]
        if lab not in labels:
            labels.append(lab)
    order = {m: i for i, m in enumerate(MODELS)}
    labels.sort(key=lambda s: (order.get(s.split("|")[0], 99), s))
    body = []
    for lab in labels:
        m, setting = (lab.split("|") + [""])[:2]
        cells = []
        for ds in dss:
            v = res.get(f"{ds}|{lab}")
            cells += ["--", "--"] if v is None else [pct(v["total"]["cov80"]), pct(v["max"]["cov80"])]
        body.append(SHORT[m] + VARIANT[setting] + " & " + " & ".join(cells) + r" \\")
    head = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{DSN[d]}}}" for d in dss) + r" \\",
            "Forecast" + " & Total & Max." * len(dss) + r" \\"]
    write_tex(texdir, "aggregates", table(
        "Calibration of the Total and the Maximum over the Horizon", "tab:rev2agg",
        "@{}l" + " r r" * len(dss) + "@{}", head, body,
        "Coverage (\\%, ideally 80) of the central 80\\% interval of the total and the maximum of the 24 steps, from "
        "the randomized PIT among the totals or maxima of the 100 paths. Indep.\\ draws: quantile paths drawn "
        "independently at each step. Spread corr.: spread correction of Section~\\ref{sec:settings}. Schaake, "
        "Gauss.\\ copula: quantile paths joined by the empirical or Gaussian copula of past trajectories."))


# =====================================================================
# 4 Leakage
# =====================================================================

def ql_matrix(st, ds, models, sids=None):
    """(models x windows) QL averaged over seeds, and the week of each window."""
    rows, week = [], None
    for m in models:
        runs = st.runs(ds, 24, m)
        q = np.mean([rev2.quantile_loss(r["q"], r["levels"], r["y"]) for r in runs], 0)
        week = runs[0]["week"]
        keep = np.ones(len(q), bool) if sids is None else np.isin(runs[0]["sid"], sids)
        rows.append(q[keep])
        wk = week[keep]
    return np.array(rows), wk


def delta_draws(Qo, wo, Qn, wn, ref_rows, B, rng):
    """Delta_m (Eq. 13) for every model relative to the mean QL of `ref_rows` (one row: a reference model; several:
    their median), point estimate and bootstrap draws (weeks resampled independently in both periods)."""
    def stat(mo, mn):
        ro = mo / np.median(mo[ref_rows])
        rn = mn / np.median(mn[ref_rows])
        return rn / ro - 1
    point = stat(Qo.mean(1), Qn.mean(1))
    go, gn = rev2.cluster_index(wo), rev2.cluster_index(wn)
    draws = []
    for _ in range(B):
        io = np.concatenate([go[i] for i in rng.integers(0, len(go), len(go))])
        i_n = np.concatenate([gn[i] for i in rng.integers(0, len(gn), len(gn))])
        draws.append(stat(Qo[:, io].mean(1), Qn[:, i_n].mean(1)))
    return point, np.array(draws)


def section_leakage(st, texdir, figdir, B, checks):
    res = {}
    rng = np.random.default_rng(17)
    valid = checks.get("ercot", {}).get("validated_zones") if checks else None
    zones = ["COAST", "EAST", "FAR_WEST", "NORTH", "NORTH_C", "SOUTHERN", "SOUTH_C", "WEST"]
    for old, new in PAIRS:
        o, n = old.split("|")[0], new.split("|")[0]
        sids = None
        if old.endswith("|valid"):
            if not valid:
                continue
            sids = [zones.index(z) for z in valid]
        models = [m for m in st.models(o, 24) if m in st.models(n, 24) and m in MODELS]
        if "bolt" not in models:
            continue
        Qo, wo = ql_matrix(st, o, models, sids)
        Qn, wn = ql_matrix(st, n, models, sids)
        src = SOURCE[o]
        refs = {"bolt": [models.index("bolt")],
                "median": [models.index(m) for m in NO_OVERLAP[src] if m in models]}
        out = {}
        for rname, rows in refs.items():
            if not rows:
                continue
            point, draws = delta_draws(Qo, wo, Qn, wn, rows, B, rng)
            for i, m in enumerate(models):
                out.setdefault(m, {})[rname] = dict(delta=float(point[i]), ci=rev2.interval(draws[:, i]))
        for i, m in enumerate(models):
            out[m]["ql_old"], out[m]["ql_new"] = float(Qo[i].mean()), float(Qn[i].mean())
        key = f"{old}->{new}"
        res[key] = dict(models=out, n_old=int(Qo.shape[1]), n_new=int(Qn.shape[1]),
                        weeks_old=int(len(np.unique(wo))), weeks_new=int(len(np.unique(wn))),
                        reference_median=[models[i] for i in refs["median"]], zones=valid if sids else None)
        print(f"[leak] {key}: " + " ".join(f"{m}={out[m]['bolt']['delta']:+.3f}" for m in models), flush=True)
    OUT["leakage"] = res
    if not res:
        return
    keys = list(res)
    body = []
    for m in MODELS:
        rows = []
        for rn, rlab in [("bolt", "Bolt"), ("median", "Median")]:      # one row per reference keeps the table narrow
            cells = []
            for k in keys:
                x = (res[k]["models"].get(m) or {}).get(rn)
                cells.append("--" if x is None else
                             f"{100 * x['delta']:+.0f} [{100 * x['ci'][0]:+.0f}, {100 * x['ci'][1]:+.0f}]")
            rows.append((rlab, cells))
        if any(c != "--" for _, cells in rows for c in cells):
            body += [(SHORT[m] if i == 0 else "") + f" & {rlab} & " + " & ".join(cells) + r" \\"
                     for i, (rlab, cells) in enumerate(rows)]
    nz = len(valid) if valid else 0
    name = lambda k: r"\makecell{" + r" $\to$ \\ ".join(
        LEAKN[p.split("|")[0]].replace(" Jan.", r" \\ Jan.") + (rf" \\ ({nz} zones)" if "|valid" in p else "")
        for p in k.split("->")) + "}"
    head = ["Model & Ref. & " + " & ".join(name(k) for k in keys) + r" \\"]
    # plain-text panel titles for the figure: "older data" on the first line, "-> newer data" on the second
    plain = lambda p: (LEAKN[p.split("|")[0]] + (f" ({nz} zones)" if "|valid" in p else "")).replace(
        "\\ ", " ").replace(" Jan.", ", Jan.").replace("--", "–")
    title = lambda k: "\n→ ".join(plain(p) for p in k.split("->"))
    write_tex(texdir, "leakage", table(
        "Leakage Test across Periods and Sources: Change in Standing $\\Delta_m$ (\\%)", "stab:rev2leak",
        "@{}l l" + " c" * len(keys) + "@{}", head, body,
        "$\\Delta_m$, Eq.~(\\ref{eq:delta}), in percent with 95\\% intervals from " + f"{B:,}" + " bootstrap samples "
        "of calendar weeks. Bolt: reference Chronos-Bolt small. Median: median QL of the models without known overlap "
        "with the older data, namely Chronos and Chronos-Bolt in both sizes and for ERCOT also TimesFM~2.5. Jan.--Sep.: "
        "origins restricted to the months of the 2026 data. 5 zones: the weather zones " + "/".join(valid or []).replace(
            "_", "\\_") + ", whose 2021 values in EIA-930 agree with the ERCOT dataset."))
    import figstyle
    import matplotlib.pyplot as plt
    figstyle.apply()
    ncol = min(3, len(keys))
    nrow = -(-len(keys) // ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(7.16, 2.2 * nrow), sharey=True, squeeze=False)
    for j in range(len(keys), nrow * ncol):
        ax[j // ncol, j % ncol].set_visible(False)
    ms_all = [m for m in MODELS if any(m in res[k]["models"] for k in keys)]
    for i, k in enumerate(keys):
        a = ax[i // ncol, i % ncol]
        for m in [m for m in ms_all if m in res[k]["models"]]:
            j = ms_all.index(m)
            for rn, off, mk in [("bolt", -0.15, "o"), ("median", 0.15, "s")]:
                x = res[k]["models"][m].get(rn)
                if x is None:
                    continue
                a.errorbar(100 * x["delta"], j + off, xerr=[[100 * (x["delta"] - x["ci"][0])],
                                                            [100 * (x["ci"][1] - x["delta"])]],
                           fmt=mk, ms=2.5, color=COLOR.get(m), elinewidth=0.6, capsize=1)
        a.axvline(0, color=figstyle.INK, lw=0.6)
        a.set_yticks(range(len(ms_all)))
        a.set_yticklabels([SHORT[m] for m in ms_all], fontsize=6)
        a.set_ylim(len(ms_all) - 0.5, -0.5)
        a.set_title(title(k), fontsize=6.5)
        a.set_xlabel(r"$\Delta_m$ (%)")
    from matplotlib.lines import Line2D
    fig.subplots_adjust(hspace=0.62)
    fig.legend([Line2D([], [], marker="o", ls="", color=figstyle.INK, ms=3),
                Line2D([], [], marker="s", ls="", color=figstyle.INK, ms=3)],
               ["Reference: Chronos-Bolt small", "Reference: median of the models without overlap"],
               loc="lower center", ncol=2, fontsize=6, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(figdir, "rev2_leakage.pdf"))
    plt.close(fig)


# =====================================================================
# 5 Decision case study
# =====================================================================

def section_decision(st, texdir):
    res = {}
    for ds in ["ercot", "aemo", "ercot2026"]:
        hrs = STEP_HOURS[ds]
        for m in st.models(ds, 24):
            out = {}
            for r in st.runs(ds, 24, m):
                y = r["y_orig"].astype(np.float64)
                qmw = r["q"] * r["sd"][:, None, None] + r["m"][:, None, None]          # MW, (N, K, H)
                i50 = rev2.level_index(r["levels"], [0.5])[0]
                for tau in [0.9, 0.95, 0.99]:
                    j = rev2.level_index(r["levels"], [tau])
                    if j is None:
                        continue
                    P = qmw[:, j[0]]
                    short, over = np.maximum(y - P, 0), np.maximum(P - y, 0)
                    v = dict(freq=float((y > P).mean()), shortfall_mwh=float(short.mean() * hrs),
                             over_mwh=float(over.mean() * hrs), reserve_mw=float((P - qmw[:, i50]).mean()))
                    for k in [5, 10, 20]:
                        v[f"cost{k}"] = float((k * short + over).mean() * hrs)
                    out.setdefault(str(tau), []).append(v)
                x = with_paths(r) * r["sd"][:, None, None] + r["m"][:, None, None]   # paths in MW
                Pk = np.quantile(x.max(2), 0.9, axis=1)
                ymax = y.max(1)
                out.setdefault("peak", []).append(dict(freq=float((ymax > Pk).mean()),
                                                       shortfall_mw=float(np.maximum(ymax - Pk, 0).mean()),
                                                       over_mw=float(np.maximum(Pk - ymax, 0).mean())))
            res[f"{ds}|{m}"] = {k: {kk: float(np.mean([d[kk] for d in v])) for kk in v[0]} for k, v in out.items()}
            for setting in ["schaake", "gausscop"]:
                runs = st.runs(ds, 24, m, setting)
                if runs:
                    vals = []
                    for r in runs:
                        x = r["paths"] * r["sd"][:, None, None] + r["m"][:, None, None]
                        Pk = np.quantile(x.max(2), 0.9, axis=1)
                        ymax = r["y_orig"].max(1)
                        vals.append(dict(freq=float((ymax > Pk).mean()),
                                         shortfall_mw=float(np.maximum(ymax - Pk, 0).mean()),
                                         over_mw=float(np.maximum(Pk - ymax, 0).mean())))
                    res[f"{ds}|{m}|{setting}"] = {"peak": {kk: float(np.mean([d[kk] for d in vals])) for kk in vals[0]}}
    OUT["decision"] = res
    if not res:
        return
    dss = [d for d in ["ercot", "aemo", "ercot2026"] if any(k.startswith(d + "|") for k in res)]
    body = []
    for ds in dss:
        body.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\textit{{{DSN[ds]} (MWh per forecast step)}}}} \\")
        for key in [k for k in res if k.startswith(ds + "|")]:
            parts = key.split("|")
            m, setting = parts[1], (parts[2] if len(parts) > 2 else "")
            v = res[key]
            t = v.get("0.9")
            pk = v.get("peak")
            cells = (["--"] * 6 if t is None else
                     [pct(t["freq"]), f"{t['reserve_mw']:,.0f}", f"{t['shortfall_mwh']:,.1f}", f"{t['over_mwh']:,.0f}",
                      f"{t['cost10']:,.0f}", fmt(v.get("0.95", {}).get("cost10"), 0)])
            cells += ["--"] * 3 if pk is None else [pct(pk["freq"]), f"{pk['shortfall_mw']:,.0f}", f"{pk['over_mw']:,.0f}"]
            body.append(SHORT[m] + VARIANT[setting] + " & " + " & ".join(cells) + r" \\")
        body.append(r"\midrule")
    body = body[:-1]
    head = [r" & \multicolumn{6}{c}{Reserve at the 0.9 quantile} & \multicolumn{3}{c}{Peak reserve} \\",
            r"Forecast & Short. (\%) & Reserve (MW) & Shortfall & Over & Cost$_{10}$ & Cost$_{10}$ (0.95) & "
            r"Short. (\%) & Shortfall (MW) & Over (MW) \\"]
    write_tex(texdir, "decision", table(
        "Reserve Sizing from the Forecasts, 24-Step Forecasts in Original Units", "stab:rev2decision",
        "@{}l r r r r r r r r r@{}", head, body,
        "Procurement at the forecast 0.9 quantile. Reserve: mean distance of this quantile above the forecast median. Short.: share of steps with demand above the procurement, ideally "
        "10\\%. Shortfall and Over: expected unserved and over-procured energy per step. Cost$_{10}$: $10 \\times$ "
        "shortfall $+$ over-procurement; the costs for the ratios 5 and 20 are in \\texttt{report.json}. (0.95): "
        "procurement at the 0.95 quantile where the model provides it. Peak reserve: the 0.9 quantile of the maximum "
        "of the paths over the 24 steps against the realized maximum; Short.: share of windows."))


# =====================================================================
# 6-8 Settings: Euler steps, chunked rollout, context length
# =====================================================================

def setting_scores(r):
    b = basic(r)
    out = {k: float(v.mean()) for k, v in b.items()}
    if "paths" in r:
        rm = rev2.rho_model_windows(r["paths"], [1])
        out["rho_model"] = float(np.nanmean(rm[1]))
    out["rho_obs"] = rev2.rho_obs(r["pit"], [1])[1]
    return out


def section_euler(st, texdir, figdir):
    res = {}
    settings = ["euler5", "euler10", "euler25", "", "euler100", "euler200", "ode50", "sde50"]
    for ds in MAIN4:
        for m in ["sundial", "aurora"]:
            for s in settings:
                runs = st.runs(ds, 24, m, s)[:1]
                if runs:
                    res[f"{ds}|{m}|{s or 'euler50'}"] = setting_scores(runs[0])
    OUT["euler"] = res
    if not res:
        return
    steps = [5, 10, 25, 50, 100, 200]
    body = []
    for m in ["sundial", "aurora"]:
        for s in ["euler5", "euler10", "euler25", "euler50", "euler100", "euler200", "ode50", "sde50"]:
            cells = []
            for ds in MAIN4:
                v = res.get(f"{ds}|{m}|{s}")
                cells += ["--"] * 3 if v is None else [fmt(v["ql"]), pct(v["cov80"]),
                                                       f"{v.get('rho_model', float('nan')):.2f} / {v['rho_obs']:.2f}"]
            if any(c != "--" for c in cells):
                lab = {"ode50": "ODE, 50", "sde50": "SDE, 50"}.get(s, s.replace("euler", "Released, "))
                body.append(f"{SHORT[m]}, {lab} & " + " & ".join(cells) + r" \\")
    head = [" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{DSN[d]}}}" for d in MAIN4) + r" \\",
            "Model, sampler, steps" + r" & QL & Cov. & $\rho_{\mathrm{m}}$ / $\rho_{\mathrm{o}}$" * len(MAIN4) + r" \\"]
    write_tex(texdir, "euler", table(
        "Number of Sampling Steps of the Flow Heads, 24-Step Forecasts", "stab:rev2euler",
        "@{}l" + " r r c" * len(MAIN4) + "@{}", head, body,
        "Released, $K$: the released sampler with $K$ steps (50 by default). ODE and SDE: the samplers of "
        "Section~\\ref{snote:rev}, 50 steps. One seed. Cov.: coverage (\\%) of the central 80\\% interval; "
        "$\\rho_{\\mathrm{m}}$ / $\\rho_{\\mathrm{o}}$: lag-one rank correlations, Eqs.~(\\ref{eq:rhomodel}) and (\\ref{eq:rhoobs})."))
    import figstyle
    import matplotlib.pyplot as plt
    figstyle.apply()
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2), squeeze=False)
    xo = {"ode50": 600, "sde50": 1600}            # the two other samplers (50 steps), right of the released one
    for m, ls in [("sundial", "-"), ("aurora", "--")]:
        for ds in MAIN4:
            xs = [k for k in steps if f"{ds}|{m}|euler{k}" in res]
            if not xs:
                continue
            for a, key, f in [(ax[0, 0], "ql", 1), (ax[0, 1], "cov80", 100), (ax[0, 2], "rho_model", 1)]:
                a.plot(xs, [f * res[f"{ds}|{m}|euler{k}"].get(key, np.nan) for k in xs], ls=ls, marker="o", ms=2,
                       lw=1, color=DSCOLOR[ds], label=f"{SHORT[m]}, {DSF[ds]}")
                pts = [(x, f * res[f"{ds}|{m}|{s}"].get(key, np.nan)) for s, x in xo.items()
                       if f"{ds}|{m}|{s}" in res]
                if pts:
                    a.plot(*zip(*pts), ls="none", marker="o" if m == "sundial" else "^", ms=3, mew=0.8,
                           mfc="none", color=DSCOLOR[ds])
    from matplotlib.ticker import NullLocator
    for a, lab in zip(ax[0], ["QL", "Coverage of 80% interval (%)", r"$\rho_{\mathrm{model}}$, lag 1"]):
        a.set_xscale("log")
        a.xaxis.set_minor_locator(NullLocator())
        a.set_xticks(steps + list(xo.values()))
        a.set_xticklabels(["" if s == 100 else str(s) for s in steps] + ["ODE", "SDE"])    # 100 would touch 200
        a.axvline(340, color=figstyle.MUTED, lw=0.5, ls=":")
        a.set_xlabel("Sampling steps / sampler")
        a.set_ylabel(lab)
    ax[0, 1].axhline(80, color=figstyle.INK, lw=0.6, ls=":")
    fig.subplots_adjust(wspace=0.42)
    fig_legend(fig, ax, ncol=4)
    fig.savefig(os.path.join(figdir, "rev2_euler.pdf"))
    plt.close(fig)


def section_chunked(st, texdir, figdir):
    res = {}
    settings = ["", "chunk64median", "chunk24median", "chunk64paths"]
    names = {"": "One pass", "chunk64median": "64-step chunks, median fed back",
             "chunk24median": "24-step chunks, median fed back", "chunk64paths": "64-step chunks, paths fed back"}
    for ds in ["ercot", "traffic", "ausel"]:
        for s in settings:
            runs = st.runs(ds, 720, "chronos2", s)[:1]
            if not runs:
                continue
            r = runs[0]
            day = int(r["meta"].get("season") or (48 if ds == "ausel" else 24))
            qlh = rev2.quantile_loss_lead(r["q"], r["levels"], r["y"]).mean(0)
            cvh = covered(r, 0.1, 0.9).mean(0)
            nd = len(qlh) // day
            res[f"{ds}|{s or 'onepass'}"] = dict(
                ql=float(qlh.mean()), cov80=float(cvh.mean()),
                ql_day=qlh[:nd * day].reshape(nd, day).mean(1).tolist(),
                cov_day=cvh[:nd * day].reshape(nd, day).mean(1).tolist(),
                rho_obs=rev2.rho_obs(r["pit"], [1])[1],
                rho_model=float(np.nanmean(rev2.rho_model_windows(r["paths"], [1])[1])) if "paths" in r else None)
    OUT["chunked"] = res
    if not res:
        return
    dss = [d for d in ["ercot", "traffic", "ausel"] if any(k.startswith(d + "|") for k in res)]
    body = []
    for s in settings:
        cells = []
        for ds in dss:
            v = res.get(f"{ds}|{s or 'onepass'}")
            cells += ["--"] * 4 if v is None else [fmt(v["ql"]), pct(v["cov80"]), fmt(v["ql_day"][-1]),
                                                   pct(v["cov_day"][-1])]
        body.append(names[s] + " & " + " & ".join(cells) + r" \\")
    head = [" & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{DSN[d]}}}" for d in dss) + r" \\",
            "Chronos-2" + " & QL & Cov. & QL$_{\\mathrm{last}}$ & Cov.$_{\\mathrm{last}}$" * len(dss) + r" \\"]
    write_tex(texdir, "chunked", table(
        "Chronos-2 at $H = 720$ in One Pass and Rolled Out in Chunks", "stab:rev2chunk",
        "@{}l" + " r r r r" * len(dss) + "@{}", head, body,
        "150 windows, context 512. QL and Cov.\\ (\\%, central 80\\% interval) over the 720 steps; "
        "QL$_{\\mathrm{last}}$ and Cov.$_{\\mathrm{last}}$: the same on the last forecast day."))
    import figstyle
    import matplotlib.pyplot as plt
    figstyle.apply()
    fig, ax = plt.subplots(2, len(dss), figsize=(7.16, 3.2), sharex="col", squeeze=False)
    cols = {"": figstyle.ORANGE, "chunk64median": figstyle.BLUE, "chunk24median": figstyle.PURPLE,
            "chunk64paths": figstyle.GREEN}
    for i, ds in enumerate(dss):
        for s in settings:
            v = res.get(f"{ds}|{s or 'onepass'}")
            if v is None:
                continue
            d = np.arange(1, len(v["ql_day"]) + 1)
            ax[0, i].plot(d, v["ql_day"], color=cols[s], lw=1, label=names[s])
            ax[1, i].plot(d, 100 * np.array(v["cov_day"]), color=cols[s], lw=1)
        ax[1, i].axhline(80, color=figstyle.INK, lw=0.6, ls=":")
        ax[0, i].set_title(DSF[ds])
        ax[1, i].set_xlabel("Forecast day")
    ax[0, 0].set_ylabel("QL")
    ax[1, 0].set_ylabel("Coverage of 80% interval (%)")
    fig_legend(fig, ax)
    fig.savefig(os.path.join(figdir, "rev2_chunked.pdf"))
    plt.close(fig)


def section_context(st, texdir):
    res = {}
    for ds in MAIN4:
        for m in ["timesfm", "chronos2"]:
            for s in ["", "ctx2048", "ctx4096"]:
                runs = st.runs(ds, 24, m, s)[:1]
                if runs:
                    v = basic(runs[0])
                    res[f"{ds}|{m}|{s or 'ctx512'}"] = dict(ql=float(v["ql"].mean()), cov80=float(v["cov80"].mean()),
                                                           crps=float(v["crps"].mean()))
    OUT["context"] = res
    if not res:
        return
    body = []
    for m in ["timesfm", "chronos2"]:
        for s in ["ctx512", "ctx2048", "ctx4096"]:
            cells = []
            for ds in MAIN4:
                v = res.get(f"{ds}|{m}|{s}")
                cells += ["--", "--"] if v is None else [fmt(v["ql"]), pct(v["cov80"])]
            if any(c != "--" for c in cells):
                body.append(f"{SHORT[m]}, {int(s[3:]):,} & " + " & ".join(cells) + r" \\")
    head = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{DSN[d]}}}" for d in MAIN4) + r" \\",
            "Model, context" + " & QL & Cov." * len(MAIN4) + r" \\"]
    write_tex(texdir, "context", table(
        "Longer Contexts for TimesFM 2.5 and Chronos-2, 24-Step Forecasts", "tab:rev2ctx",
        "@{}l" + " r r" * len(MAIN4) + "@{}", head, body,
        "The windows of Table~\\ref{tab:real} with 512 (as there), 2,048, and 4,096 context steps. For AEMO 2026 the context is "
        "capped at the history available from December 2025 (at least 2,880 steps).", wide=False))


# =====================================================================
# 9 Post-processing: spread correction and copulas
# =====================================================================

def section_postprocessing(st, texdir):
    res = {}
    for ds in ALL24:
        for m in st.models(ds, 24, "spread"):
            a, b = st.runs(ds, 24, m)[:1], st.runs(ds, 24, m, "spread")[:1]
            if a and b:
                sa, sb = setting_scores(a[0]), setting_scores(b[0])
                sp = b[0]["meta"].get("spread", {})
                res[f"spread|{ds}|{m}"] = dict(before=sa, after=sb, c=sp.get("c", sp.get("c_by_month")))
        for m in QMODELS:
            base = st.runs(ds, 24, m)[:1]
            if not base:
                continue
            for s in ["schaake", "gausscop"]:
                runs = st.runs(ds, 24, m, s)[:1]
                if not runs:
                    continue
                r = runs[0]
                xi = with_paths(base[0])
                v = dict(es=float(energy_score(r["paths"], r["y"]).mean()), vs=float(variogram_score(r["paths"], r["y"]).mean()),
                         es_indep=float(energy_score(xi, r["y"]).mean()), vs_indep=float(variogram_score(xi, r["y"]).mean()),
                         rho_model=float(np.nanmean(rev2.rho_model_windows(r["paths"], [1])[1])),
                         rho_obs=rev2.rho_obs(r["pit"], [1])[1], n_hist_min=r["meta"].get("n_hist_min"))
                res[f"copula|{ds}|{m}|{s}"] = v
    OUT["postprocessing"] = res
    if not res:
        return
    body = []
    for k, v in res.items():
        if not k.startswith("spread|"):
            continue
        _, ds, m = k.split("|")
        c = v["c"]
        cs = f"{c:.2f}" if isinstance(c, (int, float)) else (f"{min(c.values()):.2f}--{max(c.values()):.2f}" if c else "--")
        b, a = v["before"], v["after"]
        body.append(f"{DSN[ds]} & {SHORT[m]} & {cs} & {pct(b['cov80'])} $\\to$ {pct(a['cov80'])} & "
                    f"{fmt(b['ql'])} $\\to$ {fmt(a['ql'])} & {fmt(b['crps'])} $\\to$ {fmt(a['crps'])} & "
                    f"{fmt(b.get('rho_model'), 2)} $\\to$ {fmt(a.get('rho_model'), 2)} & "
                    f"{fmt(b['rho_obs'], 2)} $\\to$ {fmt(a['rho_obs'], 2)} \\\\")
    if body:
        write_tex(texdir, "spread", table(
            "Spread Correction of the Sample Paths", "stab:rev2spread", "@{}l l r c c c c c@{}",
            [r"Dataset & Model & $c$ & Cov. (\%) & QL & CRPS & $\rho_{\mathrm{model}}$ & $\rho_{\mathrm{obs}}$ \\"],
            body,
            "Calibration windows for $c$: 150 windows from 60--80\\% of each held-out series, and the other months "
            "for the newer data (range of $c$ over the months). Before $\\to$ after, seed 0. Cov.: coverage of the central 80\\% interval; $\\rho_{\\mathrm{model}}$ and $\\rho_{\\mathrm{obs}}$: lag-one rank "
            "correlations."))
    body = []
    for k, v in res.items():
        if not k.startswith("copula|"):
            continue
        _, ds, m, s = k.split("|")
        body.append(f"{DSN[ds]} & {SHORT[m]}{VARIANT[s]} & {fmt(v['es_indep'], 2)} $\\to$ {fmt(v['es'], 2)} & "
                    f"{fmt(v['vs_indep'], 3)} $\\to$ {fmt(v['vs'], 3)} & {fmt(v['rho_model'], 2)} & "
                    f"{fmt(v['rho_obs'], 2)} & {v['n_hist_min']} \\\\")
    if body:
        write_tex(texdir, "copula", table(
            "Copulas for the Quantile Models", "stab:rev2copula", "@{}l l c c r r r@{}",
            [r"Dataset & Forecast & ES & VS & $\rho_{\mathrm{model}}$ & $\rho_{\mathrm{obs}}$ & Min. traj. \\"], body,
            "ES and VS ($p = 0.5$) of 100 paths drawn independently at each step "
            "$\\to$ after joining the draws with the copula of the most recent (up to 100) past trajectories of the same "
            "series that start at the same time of day. Lag-one rank correlations of the copula paths. Min.\\ traj.: "
            "smallest number of past trajectories available for a window."))


COLOR, LS, DSCOLOR = {}, {}, {}
DSF = {"ercot": "ERCOT", "traffic": "Traffic", "ausel": "Australia 2002–15", "aemo": "AEMO 2026",
       "aemo_mid": "AEMO 2016–19", "ausel_season": "Australia, Jan–Sep", "ercot2026": "ERCOT 2026"}


def fig_legend(fig, ax, ncol=None):
    """One legend for all panels, above them (each label once)."""
    seen = {}
    for a in np.ravel(ax):
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    if seen:
        fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=ncol or min(5, len(seen)),
                   fontsize=6, bbox_to_anchor=(0.5, 1.0))


def _colors():
    import figstyle
    COLOR.update({"naive": figstyle.GREY, "chronos": figstyle.BLUE, "chronos_base": figstyle.darker(figstyle.BLUE),
                  "toto": figstyle.CYAN, "timesfm": figstyle.PURPLE, "bolt": figstyle.ORANGE,
                  "bolt_base": figstyle.darker(figstyle.ORANGE), "chronos2": figstyle.BROWN,
                  "sundial": figstyle.GREEN, "aurora": figstyle.RED})
    LS.update({m: "--" for m in QMODELS})
    DSCOLOR.update({"ercot": figstyle.BLUE, "traffic": figstyle.ORANGE, "ausel": figstyle.GREEN,
                    "aemo": figstyle.RED})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev2dir", default=rev2.REV2)
    ap.add_argument("--texdir", default=os.path.join(HERE, "..", "outputs", "tables"))
    ap.add_argument("--figdir", default=os.path.join(HERE, "..", "outputs", "figures"))
    ap.add_argument("--json", default=None, help="default: <rev2dir>/report.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--sections", nargs="+", default=["calibration", "dependence", "aggregates", "leakage",
                                                       "decision", "euler", "chunked", "context", "postprocessing"])
    a = ap.parse_args()
    _colors()
    os.makedirs(a.figdir, exist_ok=True)
    st = Store(a.rev2dir)
    print(f"{len(st.idx)} rev2 files in {a.rev2dir}", flush=True)
    cpath = os.path.join(a.rev2dir, "data_checks.json")
    if not os.path.exists(cpath):
        cpath = os.path.join(rev2.REV2, "data_checks.json")
    checks = json.load(open(cpath)) if os.path.exists(cpath) else {}
    OUT["files"] = [os.path.basename(p) for p, _ in st.idx]
    OUT["data_checks"] = checks
    for s in a.sections:
        print(f"== {s}", flush=True)
        if s == "calibration":
            section_calibration(st, a.texdir, a.figdir)
        elif s == "dependence":
            section_dependence(st, a.texdir, a.figdir, a.boot // 2)
        elif s == "aggregates":
            section_aggregates(st, a.texdir)
        elif s == "leakage":
            section_leakage(st, a.texdir, a.figdir, a.boot, checks)
        elif s == "decision":
            section_decision(st, a.texdir)
        elif s == "euler":
            section_euler(st, a.texdir, a.figdir)
        elif s == "chunked":
            section_chunked(st, a.texdir, a.figdir)
        elif s == "context":
            section_context(st, a.texdir)
        elif s == "postprocessing":
            section_postprocessing(st, a.texdir)
    jpath = a.json or os.path.join(a.rev2dir, "report.json")
    with open(jpath, "w") as f:
        json.dump(OUT, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"written {jpath}", flush=True)


if __name__ == "__main__":
    main()
