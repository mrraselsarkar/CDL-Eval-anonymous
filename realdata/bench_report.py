"""Tables and figures of the revised benchmark (bench.py).

Reads results/bench_*.json and results/bench_*_perwindow.npz (all environments), the second-revision runs of
EXTRA_RUNS (Chronos base on AEMO 2026), results/robust.json (small forecasters), results/sundial_official.json, and
writes
  ../outputs/tables/real_table.tex                Table IV: held-out data, H = 24
  ../outputs/figures/accuracy.pdf                   Fig. 3: quantile loss and coverage, H = 24
  ../outputs/figures/dependence.pdf                 Fig. 4: dependence along the paths and the shuffle test
  ../outputs/figures/horizon.pdf                    Fig. 5: accuracy by lead time at H = 720 and coverage by horizon
  ../outputs/figures/leakage.pdf                    earlier leakage figure, not in the paper (Fig. 6: revision_report.py)
  ../outputs/tables/fresh_table.tex          Table S9: leakage test on Australian demand, 2002-2015 and 2026
  ../outputs/tables/bench_all_table.tex      Table S13: all scores at H = 24, with spread over sampling seeds
  ../outputs/tables/bench_long_table.tex     Table S2: scores at H = 192 and 720
  ../outputs/tables/bench_ci_table.tex       Table S3: cluster-bootstrap intervals for the main comparisons
  ../outputs/tables/bench_dep_table.tex      Table S4: shuffle gain with intervals and rank dependence
  ../outputs/tables/sundial_check_table.tex  Table S7: Sundial, forward pass against its generation routine
and prints the numbers quoted in the text.

Every bootstrap resamples calendar weeks of the forecast origin across all series of a dataset (cluster_stats.py,
which needs results/window_weeks.npz; `python cluster_stats.py` writes it).
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEC = SUPP = os.path.join(HERE, "..", "outputs", "tables")
FIG = os.path.join(HERE, "..", "outputs", "figures")
sys.path.insert(0, os.path.join(HERE, ".."))
import figstyle  # noqa: E402
from cluster_stats import ratio_draws, series_week_blocks, weeks  # noqa: E402

figstyle.apply()
DS = {"ercot": "ERCOT load", "traffic": "Traffic occupancy", "ausel": "Australian electricity"}
FM = ["chronos", "chronos_base", "toto", "timesfm", "bolt", "bolt_base", "chronos2", "sundial", "aurora"]
PATHS = {"naive", "chronos", "chronos_base", "toto", "sundial", "aurora"}       # models with their own paths
LABEL = {"naive": "Seasonal naive with error paths", "chronos": "Chronos small (AR, tokens)",
         "chronos_base": "Chronos base (AR, tokens)", "toto": "Toto (AR, patches)",
         "timesfm": "TimesFM 2.5 (AR, quantiles)", "bolt": "Chronos-Bolt small (direct)",
         "bolt_base": "Chronos-Bolt base (direct)", "chronos2": "Chronos-2 (direct)",
         "sundial": "Sundial (AR + flow head)", "aurora": "Aurora (full-window flow)",
         "ar": "AR, Gaussian head (trained)", "flow": "Full-window flow matching (trained)"}
SHORT = {"naive": "Naive", "chronos": "Chronos small", "chronos_base": "Chronos base", "toto": "Toto",
         "timesfm": "TimesFM 2.5", "bolt": "Chronos-Bolt small", "bolt_base": "Chronos-Bolt base",
         "chronos2": "Chronos-2", "sundial": "Sundial", "aurora": "Aurora"}
TYPEKEY = {"chronos": "chronos", "chronos_base": "chronos", "toto": "toto", "timesfm": "timesfm_ind",
           "bolt": "bolt_ind", "bolt_base": "bolt_ind", "chronos2": "chronos2_ind", "sundial": "sundial",
           "aurora": "aurora"}
LEAK = {("toto", "ercot"): "a", ("toto", "traffic"): "b", ("toto", "ausel"): "b",
        ("timesfm", "traffic"): "b", ("timesfm", "ausel"): "b",
        ("sundial", "traffic"): "c", ("sundial", "ausel"): "c", ("sundial", "ercot"): "c",
        ("chronos2", "traffic"): "c", ("chronos2", "ausel"): "c", ("chronos2", "ercot"): "c",
        ("aurora", "ercot"): "d", ("aurora", "ausel"): "d"}
LEAK_TEST = {("aurora", "ausel"): "e"}       # leakage indicated by the test of Section VI-E1 (Table S9)
B = 5000
CLUSTER = "week"      # bootstrap clusters: calendar weeks across all series ("series_week": the earlier blocks)
# Runs of the second revision that complete the 24-step benchmark, read from the summary files that bench.py writes
# with --store rev2 (results/rev2/bench_<tag>.json and _perwindow.npz; per-window scores of bench.window_scores, as
# in the files above): Chronos base on AEMO 2026 (seed 0, as on the other datasets). The rev2 runs of the other
# models on AEMO 2026 reproduce their stored per-window scores exactly.
EXTRA_RUNS = ["rev2/bench_rev2_aemo_chronos_base"]


def load():
    """All result files; a later file (in name order, e.g. bench_main_fix after bench_main) overrides rows and
    per-window scores of the same dataset, horizon, model and seed. The runs of EXTRA_RUNS are read first (so a
    result file of results/ overrides them), and their windows must be those of the other files."""
    rows, pw, ids = {}, {}, {}
    for base in EXTRA_RUNS:
        for r in json.load(open(os.path.join(RES, base + ".json"))):
            rows[(r["dataset"], r["horizon"], r["model"], r["seed"])] = r
        z = dict(np.load(os.path.join(RES, base + "_perwindow.npz")))
        ids.update({k: v for k, v in z.items() if k.endswith("|sid") or k.endswith("|block")})
        pw.update(z)
    for f in sorted(glob.glob(os.path.join(RES, "bench_*.json"))):
        for r in json.load(open(f)):
            rows[(r["dataset"], r["horizon"], r["model"], r["seed"])] = r
    for f in sorted(glob.glob(os.path.join(RES, "bench_*_perwindow.npz"))):
        pw.update(dict(np.load(f)))
    for k, v in ids.items():
        assert np.array_equal(pw[k], v), f"{k}: the windows of {EXTRA_RUNS} differ from those of results/bench_*"
    return list(rows.values()), pw


def sel(rows, ds, h, m):
    return [r for r in rows if r["dataset"] == ds and r["horizon"] == h and r["model"] == m]


def mean(rows, ds, h, m, k):
    v = [r[k] for r in sel(rows, ds, h, m) if k in r]
    return float(np.mean(v)) if v else float("nan")


def sd(rows, ds, h, m, k):
    v = [r[k] for r in sel(rows, ds, h, m) if k in r]
    return float(np.std(v)) if len(v) > 1 else float("nan")


def win(pw, ds, h, m, k):
    """Per-window score averaged over the sampling seeds."""
    keys = [x for x in pw if x.startswith(f"{ds}|{h}|{m}|") and x.endswith(f"|{k}")]
    return np.mean([pw[x] for x in keys], 0) if keys else None


def clusters(pw, ds, h):
    """Bootstrap cluster of each test window: the calendar week of its forecast origin, shared by all series of
    the dataset (CLUSTER = "series_week" gives the earlier blocks of one series and one week)."""
    return weeks(pw, ds, h) if CLUSTER == "week" else series_week_blocks(pw, ds, h)


def n_weeks(pw, ds, h=24):
    return len(np.unique(weeks(pw, ds, h)))


def boot_ratio(pw, ds, h, a, b, k="ql", rng=None, nboot=B):
    """Ratio of mean scores of a and b with a cluster bootstrap over calendar weeks (all series of the dataset)."""
    xa, xb = win(pw, ds, h, a, k), win(pw, ds, h, b, k)
    if xa is None or xb is None:
        return None
    rng = rng or np.random.default_rng(0)
    r = ratio_draws(clusters(pw, ds, h), xa, xb, nboot, rng)
    return xa.mean() / xb.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5), r


def nz(v, nd=2):
    return 0.0 if round(v, nd) == 0 else v           # no negative zero at nd decimals


def fmt_cell(v, nd, bold=False, mark=""):
    s = "--" if v != v else f"{v:.{nd}f}"
    s = rf"\textbf{{{s}}}" if bold and v == v else s
    return s + (rf"$^{{{mark}}}$" if mark else "")


# ---------------------------------------------------------------------------------------------- main table
MAIN_DS = {**DS, "aemo": "AEMO 2026"}      # datasets of the main table (the four of the ranking)


def table_main(rows, small):
    models = ["naive"] + [m for m in FM if sel(rows, "ercot", 24, m)]
    ncol = 1 + 3 * len(MAIN_DS)
    head = " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for name in MAIN_DS.values()) + r" \\"
    rules = "".join(rf"\cmidrule({'lr' if i + 1 < len(MAIN_DS) else 'l'}){{{2 + 3 * i}-{4 + 3 * i}}}"
                    for i in range(len(MAIN_DS)))
    L = [r"\begin{table*}[!t]", r"\centering", r"\caption{Foundation Models on Held-Out and 2026 Data, 24-Step Forecasts}",
         r"\label{tab:real}", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{fitwidth}", r"\begin{tabular}{@{}l" + " r" * (ncol - 1) + "@{}}", r"\toprule", head, rules,
         "Model" + " & QL & Cov. & ES" * len(MAIN_DS) + r" \\"]
    # bold: best among the baseline and the zero-shot models without known (a), probable (b) or test-indicated (e)
    # leakage, to the displayed precision (ties are all bold); the trained references are not ranked
    clean = lambda m, ds: LEAK.get((m, ds), "") not in ("a", "b") and (m, ds) not in LEAK_TEST
    marks = lambda m, ds: ",".join(x for x in [LEAK.get((m, ds), ""), LEAK_TEST.get((m, ds), "")] if x)
    best = {}
    for ds in MAIN_DS:
        for k, nd in [("ql", 3), ("es", 2)]:
            cand = [m for m in models if clean(m, ds) and (k == "ql" or m in PATHS)]
            low = min(round(mean(rows, ds, 24, m, k), nd) for m in cand)
            best[(ds, k)] = {m for m in cand if round(mean(rows, ds, 24, m, k), nd) == low}
    for group, ms in [("Baseline", ["naive"]), ("Foundation models, zero-shot", models[1:])]:
        L += [r"\midrule", rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{group}}}}} \\"]
        for m in ms:
            cells = []
            for ds in MAIN_DS:
                cells.append(fmt_cell(mean(rows, ds, 24, m, "ql"), 3, m in best[(ds, "ql")], marks(m, ds)))
                cells.append(f"{np.floor(100 * mean(rows, ds, 24, m, 'cov80') + 0.5):.0f}")
                cells.append(fmt_cell(mean(rows, ds, 24, m, "es"), 2, m in best[(ds, "es")],
                                      LEAK_TEST.get((m, ds), "")) if m in PATHS else "--")
            L.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"\midrule", rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{Trained on each dataset (reference)}}}} \\"]
    for m in ["ar", "flow"]:
        cells = []
        for ds in MAIN_DS:
            rr = [r for r in small if r["dataset"] == ds and r["model"].rsplit("_s", 1)[0] == m]
            cells += ([f"{np.mean([r['qloss'] for r in rr]):.3f}",
                       f"{np.floor(100 * np.mean([r['pit80'] for r in rr]) + 0.5):.0f}",
                       f"{np.mean([r['es'] for r in rr]):.2f}"] if rr else ["--"] * 3)     # not run on AEMO 2026
        L.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{fitwidth}", r"", r"\vspace{3pt}",
          r"\parbox{\textwidth}{\footnotesize QL: quantile loss (twice the mean pinball loss) at the levels 0.1 to 0.9, "
          r"which every model provides. Cov.: coverage (\%) of the central 80\% interval between these levels, ideally "
          r"80. ES: energy score of the sample paths, given for models that generate sample paths. Lower is "
          r"better for QL and ES\@. Scores are computed on 300 test windows per dataset, scaled by the last week of each "
          r"window's context. Foundation models were applied without fine-tuning to 512 context steps; sample-based "
          r"models drew 100 paths per window and are averaged over three sampling seeds (one for Chronos base, and for "
          r"Chronos small on AEMO 2026), and Table~\ref{S-stab:benchall} gives the spread. Quantile models enter the QL "
          r"and the coverage with the quantiles at the levels 0.1 to 0.9; the native levels of these models serve "
          r"solely for the scores of Table~\ref{S-stab:benchall} that need the whole distribution (CRPS, VS, and "
          r"Cov.\ 90). The two small forecasters of "
          r"Section~\ref{S-snote:sim} were trained on each held-out dataset with 48 context steps (three seeds); "
          r"Table~\ref{tab:extra} gives versions with 512 context steps, which also cover AEMO 2026. In bold: the "
          r"best QL and ES among the baseline and the zero-shot foundation models. Entries with known or probable "
          r"leakage ($^{a}$ and $^{b}$) and with test-indicated leakage ($^{e}$) are left out. Ties at the displayed "
          r"precision are all in bold; the trained references are not ranked. "
          r"The type labels refer to each model's decoder; at 24 steps every model except Chronos (both sizes) "
          r"produces the horizon in one decoding pass, as Table~\ref{tab:models} shows. "
          r"Leakage marks from Table~\ref{tab:models}: $^{a}$ in the pretraining corpus; $^{b}$ probably in it; "
          r"$^{c}$ exclusion stated by the authors while the corpus contains copies of the data; $^{d}$ not known. "
          r"Mark $^{e}$: leakage indicated by the test of Section~\ref{sec:leakage}. The AEMO 2026 data post-date every "
          r"model.}",
          r"\end{table*}"]
    open(os.path.join(SEC, "real_table.tex"), "w").write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------------------- fresh-data test
def fresh_results(rows, pw):
    """Leakage test: QL ratio to Chronos-Bolt small on 2002-2015 and 2026, and the change Delta, with independent
    cluster bootstraps of the two periods. Chronos base, run on AEMO 2026 only in the second revision (EXTRA_RUNS),
    is left out: the test keeps the models it was made with, since a further model would change the random draws of
    the models after it (revision_report.py gives the test for Chronos base)."""
    models = [m for m in FM + ["naive"] if m not in ("bolt", "chronos_base") and sel(rows, "aemo", 24, m)
              and sel(rows, "ausel", 24, m)]
    rng = np.random.default_rng(0)
    out = {}
    for m in models:
        old = boot_ratio(pw, "ausel", 24, m, "bolt", rng=rng)
        new = boot_ratio(pw, "aemo", 24, m, "bolt", rng=rng)
        if old is None or new is None:
            continue
        ch = new[3] / old[3] - 1
        out[m] = (old, new, np.percentile(ch, [2.5, 97.5]))
    return out


def table_fresh(rows, pw):
    out = fresh_results(rows, pw)
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.12}",
         r"\caption{\textbf{A test for leakage on Australian electricity demand.} Quantile loss of each model "
         r"divided by that of Chronos-Bolt small, which was trained without these data, on the Monash series "
         r"(2002--2015, contained in the GIFT-Eval and LOTSA pretraining collections) and on the same five states in "
         r"January to September 2026 from AEMO, which post-date every model. Values below 1 favor the model. "
         r"Change: relative change $\Delta$ of the ratio from 2002--2015 to 2026. 95\% intervals from a cluster "
         r"bootstrap that resamples calendar weeks of the forecast origin across the five states "
         rf"({n_weeks(pw, 'ausel')} weeks in 2002--2015 and {n_weeks(pw, 'aemo')} in 2026; {B:,} resamples), "
         r"independently for the two periods. A model that learned the old series should lose its advantage on the "
         r"new data; a model without this exposure should keep it.}",
         r"\label{stab:fresh}", r"\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{@{}l l r r r@{}}", r"\toprule",
         r"Model & Leakage of the 2002--2015 series & 2002--2015 & 2026 & Change $\Delta$ \\", r"\midrule"]
    leak_txt = {"a": "In corpus", "b": "Probably in corpus", "c": "Exclusion stated; copies in corpus",
                "d": "Not known", "": "Excluded"}
    for m, (old, new, ch) in out.items():
        L.append(f"{SHORT[m]} & {leak_txt[LEAK.get((m, 'ausel'), '')]} & {old[0]:.2f} ({old[1]:.2f}, {old[2]:.2f}) & "
                 f"{new[0]:.2f} ({new[1]:.2f}, {new[2]:.2f}) & ${100 * (new[0] / old[0] - 1):+.0f}\\%$ "
                 f"(${100 * ch[0]:+.0f}$, ${100 * ch[1]:+.0f}$) \\\\")
        print(f"fresh {m:12s} old {old[0]:.2f} [{old[1]:.2f},{old[2]:.2f}] new {new[0]:.2f} [{new[1]:.2f},{new[2]:.2f}] "
              f"change {100*(new[0]/old[0]-1):+.0f}% CI [{100*ch[0]:+.0f},{100*ch[1]:+.0f}]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "fresh_table.tex"), "w").write("\n".join(L) + "\n")
    return out


# ---------------------------------------------------------------------------------------------- figure 6
def bar_style(m):
    base = figstyle.MODEL_COLOR.get(TYPEKEY.get(m, m), figstyle.GREY)
    shade = {"chronos": 0.62, "chronos_base": 0.45, "timesfm": 1.35, "bolt_base": 0.8, "chronos2": 0.62}.get(m, 1.0)
    fill = figstyle.darker(base, shade) if shade < 1 else figstyle.lighter(base, shade - 1) if shade > 1 else base
    return fill, figstyle.MODEL_HATCH.get(TYPEKEY.get(m, m), "")


def fig_real(rows):
    models = ["naive"] + [m for m in FM if sel(rows, "ercot", 24, m)]
    pathm = [m for m in models if m in PATHS]
    figstyle.MODEL_COLOR.setdefault("naive", figstyle.GREY)
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.0))
    w = 0.8 / len(models)
    for r_, (key, yl) in enumerate([("ql", "Quantile loss (lower is better)"), ("cov80", "Coverage of 80% intervals (%)")]):
        ax = axes[r_]
        for i, m in enumerate(models):
            hs = [mean(rows, ds, 24, m, key) * (100 if key == "cov80" else 1) for ds in DS]
            x = np.arange(len(DS)) + (i - (len(models) - 1) / 2) * w
            labels = [f"{np.floor(h + 0.5):.0f}" for h in hs] if key == "cov80" else [f"{h:.3f}" for h in hs]
            fill, hatch = bar_style(m)
            figstyle.bars(ax, x, hs, w * 0.9, fill, hatch, labels=labels, fontsize=5.0, rotation=90, pad=1.0)
        if key == "cov80":
            ax.axhline(80, color=figstyle.INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.set_ylim(0, 108)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        ax.set_xticks(range(len(DS)))
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(yl)
        ax.set_title(f"{'ab'[r_]}  " + ["Probabilistic accuracy", "Calibration"][r_])
    ax = axes[2]
    w2 = 0.8 / len(pathm)
    for i, m in enumerate(pathm):
        x = np.arange(len(DS)) + (i - (len(pathm) - 1) / 2) * w2
        hm = [mean(rows, ds, 24, m, "rho_model") for ds in DS]
        ho = [mean(rows, ds, 24, m, "rho_obs") for ds in DS]
        fill, hatch = bar_style(m)
        figstyle.bars(ax, x, hm, w2 * 0.9, fill, hatch, labels=[f"{h:.2f}" for h in hm], fontsize=5.0, rotation=90, pad=1.0)
        ax.scatter(x, ho, marker="_", s=90, color=figstyle.INK, linewidths=1.6, zorder=5)
    ax.set_ylim(-0.1, 1.15)
    ax.axhline(0, color=figstyle.INK, lw=0.6)
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels(list(DS.values()))
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Rank correlation of adjacent steps")
    ax.set_title("c  Dependence along the sample paths (bars: model; dashes: observed)")
    handles = [figstyle.patch(*bar_style(m), LABEL[m]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.985), columnspacing=1.0)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(os.path.join(FIG, "real.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- figure 7
def fig_horizon(rows):
    h = 720
    models = [m for m in ["naive", "toto", "timesfm", "bolt", "chronos2", "sundial", "aurora"] if sel(rows, "ercot", h, m)]
    if not models:
        return
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=False)
    for c, ds in enumerate(DS):
        ax = axes[c]
        for m in models:
            rr = sel(rows, ds, h, m)
            if not rr:
                continue
            v = np.array(rr[0]["lead_ql"])
            day = 24 if ds != "ausel" else 48
            v = v[: (len(v) // day) * day].reshape(-1, day).mean(1)                # daily means
            fill, _ = bar_style(m)
            ax.plot(np.arange(1, len(v) + 1), v, color=fill, lw=1.2, label=SHORT[m],
                    ls="--" if m == "naive" else "-")
        ax.set_title(f"{'abc'[c]}  {DS[ds]}")
        ax.set_xlabel("Forecast day")
        if c == 0:
            ax.set_ylabel("Quantile loss")
    axes[0].legend(fontsize=6.2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "horizon.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------------------------- SI tables
def table_all(rows):
    """Scores of the 24-step benchmark that Table tab:real leaves out (its QL, Cov. 80 and ES are not repeated; their
    spread over the sampling seeds is given here). The sequential calls are those of Table tab:models."""
    models = ["naive"] + [m for m in FM if sel(rows, "ercot", 24, m)]
    head = r"Model & CRPS & VS & Cov.\ 90 & SD of QL & SD of Cov.\ 80 \\"
    L = [r"{\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{longtable}{@{}l r r r r r@{}}",
         r"\caption{\textbf{Further scores of the benchmark, 24-step forecasts.} Table~\ref{tab:real} gives the QL, "
         r"the coverage of the central 80\% interval and the ES\@. Mean, with the standard deviation (SD) over three "
         r"sampling seeds in parentheses for sample-based models (one seed for Chronos base and for Chronos small on "
         r"AEMO 2026); the last two columns give this SD for scores of Table~\ref{tab:real}. For quantile models, the "
         r"CRPS and the VS use paths drawn independently at each step, and the VS depends on this sampling and is set "
         r"in italics. Cov.\ 90: coverage (\%) of the central 90\% interval.}",
         r"\label{stab:benchall}\\", r"\toprule", head, r"\endfirsthead",
         r"\multicolumn{6}{@{}l}{\textit{Table~\ref{stab:benchall} continued}}\\", r"\toprule", head,
         r"\endhead", r"\bottomrule", r"\endfoot"]
    for j, ds in enumerate(list(DS) + ["aemo"]):
        name = DS.get(ds, "Australian electricity, AEMO 2026")
        L += ([r"\midrule"] if j else []) + [rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{name}}}}} \\"]
        for m in models:
            if not sel(rows, ds, 24, m):
                continue
            def c(k, nd, scale=1):
                v, s = mean(rows, ds, 24, m, k) * scale, sd(rows, ds, 24, m, k) * scale
                t = f"{v:.{nd}f}" + (f" ({s:.{nd}f})" if s == s else "")
                return t if m in PATHS or k != "vs" else rf"\textit{{{t}}}"
            def s_only(k, nd, scale=1):
                s = sd(rows, ds, 24, m, k) * scale
                return f"{s:.{nd}f}" if s == s else "--"
            L.append(f"{LABEL[m]} & {c('crps', 3)} & {c('vs', 4)} & {c('cov90', 1, 100)} & "
                     f"{s_only('ql', 3)} & {s_only('cov80', 1, 100)} \\\\")
    L += [r"\end{longtable}", r"}"]
    open(os.path.join(SUPP, "bench_all_table.tex"), "w").write("\n".join(L) + "\n")


def table_long(rows):
    models = [m for m in ["naive", "chronos", "toto", "timesfm", "bolt", "chronos2", "sundial", "aurora"]]
    hs = [h for h in (24, 192, 720) if any(r["horizon"] == h for r in rows)]
    head = r"Model & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$H={h}$}}" for h in hs) + r" & Calls at $H=720$ \\"
    sub = r" & " + " & ".join(["QL & Cov." for _ in hs]) + r" & \\"
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.08}",
         r"\caption{\textbf{Longer horizons.} QL and coverage (\%) of central 80\% intervals over the "
         r"whole horizon, for $H = 24$ (300 windows) and $H = 192$ and 720 (150 windows each, one sampling seed). "
         r"Chronos small was run up to 192 steps; it generates one value per network call. Calls: sequential network "
         r"evaluations per forecast at $H = 720$.}",
         r"\label{stab:benchlong}", r"\begin{tabular}{@{}l " + "r r " * len(hs) + r"r@{}}", r"\toprule", head, sub]
    for ds in DS:
        L += [r"\midrule", rf"\multicolumn{{{2 * len(hs) + 2}}}{{@{{}}l}}{{\textit{{{DS[ds]}}}}} \\"]
        for m in models:
            cells = []
            for h in hs:
                q, c = mean(rows, ds, h, m, "ql"), mean(rows, ds, h, m, "cov80")
                cells += ["--" if q != q else f"{q:.3f}", "--" if c != c else f"{100 * c:.0f}"]
            calls = sel(rows, ds, 720, m)
            L.append(f"{SHORT[m]} & " + " & ".join(cells) + f" & {calls[0]['calls'] if calls else '--'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "bench_long_table.tex"), "w").write("\n".join(L) + "\n")


CI_COMPS = [("chronos2", "bolt"), ("chronos2", "toto"), ("chronos2", "timesfm"), ("toto", "chronos"),
            ("bolt_base", "bolt"), ("chronos_base", "chronos"), ("sundial", "bolt"), ("aurora", "sundial"),
            ("chronos2", "naive"), ("sundial", "naive")]
DEP_MODELS = ["naive", "chronos", "chronos_base", "toto", "sundial", "aurora"]


def ci_results(pw):
    """Table S3: QL ratios of the main comparisons with cluster-bootstrap draws (one random stream, table order)."""
    rng = np.random.default_rng(1)
    out = {}
    for ds in DS:
        for a, b in CI_COMPS:
            res = boot_ratio(pw, ds, 24, a, b, rng=rng)
            if res is not None:
                out[(ds, a, b)] = res
    return out


def gain_results(rows, pw, seed, nboot, model_first):
    """Shuffle gain mean(vs_shuf) / mean(vs) - 1 with cluster-bootstrap percentiles: Table S4 (datasets in the outer
    loop, seed 2, 5,000 resamples) and Fig. 4(b) (models in the outer loop, seed 3, 2,000 resamples)."""
    models = [m for m in DEP_MODELS if sel(rows, "ercot", 24, m)]
    rng = np.random.default_rng(seed)
    pairs = [(ds, m) for m in models for ds in DS] if model_first else [(ds, m) for ds in DS for m in models]
    out = {}
    for ds, m in pairs:
        vs, vss = win(pw, ds, 24, m, "vs"), win(pw, ds, 24, m, "vs_shuf")
        if vs is None or vss is None:
            continue
        r = ratio_draws(clusters(pw, ds, 24), vss, vs, nboot, rng) - 1
        out[(ds, m)] = (vss.mean() / vs.mean() - 1, np.percentile(r, 2.5), np.percentile(r, 97.5))
    return out


def pit_dependence(pw, ds, m, S=100):
    """From the stored randomized PIT values u = (#paths below + V) / (S + 1), V ~ U(0, 1): the share of outcomes
    outside the range of the paths (below all or above all), and the rank correlation of adjacent steps across
    windows computed from the mid-PIT (#below + 1/2) / (S + 1) with tied ranks averaged; mean over sampling seeds."""
    from scipy import stats
    keys = sorted(k for k in pw if k.startswith(f"{ds}|24|{m}|") and k.endswith("|pit"))
    out, rho = [], []
    for k in keys:
        below = np.clip(np.floor(pw[k].astype(np.float64) * (S + 1)), 0, S)
        out.append(np.mean((below == 0) | (below == S)))
        r = stats.rankdata((below + 0.5) / (S + 1), axis=0)
        r = (r - r.mean(0)) / r.std(0)
        rho.append(np.mean((r[:, :-1] * r[:, 1:]).mean(0)))
    return float(np.mean(out)), float(np.mean(rho))


def table_ci(rows, pw):
    res_all = ci_results(pw)
    nw = ", ".join(str(n_weeks(pw, ds)) for ds in list(DS)[:-1]) + f", and {n_weeks(pw, list(DS)[-1])}"
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.08}",
         r"\caption{\textbf{Uncertainty of the main comparisons, 24-step forecasts.} Relative difference in mean "
         r"quantile loss of model A and model B (negative values favor A), with 95\% intervals from the cluster "
         rf"bootstrap over calendar weeks of Section~\ref{{sec:aggregation}} ({B:,} resamples). Sample-based models "
         r"use scores averaged over the sampling seeds.}",
         r"\label{stab:benchci}", r"\begin{tabular}{@{}l l l r@{}}", r"\toprule",
         r"Data & Model A & Model B & A vs B (95\% CI) \\", r"\midrule"]
    for j, ds in enumerate(DS):
        if j:
            L.append(r"\addlinespace")
        for a, b in CI_COMPS:
            res = res_all.get((ds, a, b))
            if res is None:
                continue
            L.append(f"{DS[ds] if (a, b) == CI_COMPS[0] else ''} & {SHORT[a]} & {SHORT[b]} & "
                     f"${100 * (res[0] - 1):+.1f}\\%$ (${100 * (res[1] - 1):+.1f}$, ${100 * (res[2] - 1):+.1f}$) \\\\")
            print(f"ci {ds:8s} {a:12s} vs {b:12s} {100*(res[0]-1):+.1f}% [{100*(res[1]-1):+.1f}, {100*(res[2]-1):+.1f}]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "bench_ci_table.tex"), "w").write("\n".join(L) + "\n")


def table_dep(rows, pw):
    gains = gain_results(rows, pw, 2, B, model_first=False)
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.08}",
         r"\caption{\textbf{Dependence along the sample paths, 24-step forecasts.} Shuffle gain, "
         r"Eq.~(\ref{eq:shuffle}), with a 95\% interval from the cluster bootstrap of Table~\ref{stab:benchci}. "
         r"$\rho$ model and $\rho$ observed: Eqs.~(\ref{eq:rhomodel}) and~(\ref{eq:rhoobs}). Outside: share (\%) of "
         r"outcomes below or above all 100 paths. $\rho$ mid-PIT: $\rho$ observed from the mid-PIT with tied ranks "
         r"averaged. The table shows the models that generate sample paths.}",
         r"\label{stab:benchdep}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l l r r r r r@{}}",
         r"\toprule",
         r"Data & Model & Shuffle gain (\%) & $\rho$ model & $\rho$ observed & Outside (\%) & $\rho$ mid-PIT \\",
         r"\midrule"]
    for j, ds in enumerate(DS):
        if j:
            L.append(r"\addlinespace")
        first = True
        for m in DEP_MODELS:
            if (ds, m) not in gains:
                continue
            g, lo, hi = gains[(ds, m)]
            outside, rho_mid = pit_dependence(pw, ds, m)
            L.append(f"{DS[ds] if first else ''} & {SHORT[m]} & ${100 * g:+.1f}$ (${100 * lo:+.1f}$, "
                     f"${100 * hi:+.1f}$) & {nz(mean(rows, ds, 24, m, 'rho_model')):.2f} & "
                     f"{nz(mean(rows, ds, 24, m, 'rho_obs')):.2f} & {100 * outside:.0f} & {rho_mid:.2f} \\\\")
            first = False
            print(f"dep {ds:8s} {m:12s} gain {100*g:+.1f}% [{100*lo:+.1f},{100*hi:+.1f}] "
                  f"rho_model {mean(rows, ds, 24, m, 'rho_model'):.3f} rho_obs {mean(rows, ds, 24, m, 'rho_obs'):.3f} "
                  f"outside {100*outside:.1f}% rho_mid {rho_mid:.3f}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "bench_dep_table.tex"), "w").write("\n".join(L) + "\n")


def intervals(rows, pw):
    """All bootstrap intervals of Tables S3, S4, S9 and Fig. 4(b), in percent, for cluster_stats.compare."""
    out = {}
    for (ds, a, b), r in ci_results(pw).items():
        out[("S3", ds, a, b)] = tuple(100 * (v - 1) for v in r[:3])
    for (ds, m), r in gain_results(rows, pw, 2, B, model_first=False).items():
        out[("S4", ds, m)] = tuple(100 * v for v in r)
    for (ds, m), r in gain_results(rows, pw, 3, 2000, model_first=True).items():
        out[("Fig4b", ds, m)] = tuple(100 * v for v in r)
    for m, (old, new, ch) in fresh_results(rows, pw).items():
        out[("S8-2002-15", m)] = tuple(old[:3])
        out[("S8-2026", m)] = tuple(new[:3])
        out[("S8-Delta", m)] = (100 * (new[0] / old[0] - 1), 100 * ch[0], 100 * ch[1])
    return out


def table_sundial(rows):
    path = os.path.join(RES, "sundial_official.json")
    if not os.path.exists(path):
        return
    off = json.load(open(path))
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{\textbf{Sundial through its forward pass and through its released generation routine.} The "
         r"benchmark calls Sundial's forward pass with \nolinkurl{transformers} 4.57; the released routine "
         r"(\nolinkurl{generate}) was run in a separate environment with \nolinkurl{transformers} 4.40.1, the version "
         r"its model card requires. Same test windows, 24-step forecasts, mean over three sampling seeds.}",
         r"\label{stab:sundialcheck}", r"\begin{tabular}{@{}l l r r r@{}}", r"\toprule",
         r"Data & Interface & CRPS & Cov.\ 80 & Cov.\ 90 \\", r"\midrule"]
    for ds in DS:
        o = [r for r in off if r["dataset"] == ds]
        L.append(f"{DS[ds]} & Forward pass & {mean(rows, ds, 24, 'sundial', 'crps'):.3f} & "
                 f"{100 * mean(rows, ds, 24, 'sundial', 'cov80'):.1f} & {100 * mean(rows, ds, 24, 'sundial', 'cov90'):.1f} \\\\")
        L.append(f" & Generation routine & {np.mean([r['crps'] for r in o]):.3f} & {100 * np.mean([r['cov80'] for r in o]):.1f} & "
                 f"{100 * np.mean([r['cov90'] for r in o]):.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "sundial_check_table.tex"), "w").write("\n".join(L) + "\n")


def summary(rows):
    for h in (24, 192, 720):
        for ds in list(DS) + ["aemo"]:
            ms = sorted({r["model"] for r in rows if r["dataset"] == ds and r["horizon"] == h})
            if not ms:
                continue
            print(f"== {ds} H={h}")
            for m in ms:
                g = lambda k: mean(rows, ds, h, m, k)
                print(f"  {m:13s} ql={g('ql'):.3f} crps={g('crps'):.3f} es={g('es'):.3f} cov80={100*g('cov80'):.0f} "
                      f"cov90={100*g('cov90'):.0f} rho_m={g('rho_model'):.2f} rho_o={g('rho_obs'):.2f} "
                      f"gain={100*(g('vs_shuf')/g('vs')-1):+.1f}%")


# ------------------------------------------------------------------------------------- figures of Paper 2
def _dataset_axes(ax, labels=True):
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels(list(DS.values()) if labels else [])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(DS) - 0.45)


LEGEND = {"naive": "Seasonal naive", "chronos": "Chronos small (AR)", "chronos_base": "Chronos base (AR)",
          "toto": "Toto (AR)", "timesfm": "TimesFM 2.5 (AR)", "bolt": "Chronos-Bolt small (direct)",
          "bolt_base": "Chronos-Bolt base (direct)", "chronos2": "Chronos-2 (direct)",
          "sundial": "Sundial (AR + flow head)", "aurora": "Aurora (full-window flow)"}
DS_TICK = {"ercot": "ERCOT", "traffic": "Traffic", "ausel": "Australia"}
TEXT_W, COL_W = 7.16, 3.5      # inches: text width and column width of an IEEE journal page (figures print at 100%)


def _group_axes(ax):
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels([DS_TICK[d] for d in DS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(DS) - 0.5)


def _dashed(label):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=figstyle.INK, lw=0.8, ls=(0, (3, 2)), label=label)


LEGEND = {"naive": "Seasonal naive", "chronos": "Chronos small (AR)", "chronos_base": "Chronos base (AR)",
          "toto": "Toto (AR)", "timesfm": "TimesFM 2.5 (AR)", "bolt": "Chronos-Bolt small (direct)",
          "bolt_base": "Chronos-Bolt base (direct)", "chronos2": "Chronos-2 (direct)",
          "sundial": "Sundial (AR + flow head)", "aurora": "Aurora (full-window flow)"}
DS_TICK = {"ercot": "ERCOT", "traffic": "Traffic", "ausel": "Australia"}
TEXT_W, COL_W = 7.16, 3.5      # inches: text width and column width of an IEEE journal page (figures print at 100%)


def _group_axes(ax):
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels([DS_TICK[d] for d in DS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(DS) - 0.5)


def _dashed(label):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=figstyle.INK, lw=0.8, ls=(0, (3, 2)), label=label)


def fig_accuracy(rows):
    """Fig. 3: quantile loss and coverage of the central 80% interval, H = 24 (full text width)."""
    models = ["naive"] + [m for m in FM if sel(rows, "ercot", 24, m)]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45))
    w = 0.84 / len(models)
    for ax, key, title, yl in [(axes[0], "ql", "a  Accuracy", "Quantile loss"),
                               (axes[1], "cov80", "b  Calibration", "Coverage of 80% interval (%)")]:
        for i, m in enumerate(models):
            hs = [mean(rows, ds, 24, m, key) * (100 if key == "cov80" else 1) for ds in DS]
            x = np.arange(len(DS)) + (i - (len(models) - 1) / 2) * w
            figstyle.bars(ax, x, hs, w * 0.92, *bar_style(m), labels=False)
        if key == "cov80":
            ax.axhline(80, color=figstyle.INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, 0.4)
        _group_axes(ax)
        ax.set_ylabel(yl)
        ax.set_title(title)
    handles = [figstyle.patch(*bar_style(m), LEGEND[m]) for m in models] + [_dashed("Nominal coverage (80%)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.98), fontsize=6.8,
               handlelength=1.3, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIG, "accuracy.pdf"))
    plt.close(fig)


def fig_dependence(rows, pw):
    """Fig. 4: rank dependence of the paths (a), shuffle gain on the same forecasts (b), and the shuffle test on
    synthetic forecasts with known dependence (c) (full text width)."""
    from matplotlib.lines import Line2D
    pathm = [m for m in ["naive", "chronos", "chronos_base", "toto", "sundial", "aurora"] if sel(rows, "ercot", 24, m)]
    fig = plt.figure(figsize=(TEXT_W, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.55, wspace=0.25,
                          top=0.84, bottom=0.11, left=0.075, right=0.995)
    w = 0.84 / len(pathm)
    ax = fig.add_subplot(gs[0, 0])
    for i, m in enumerate(pathm):
        x = np.arange(len(DS)) + (i - (len(pathm) - 1) / 2) * w
        hm = [mean(rows, ds, 24, m, "rho_model") for ds in DS]
        ho = [mean(rows, ds, 24, m, "rho_obs") for ds in DS]
        figstyle.bars(ax, x, hm, w * 0.92, *bar_style(m), labels=False)
        ax.scatter(x, ho, marker="_", s=70, color=figstyle.INK, linewidths=1.5, zorder=5)
    ax.set_ylim(0, 1.05)
    _group_axes(ax)
    ax.set_ylabel("Rank correlation")
    ax.set_title("a  Dependence of adjacent steps")
    ax = fig.add_subplot(gs[0, 1])
    top = 25
    gains = gain_results(rows, pw, 3, 2000, model_first=True)       # cluster bootstrap over calendar weeks
    for i, m in enumerate(pathm):
        x = np.arange(len(DS)) + (i - (len(pathm) - 1) / 2) * w
        g, lo, hi = [], [], []
        for ds in DS:
            g_, lo_, hi_ = gains[(ds, m)]
            g.append(100 * g_)
            lo.append(g[-1] - 100 * lo_)
            hi.append(100 * hi_ - g[-1])
        fill, hatch = bar_style(m)
        ax.bar(x, np.minimum(g, top), w * 0.92, color=fill, edgecolor=figstyle.INK, linewidth=figstyle.FRAME,
               hatch=hatch, zorder=2)
        ax.errorbar(x, g, yerr=[lo, hi], fmt="none", ecolor=figstyle.INK, elinewidth=figstyle.FRAME, capsize=1.5,
                    zorder=3)
        for xi, v in zip(x, g):
            if v > top:                                   # bar cut at the top of the axis: print its value
                ax.text(xi, top - 0.8, f"{v:.0f}", ha="center", va="top", fontsize=6, color="white", zorder=4)
    ax.axhline(0, color=figstyle.INK, lw=0.6)
    ax.set_ylim(-5, top)
    _group_axes(ax)
    ax.set_ylabel("Shuffle gain (%)")
    ax.set_title("b  Shuffle gain on the same forecasts")
    sub = gs[1, :].subgridspec(1, 2, width_ratios=[0.25, 0.75], wspace=0)
    ax = fig.add_subplot(sub[0, 1])
    ext = os.path.join(RES, "shuffle_check_ext.json")         # shuffle_check.py; the seven original cases
    sc = ([c for c in json.load(open(ext))["cases"] if c["original"]] if os.path.exists(ext)
          else json.load(open(os.path.join(RES, "shuffle_check.json"))))
    y = np.arange(len(sc))[::-1]
    gains = [c["gain"] for c in sc]
    ax.barh(y, gains, 0.6, color=figstyle.lighter(figstyle.GREY, 0.3), edgecolor=figstyle.INK,
            linewidth=figstyle.FRAME, zorder=2)
    ax.errorbar(gains, y, xerr=[[c["gain"] - c["lo"] for c in sc], [c["hi"] - c["gain"] for c in sc]], fmt="none",
                ecolor=figstyle.INK, elinewidth=figstyle.FRAME, capsize=1.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([c["case"] for c in sc], fontsize=6.8)
    ax.axvline(0, color=figstyle.INK, lw=0.6)
    ax.grid(axis="x", color=figstyle.GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Shuffle gain (%)")
    ax.set_title("c  Shuffle test on synthetic forecasts with known dependence")
    handles = [figstyle.patch(*bar_style(m), LEGEND[m]) for m in pathm] + [
        Line2D([0], [0], color=figstyle.INK, lw=0, marker="_", ms=9, mew=1.5, label="Observed dependence (PIT)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.8,
               handlelength=1.3, handletextpad=0.4, columnspacing=1.2)
    fig.savefig(os.path.join(FIG, "dependence.pdf"))
    plt.close(fig)


def fig_horizon2(rows):
    """Fig. 5: quantile loss by forecast day at H = 720 (top) and coverage of 80% intervals against the
    horizon (bottom) (full text width)."""
    models = [m for m in ["naive", "chronos", "toto", "timesfm", "bolt", "chronos2", "sundial", "aurora"]]
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W, 3.9))
    for c, ds in enumerate(DS):
        ax = axes[0, c]
        for m in models:
            rr = sel(rows, ds, 720, m)
            if not rr:
                continue
            v = np.array(rr[0]["lead_ql"])
            day = 24 if ds != "ausel" else 48
            v = v[: (len(v) // day) * day].reshape(-1, day).mean(1)
            ax.plot(np.arange(1, len(v) + 1), v, color=bar_style(m)[0], lw=1.1, label=SHORT[m],
                    ls="--" if m == "naive" else "-")
        ax.set_title(f"{'abc'[c]}  {DS[ds]}")
        ax.set_xlabel("Forecast day")
        if c == 0:
            ax.set_ylabel("Quantile loss")
        ax = axes[1, c]
        for m in models:
            hs = [(h, 100 * mean(rows, ds, h, m, "cov80")) for h in (24, 192, 720) if sel(rows, ds, h, m)]
            if len(hs) < 2:
                continue
            ax.plot([np.log(h) for h, _ in hs], [v for _, v in hs], marker="o", ms=3, color=bar_style(m)[0], lw=1.1,
                    ls="--" if m == "naive" else "-", label=SHORT[m])
        ax.axhline(80, color=figstyle.INK, lw=0.7, ls=(0, (3, 2)))
        ax.set_xticks([np.log(24), np.log(192), np.log(720)])
        ax.set_xticklabels(["24", "192", "720"])
        ax.set_ylim(20, 95)
        ax.set_xlabel("Horizon (steps)")
        ax.set_title(f"{'def'[c]}  {DS[ds]}")
        if c == 0:
            ax.set_ylabel("Coverage of 80% interval (%)")
    h_, l_ = axes[1, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=8, bbox_to_anchor=(0.5, 0.985), fontsize=6.8, columnspacing=1.0,
               handlelength=1.8, handletextpad=0.4)
    fig.tight_layout(h_pad=0.9, w_pad=1.2)
    fig.savefig(os.path.join(FIG, "horizon.pdf"))
    plt.close(fig)


def fig_leakage(out):
    """Earlier leakage figure, not in the paper (its Fig. 6 comes from revision_report.py): quantile loss relative to
    Chronos-Bolt on the 2002-2015 and the 2026 Australian data (one column)."""
    names = {"timesfm": "TimesFM\n2.5", "aurora": "Aurora", "toto": "Toto", "sundial": "Sundial", "chronos2": "Chronos-2",
             "bolt_base": "Chronos-\nBolt base", "chronos": "Chronos\nsmall", "naive": "Naive"}
    order = [m for m in ["timesfm", "aurora", "toto", "sundial", "chronos2", "bolt_base", "chronos", "naive"] if m in out]
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    for i, m in enumerate(order):
        old, new, _ = out[m]
        for j, (val, col, mk) in enumerate([(old, figstyle.GREY, "o"), (new, figstyle.INK, "s")]):
            ax.errorbar([i + (j - 0.5) * 0.3], [val[0]], yerr=[[val[0] - val[1]], [val[2] - val[0]]], fmt=mk, ms=3.6,
                        color=col, ecolor=col, elinewidth=0.8, capsize=1.5,
                        label=(["2002–2015 series (Monash)", "2026 data (AEMO)"][j] if i == 0 else None))
    ax.axhline(1, color=figstyle.INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[m] for m in order], fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylabel("QL relative to Chronos-Bolt small")
    ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.3, borderaxespad=0.3)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIG, "leakage.pdf"))
    plt.close(fig)


def main():
    rows, pw = load()
    small = json.load(open(os.path.join(RES, "robust.json")))
    table_main(rows, small)
    out = table_fresh(rows, pw)
    fig_accuracy(rows)
    fig_dependence(rows, pw)
    fig_horizon2(rows)
    fig_leakage(out)
    table_all(rows)
    table_long(rows)
    table_ci(rows, pw)
    table_dep(rows, pw)
    table_sundial(rows)
    summary(rows)


if __name__ == "__main__":
    main()
