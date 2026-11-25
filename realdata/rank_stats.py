"""Mean ranks and significance tests of the 24-step benchmark (Section VI-C of the article).

Ten forecasters (the seasonal naive baseline and the nine foundation models, all run on the four datasets; Chronos
base on AEMO 2026 in the second revision, read through bench_report.EXTRA_RUNS) are ranked on ERCOT, Traffic,
Australian electricity 2002-2015 and AEMO 2026 by three scores: quantile loss, CRPS and the absolute error of the
80% coverage. The number of forecasters is taken from MODELS everywhere (ranks 1 to k, k - 1 comparisons with
Chronos-2 for Holm's correction, Nemenyi critical value).

  - Friedman test and Iman-Davenport correction with one value per dataset, and the Nemenyi critical distance
    (Demsar, 2006).
  - Mean ranks over calendar weeks. The series of a dataset share weather and calendar, so the unit below the
    dataset is a calendar week of the forecast origin, over all series (cluster_stats.py): in each dataset the
    forecasters are ranked within every week by the mean score of its windows, the ranks are averaged over the
    weeks and then over the four datasets. A cluster bootstrap that resamples weeks within each dataset gives 95%
    intervals for the difference between each forecaster's mean rank and that of Chronos-2.
  - Wilcoxon signed-rank tests of Chronos-2 against every other model on week means (the weeks of the four
    datasets pooled), with Holm's correction.
  - Ranks without the model-dataset pairs with known, probable or test-indicated leakage, and ranks on AEMO 2026
    alone, the only dataset that post-dates every model.

Writes ../outputs/tables/rank_table.tex (Table V), ../outputs/tables/stats_table.tex, ../outputs/tables/wilcoxon_table.tex
and ../outputs/tables/rank_clean_table.tex (the last is read by wilcoxon_table.tex).
"""

import os

import numpy as np
from scipy import stats

from bench_report import B, LEAK, LEAK_TEST, SEC, SHORT, SUPP, load, mean, win
from cluster_stats import weeks

DATA = {"ercot": "ERCOT", "traffic": "Traffic", "ausel": "Austr. 2002--15", "aemo": "AEMO 2026"}
DATA_LONG = {"ercot": "ERCOT", "traffic": "Traffic", "ausel": "Australian electricity 2002--2015",
             "aemo": "AEMO 2026"}
MODELS = ["naive", "chronos", "chronos_base", "toto", "timesfm", "bolt", "bolt_base", "chronos2", "sundial", "aurora"]
METRICS = {"ql": "QL", "crps": "CRPS", "cal": "Cal.\\ error"}
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}  # Nemenyi, alpha 0.05
H = 24
REF = "chronos2"
K = len(MODELS)
WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve"}


def value(rows, ds, m, k):
    if k == "cal":
        return abs(100 * mean(rows, ds, H, m, "cov80") - 80)
    return mean(rows, ds, H, m, k)


def leaked(m, ds):
    """Pairs with known (a), probable (b) or test-indicated (e) leakage, as marked in Table IV."""
    return LEAK.get((m, ds), "") in ("a", "b") or (m, ds) in LEAK_TEST


def week_values(pw, k):
    """Per dataset, week means (weeks, models) of a per-window score: the mean over all windows whose forecast
    origin falls in the calendar week, over all series; for 'cal' the absolute error of the week's 80% coverage."""
    out = {}
    for ds in DATA:
        u, inv = np.unique(weeks(pw, ds, H), return_inverse=True)
        n = np.bincount(inv, None, len(u))
        cols = []
        for m in MODELS:
            v = np.bincount(inv, win(pw, ds, H, m, "cov80" if k == "cal" else k), len(u)) / n
            cols.append(np.abs(100 * v - 80) if k == "cal" else v)
        out[ds] = np.stack(cols, 1)
    return out


def friedman(X):
    """X: (N units, k methods), lower is better. Returns mean ranks, chi2_F, p, F_ID, p_ID, CD."""
    N, k = X.shape
    R = np.apply_along_axis(stats.rankdata, 1, X).mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    p = stats.chi2.sf(chi2, k - 1)
    F = (N - 1) * chi2 / (N * (k - 1) - chi2)
    pF = stats.f.sf(F, k - 1, (k - 1) * (N - 1))
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    return R, chi2, p, F, pF, cd


def week_rank_boot(V, nboot=B, seed=4):
    """Mean ranks over weeks (averaged within each dataset, then over datasets) and cluster-bootstrap draws of the
    differences to Chronos-2, resampling weeks within each dataset."""
    rng = np.random.default_rng(seed)
    R = {ds: stats.rankdata(V[ds], axis=1) for ds in V}               # ranks within each week
    point = np.mean([R[ds].mean(0) for ds in R], 0)
    draws = np.zeros((nboot, len(MODELS)))
    for ds in R:
        n = len(R[ds])
        draws += R[ds][rng.integers(0, n, (nboot, n))].mean(1)
    draws /= len(R)
    ref = MODELS.index(REF)
    diff = draws - draws[:, [ref]]
    lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
    pboot = np.minimum(1, 2 * np.minimum((diff <= 0).mean(0), (diff >= 0).mean(0)))
    return point, point - point[ref], lo, hi, pboot


def holm(ps):
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run_max = 0
    for r_, i in enumerate(o):                                           # Holm step-down
        run_max = max(run_max, min(1, (len(ps) - r_) * ps[i]))
        adj[i] = run_max
    return adj


def pfmt(p):
    """p value with two significant digits; scientific notation below 0.001."""
    m, e = f"{p:.1e}".split("e")
    e = int(e)
    if e < -3:
        return rf"${m}\times10^{{{e}}}$"
    return f"{float(f'{p:.1e}'):.{max(1, 1 - e)}f}"


def rank_cols(X):
    """Ranks of the columns of X (datasets, models) within each row; NaN entries (left-out pairs) are not ranked,
    and the ranks of a row with k ranked models are rescaled linearly from 1..k to 1..K (K = len(MODELS))."""
    out = np.full(X.shape, np.nan)
    for i, row in enumerate(X):
        ok = ~np.isnan(row)
        k = ok.sum()
        r = stats.rankdata(row[ok])
        out[i, ok] = 1 + (r - 1) * (len(MODELS) - 1) / (k - 1)
    return out


def table_ranks(rows):
    ranks = {}
    for met in METRICS:
        X = np.array([[value(rows, ds, m, met) for m in MODELS] for ds in DATA])
        ranks[met] = np.apply_along_axis(stats.rankdata, 1, X)        # (datasets, models)
    mr = {met: ranks[met].mean(0) for met in METRICS}
    overall = np.mean([mr[met] for met in METRICS], 0)
    order = np.argsort(overall)
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Average Rank Across the Four Datasets (Lower Is Better)}", r"\label{tab:ranks}",
         r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
         r"Model & QL & CRPS & Cal.\ error & Mean \\", r"\midrule"]
    for j in order:
        L.append(f"{SHORT[MODELS[j]]} & " + " & ".join(f"{mr[met][j]:.2f}" for met in METRICS) +
                 f" & {overall[j]:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\vspace{2pt}", rf"\parbox{{\columnwidth}}{{\footnotesize Ranks 1 to {K} per dataset and score, averaged over "
          r"ERCOT and Traffic as well as Australian electricity 2002--2015 and AEMO 2026 (24-step forecasts). Cal.\ error: "
          r"absolute difference between the coverage of the central 80\% interval and the nominal 80\% (in "
          r"percentage points). The ranks include "
          r"model--dataset pairs with leakage. According to Table~\ref{tab:models}, Toto had ERCOT in its pretraining "
          r"corpus and probably the Traffic and 2002--2015 Australian series, and TimesFM~2.5 probably the Traffic and "
          r"2002--2015 Australian series. The test of Section~\ref{sec:leakage} indicates that Aurora saw the 2002--2015 "
          r"Australian series. Table~\ref{S-stab:rankclean} gives the ranks without these pairs and on AEMO 2026 "
          r"alone.}", r"\end{table}"]
    open(os.path.join(SEC, "rank_table.tex"), "w").write("\n".join(L) + "\n")
    print("mean ranks:", {SHORT[MODELS[j]]: (round(float(mr['ql'][j]), 2), round(float(mr['crps'][j]), 2),
                                             round(float(mr['cal'][j]), 2), round(float(overall[j]), 2)) for j in order})
    per_ds_rank = {ds: {SHORT[m]: int(ranks['ql'][i, j]) for j, m in enumerate(MODELS)} for i, ds in enumerate(DATA)}
    print("QL ranks per dataset:", per_ds_rank)


def table_clean(rows):
    """Ranks without leaked pairs (rescaled to 1..K) and ranks on AEMO 2026 alone."""
    clean, aemo = {}, {}
    for met in METRICS:
        X = np.array([[np.nan if leaked(m, ds) else value(rows, ds, m, met) for m in MODELS] for ds in DATA])
        clean[met] = rank_cols(X)
        aemo[met] = stats.rankdata([value(rows, "aemo", m, met) for m in MODELS])
    n_ds = (~np.isnan(clean["ql"])).sum(0)
    cm = {met: np.nanmean(clean[met], 0) for met in METRICS}
    c_all = np.mean([cm[met] for met in METRICS], 0)
    a_all = np.mean([aemo[met] for met in METRICS], 0)
    kept = {DATA[ds]: int((~np.isnan(clean['ql'][i])).sum()) for i, ds in enumerate(DATA)}
    left_out = "; ".join(f"{SHORT[m]} on {DATA_LONG[ds]}" for m in MODELS for ds in DATA if leaked(m, ds))
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{\textbf{Average ranks without leaked model--dataset pairs and on AEMO 2026 alone} (24-step "
         r"forecasts; lower is better). Without leaked pairs: the pairs marked $a$, $b$, or $e$ in "
         r"Table~\ref{tab:real} are left out. On each dataset the $k$ remaining models are ranked from 1 to $k$, and "
         rf"the ranks are rescaled to 1 to {K} by $1 + {K - 1}(r-1)/(k-1)$ so that every dataset carries the same "
         r"weight; $n$: number of datasets on which a model is ranked. AEMO 2026: ranks on the AEMO data of 2026, "
         r"which post-date every model. Cal.\ error and Mean as in Table~\ref{tab:ranks}; rows are ordered by the "
         r"mean rank without leaked pairs.}",
         r"\label{stab:rankclean}", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{@{}l r r r r r r r r r@{}}", r"\toprule",
         r" & \multicolumn{5}{c}{Without leaked pairs} & \multicolumn{4}{c}{AEMO 2026 alone} \\",
         r"\cmidrule(lr){2-6}\cmidrule(l){7-10}",
         r"Model & QL & CRPS & Cal.\ error & Mean & $n$ & QL & CRPS & Cal.\ error & Mean \\", r"\midrule"]
    for j in np.argsort(c_all, kind="stable"):
        L.append(f"{SHORT[MODELS[j]]} & " + " & ".join(f"{cm[met][j]:.2f}" for met in METRICS) +
                 f" & {c_all[j]:.2f} & {n_ds[j]} & " + " & ".join(f"{aemo[met][j]:.0f}" if aemo[met][j] % 1 == 0
                                                                    else f"{aemo[met][j]:.1f}" for met in METRICS) +
                 f" & {a_all[j]:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "rank_clean_table.tex"), "w").write("\n".join(L) + "\n")
    print("left out:", left_out)
    for name, mrk, overall in [("without leaked pairs", cm, c_all), ("AEMO 2026 only", aemo, a_all)]:
        o = np.argsort(overall, kind="stable")
        print(f"ranks {name}: " + ", ".join(
            f"{SHORT[MODELS[j]]} QL {mrk['ql'][j]:.2f} CRPS {mrk['crps'][j]:.2f} cal {mrk['cal'][j]:.2f} "
            f"mean {overall[j]:.2f}" for j in o))
        for met in METRICS:
            print(f"  first on {met}: {SHORT[MODELS[int(np.argmin(mrk[met]))]]}"
                  + (" (tie)" if np.sum(mrk[met] == mrk[met].min()) > 1 else ""))
        print("  first on mean: " + ", ".join(SHORT[MODELS[j]] for j in o if np.isclose(overall[j], overall[o[0]])))


def main():
    rows, pw = load()
    k = len(MODELS)
    ref = MODELS.index(REF)
    table_ranks(rows)
    table_clean(rows)

    # ---------------------------------------------------------------- Friedman test with the dataset as the unit
    S = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize"]
    fr = []
    for met, name in METRICS.items():
        X = np.array([[value(rows, ds, m, met) for m in MODELS] for ds in DATA])
        R, chi2, p, F, pF, cd = friedman(X)
        o = np.argsort(R)
        fr.append(f"{name} & {X.shape[0]} & {chi2:.1f} & {pfmt(p)} & {F:.2f} & {pfmt(pF)} & {cd:.2f} & "
                  f"{SHORT[MODELS[o[0]]]} & {R[o[1]] - R[o[0]]:.2f} \\\\")
        print(f"friedman dataset {met:4s} N={X.shape[0]:4d} chi2={chi2:7.1f} p={p:.2e} F={F:.2f} pF={pF:.2e} "
              f"CD={cd:.2f} ranks " + ", ".join(f"{SHORT[MODELS[j]]} {R[j]:.2f}" for j in o))

    # ---------------------------------------------------------------- mean ranks over calendar weeks
    V = {met: week_values(pw, met) for met in METRICS}
    nw = {ds: len(V["ql"][ds]) for ds in DATA}
    nw_txt = ", ".join(str(nw[ds]) for ds in list(DATA)[:-1]) + f", and {nw[list(DATA)[-1]]}"
    wr = {met: week_rank_boot(V[met], seed=4 + i) for i, met in enumerate(METRICS)}
    # Holm's correction of the two-sided bootstrap p values (k - 1 comparisons per score): name the differences
    # whose interval excludes zero but whose adjusted p value is 0.05 or more
    others = [j for j in range(k) if j != ref]
    lost = []
    for met in METRICS:
        point, d, lo, hi, pb = wr[met]
        adj = holm(np.maximum(pb[others], 1 / B))
        for i, j in enumerate(others):
            print(f"  bootstrap {met:4s} {SHORT[MODELS[j]]:18s} p_boot {pb[j]:.4f} p_holm {adj[i]:.4f}")
            if (lo[j] > 0 or hi[j] < 0) and adj[i] >= 0.05:
                what = {"ql": "QL", "crps": "CRPS", "cal": "calibration error"}[met]
                lost.append(f"{SHORT[MODELS[j]]} on the {what} (adjusted $p = {adj[i]:.2f}$)")
    holm_txt = (rf" After Holm's correction for the {WORD[k - 1]} comparisons of each score, every difference whose "
                r"interval excludes zero remained significant at the 5\% level" +
                (", except " + " and ".join(lost) if lost else "") + ".")
    S += [rf"\caption{{\textbf{{Rank-based comparison of the {WORD[k]} forecasters of Table~\ref{{tab:ranks}}.}} (a) Friedman "
          r"test with the Iman--Davenport correction and the Nemenyi critical distance (CD) at $\alpha = 0.05$, with "
          r"the dataset as the unit. Best: forecaster with the lowest average rank; Gap: its lead over the runner-up. "
          r"(b) Ranks within each calendar week by the mean score of the week's windows, averaged over the weeks of "
          r"each dataset and then over the four datasets. $\Delta$: mean rank minus that of Chronos-2, with a 95\% "
          rf"cluster-bootstrap interval over weeks within each dataset ({B:,} resamples)." + holm_txt + "}",
          r"\label{stab:friedman}",
          r"\textit{(a) Dataset as the unit}\\[3pt]",
          r"\begin{tabular}{@{}l r r r r r r l r@{}}", r"\toprule",
          r"Score & $N$ & $\chi^2_F$ & $p$ & $F_{\mathrm{ID}}$ & $p_{\mathrm{ID}}$ & CD & Best & Gap \\", r"\midrule"] + fr + \
         [r"\bottomrule", r"\end{tabular}", r"", r"\vspace{8pt}",
          r"\textit{(b) Calendar week as the unit, cluster bootstrap}\\[3pt]",
          r"\setlength{\tabcolsep}{4pt}",
          r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
          r" & \multicolumn{2}{c}{QL} & \multicolumn{2}{c}{CRPS} & \multicolumn{2}{c}{Cal.\ error} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
          r"Model & Rank & $\Delta$ (95\% CI) & Rank & $\Delta$ (95\% CI) & Rank & $\Delta$ (95\% CI) \\", r"\midrule"]
    for j in np.argsort(wr["ql"][0], kind="stable"):
        cells = []
        for met in METRICS:
            point, d, lo, hi, pb = wr[met]
            cells += [f"{point[j]:.2f}", "--" if j == ref else f"${d[j]:+.2f}$ (${lo[j]:+.2f}$, ${hi[j]:+.2f}$)"]
        S.append(f"{SHORT[MODELS[j]]} & " + " & ".join(cells) + r" \\")
    S += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "stats_table.tex"), "w").write("\n".join(S) + "\n")
    print("weeks per dataset:", nw, "total", sum(nw.values()))
    for met in METRICS:
        point, d, lo, hi, pb = wr[met]
        print(f"week ranks {met:4s}: " + ", ".join(
            f"{SHORT[MODELS[j]]} {point[j]:.2f}" + ("" if j == ref else f" (d {d[j]:+.2f} [{lo[j]:+.2f},{hi[j]:+.2f}] "
                                                               f"p_boot {pb[j]:.4f})")
            for j in np.argsort(point, kind="stable")))

    # ---------------------------------------------------------------- Wilcoxon on week means, Chronos-2 vs each model
    N_all = sum(nw.values())
    W = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize",
         r"\caption{\textbf{Wilcoxon signed-rank tests of Chronos-2 against each other forecaster} on week means of "
         r"the quantile loss and the CRPS, 24-step forecasts. A week mean averages the windows of all series whose "
         r"forecast origin falls in one calendar week. Better: share of weeks in which Chronos-2 had the lower score. "
         rf"(a) The weeks of the four datasets pooled ($N = {N_all}$), with $p_{{\mathrm{{Holm}}}}$ after Holm's "
         rf"correction for the {WORD[k - 1]} comparisons of each score. (b) Chronos-2 against TimesFM 2.5 on each "
         r"dataset and on the pooled weeks of ERCOT and AEMO 2026, the two datasets without probable leakage for TimesFM 2.5, with unadjusted $p$ "
         r"values.}",
         r"\label{stab:wilcoxon}",
         r"\textit{(a) Chronos-2 against each forecaster, four datasets pooled}\\[3pt]",
         r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Quantile loss} & \multicolumn{2}{c}{CRPS} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
         r"Chronos-2 against & Better (\%) & $p_{\mathrm{Holm}}$ & Better (\%) & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    res = {}
    for met in ["ql", "crps"]:
        X = np.concatenate([V[met][ds] for ds in DATA])                 # (weeks of all datasets, models)
        ps = np.array([stats.wilcoxon(X[:, ref], X[:, j]).pvalue for j in others])
        better = [float(np.mean(X[:, ref] < X[:, j])) for j in others]
        adj = holm(ps)
        for i, j in enumerate(others):
            res[(met, MODELS[j])] = (better[i], adj[i])
        print(f"wilcoxon {met}: n_weeks={X.shape[0]} " +
              ", ".join(f"{SHORT[MODELS[j]]} better {100 * better[i]:.0f}% p_raw={ps[i]:.1e} p_holm={adj[i]:.1e}"
                        for i, j in enumerate(others)))
    for j in others:
        m = MODELS[j]
        W.append(f"{SHORT[m]} & {100 * res[('ql', m)][0]:.0f} & {pfmt(res[('ql', m)][1])} & "
                 f"{100 * res[('crps', m)][0]:.0f} & {pfmt(res[('crps', m)][1])} \\\\")
    # (b) Chronos-2 against TimesFM 2.5 per dataset, and on the datasets without probable leakage for TimesFM 2.5
    tf = MODELS.index("timesfm")
    groups = [(DATA_LONG[ds], [ds]) for ds in DATA] + \
             [("ERCOT and AEMO 2026 pooled", [ds for ds in DATA if not leaked("timesfm", ds)])]
    W += [r"\bottomrule", r"\end{tabular}", r"", r"\vspace{8pt}",
          r"\textit{(b) Chronos-2 against TimesFM 2.5 by dataset}\\[3pt]",
          r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
          r" & & \multicolumn{2}{c}{Quantile loss} & \multicolumn{2}{c}{CRPS} \\",
          r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
          r"Data & Weeks & Better (\%) & $p$ & Better (\%) & $p$ \\", r"\midrule"]
    for gi, (name, dss) in enumerate(groups):
        cells = []
        for met in ["ql", "crps"]:
            X = np.concatenate([V[met][ds] for ds in dss])
            p = stats.wilcoxon(X[:, ref], X[:, tf]).pvalue
            better = float(np.mean(X[:, ref] < X[:, tf]))
            cells += [f"{100 * better:.0f}", pfmt(p)]
            print(f"wilcoxon vs TimesFM 2.5 {met:4s} {name:35s} weeks={len(X):3d} better {100 * better:.1f}% p={p:.2e}")
        if gi == len(groups) - 1:
            W.append(r"\midrule")
        W.append(f"{name} & {len(X)} & " + " & ".join(cells) + r" \\")
    W += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"", r"\input{supplementary/rank_clean_table}"]
    open(os.path.join(SUPP, "wilcoxon_table.tex"), "w").write("\n".join(W) + "\n")


if __name__ == "__main__":
    main()
