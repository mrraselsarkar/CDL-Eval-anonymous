"""Report of the real-data robustness checks.

Reads results/robust.json and results/robust_perwindow.npz (robust.py), results/real_fm.json and
results/real_fm_perwindow.npz (run_fm.py) and results/aurora_context.json (aurora_context.py), and writes
  ../outputs/tables/real_robust_table.tex   Table S14: robustness of the real-data study
  ../outputs/tables/real_aurora_table.tex   Table S8: Aurora with 96 to 2,880 context steps
  ../outputs/tables/real_ci_table.tex       cluster-bootstrap intervals (calendar weeks; not part of the Supplement)
and prints the numbers quoted in the text.
"""

import json
import os

import numpy as np

from cluster_stats import ratio_draws, weeks

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SUPP = os.path.join(HERE, "..", "outputs", "tables")
DS = {"ercot": "ERCOT load", "traffic": "Traffic occupancy", "ausel": "Australian electricity"}
PRE = [("chronos_512_default", "Chronos small, 512 steps, top-$k$ 50 (default)"),
       ("chronos_512_full", "Chronos small, 512 steps, no top-$k$"),
       ("chronos_48_default", "Chronos small, 48 steps"),
       ("bolt_512", "Chronos-Bolt small, 512 steps"),
       ("bolt_48", "Chronos-Bolt small, 48 steps"),
       ("sundial_512", "Sundial, 512 steps"),
       ("sundial_2880", "Sundial, 2,880 steps"),
       ("sundial_48", "Sundial, 48 steps"),
       # foundation models of run_fm.py (512 context steps)
       ("toto", "Toto, 512 steps"),
       ("timesfm_ind", "TimesFM 2.5, 512 steps"),
       ("chronos2_ind", "Chronos-2, 512 steps, levels 0.1--0.9"),
       ("chronos2_native_ind", "Chronos-2, 512 steps, 21 native levels"),
       ("aurora", "Aurora, 512 steps, token length 48"),
       ("aurora_tok24", "Aurora, 512 steps, token length 24")]
SMALL = [("ar", "AR, Gaussian head"), ("direct", "Direct multi-step"), ("ardiff", "AR + diffusion head"),
         ("fulldiff", "Full-window diffusion"), ("flow", "Full-window flow matching"),
         ("masked", "Masked diffusion"), ("masked_rand", "Masked diffusion, random order")]
B = 5000
CI_NAMES = {"chronos_512_default": "Chronos small (top-$k$ 50, default)", "chronos_512_full": "Chronos small (no top-$k$)",
            "bolt_512": "Chronos-Bolt small", "sundial_512": "Sundial", "flow": "Flow matching",
            "toto": "Toto", "timesfm_ind": "TimesFM 2.5", "chronos2_ind": "Chronos-2", "aurora": "Aurora"}


def rows_of(rows, ds, model):
    if model in dict(PRE):
        return [r for r in rows if r["dataset"] == ds and r["model"] == model]
    return [r for r in rows if r["dataset"] == ds and r["model"].rsplit("_s", 1)[0] == model]


def mean(rows, key):
    v = [r[key] for r in rows if key in r]
    return np.mean(v) if v else float("nan")


def table(rows):
    head = r"Forecaster & CRPS & CRPS (fair) & ES & ES (fair) & QL & Cov.\ 80 & Cov.\ 90 \\"
    L = [r"{\footnotesize", r"\renewcommand{\arraystretch}{1.08}", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{longtable}{@{}l r r r r r r l@{}}",
         r"\caption{\textbf{Robustness of the real-data study.} The test windows of Table~\ref{tab:real} with other "
         r"settings and one sampling seed. CRPS and ES: V-statistic and unbiased (fair) estimators. Cov.\ 80 and Cov.\ 90: coverage (\%) of "
         r"central intervals from the PIT. For Chronos-Bolt small, Cov.\ 80 is read from its 0.1 and 0.9 quantiles, "
         r"and Cov.\ 90 is given for three tail models (linear extrapolation / Gaussian / clamped).}",
         r"\label{stab:realrobust}\\", r"\toprule", head, r"\endfirsthead",
         r"\multicolumn{8}{@{}l}{\textit{Table~\ref{stab:realrobust} continued}}\\",
         r"\toprule", head, r"\endhead", r"\bottomrule", r"\endfoot"]
    first = True
    for ds, dname in DS.items():
        L += ([] if first else [r"\midrule"]) + [rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{dname}}}}} \\"]
        first = False
        for key, name in PRE:                    # the small forecasters are in Table 2 of the main text
            rr = rows_of(rows, ds, key)
            if not rr:
                continue
            if key.startswith("bolt"):
                c80 = 100 * mean(rr, "cov80_native")
                c90 = "/".join(f"{100 * mean(rr, 'cov90_' + t):.0f}" for t in ["linear", "gaussian", "clamped"])
            else:
                c80 = 100 * mean(rr, "pit80")
                c90 = f"{100 * mean(rr, 'pit90'):.1f}"
            L.append(f"{name} & {mean(rr, 'crps'):.3f} & {mean(rr, 'crps_fair'):.3f} & {mean(rr, 'es'):.2f} & "
                     f"{mean(rr, 'es_fair'):.2f} & {mean(rr, 'qloss'):.3f} & {c80:.1f} & {c90} \\\\")
    L += [r"\end{longtable}", r"}"]
    with open(os.path.join(SUPP, "real_robust_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


class PerWindow(dict):
    """A dict of per-window arrays with the .files attribute of an npz archive."""
    @property
    def files(self):
        return list(self.keys())


def window_scores(pw, ds, model, metric):
    keys = [k for k in pw.files if k.startswith(f"{ds}|") and k.endswith(f"|{metric}")
            and (k.split("|")[1] == model or k.split("|")[1].rsplit("_s", 1)[0] == model)]
    return np.mean([pw[k] for k in keys], 0)


COMPARE = [("chronos2_ind", "bolt_512", "crps"), ("chronos2_ind", "toto", "crps"),
           ("toto", "chronos_512_default", "crps"), ("toto", "sundial_512", "crps"), ("toto", "sundial_512", "es"),
           ("timesfm_ind", "bolt_512", "crps"), ("aurora", "sundial_512", "crps"),
           ("bolt_512", "sundial_512", "crps"), ("bolt_512", "chronos_512_default", "crps"),
           ("chronos_512_full", "chronos_512_default", "crps"), ("flow", "fulldiff", "crps"),
           ("flow", "ar", "es"), ("bolt_512", "flow", "crps")]


def block_ci(pw):
    """Intervals of the first-round comparisons, from a cluster bootstrap over calendar weeks of the forecast
    origin across all series of a dataset (cluster_stats.py). The table is not part of the Supplementary Material."""
    rng = np.random.default_rng(0)
    out = []
    for ds in DS:
        wk = weeks(pw, ds, 24)                           # the first-round windows are the 24-step bench windows
        for a, b, metric in COMPARE:
            xa, xb = window_scores(pw, ds, a, metric), window_scores(pw, ds, b, metric)
            rel = ratio_draws(wk, xa, xb, B, rng) - 1
            lo, hi = np.percentile(rel, [2.5, 97.5])
            out.append(dict(ds=ds, a=a, b=b, metric=metric, rel=xa.mean() / xb.mean() - 1, lo=lo, hi=hi,
                            nblocks=len(np.unique(wk))))
            print(f"{ds:8s} {metric} {a:20s} vs {b:20s} rel={100*out[-1]['rel']:+.1f}% "
                  f"CI [{100*lo:+.1f}, {100*hi:+.1f}] weeks={out[-1]['nblocks']}")
    name = dict(PRE + SMALL)
    name.update(CI_NAMES)
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{\textbf{Uncertainty of real-data comparisons.} Relative difference in the mean score of "
         r"forecaster A and forecaster B (negative values favor A), with 95\% intervals from a cluster bootstrap "
         r"that resamples calendar weeks of the forecast origin across all series of a dataset (" + f"{B:,}" +
         r" resamples). Pretrained models use 512 context steps; small forecasters are averaged over three seeds.}",
         r"\label{stab:realci}",
         r"\begin{tabular}{@{}l l l l r r@{}}", r"\toprule",
         r"Data & Score & Forecaster A & Forecaster B & A vs B (95\% CI) & Weeks \\", r"\midrule"]
    prev = None
    for r in out:
        if prev is not None and r["ds"] != prev:
            L.append(r"\addlinespace")
        lab = DS[r["ds"]] if r["ds"] != prev else ""
        prev = r["ds"]
        L.append(f"{lab} & {r['metric'].upper()} & {name[r['a']]} & {name[r['b']]} & "
                 f"${100*r['rel']:+.1f}\\%$ (${100*r['lo']:+.1f}$, ${100*r['hi']:+.1f}$) & {r['nblocks']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(SUPP, "real_ci_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def aurora_table():
    """Supplementary table: Aurora with context lengths from 96 to 2,880 steps (aurora_context.py)."""
    path = os.path.join(RES, "aurora_context.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        rows = json.load(f)
    ctxs = sorted({r["context"] for r in rows})
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{\textbf{Aurora with different context lengths.} CRPS and coverage (\%) of nominal 90\% "
         r"intervals from the PIT on the test windows of Table~\ref{tab:real}, with "
         r"Aurora's native token length of 48. The main results use 512 context steps.}",
         r"\label{stab:auroractx}",
         r"\begin{tabular}{@{}l " + "r " * len(ctxs) + r"@{}}", r"\toprule",
         r"Context steps & " + " & ".join(f"{c:,}" for c in ctxs) + r" \\", r"\midrule"]
    for ds, dname in DS.items():
        get = {r["context"]: r for r in rows if r["dataset"] == ds}
        L.append(f"{dname}, CRPS & " + " & ".join(f"{get[c]['crps']:.3f}" for c in ctxs) + r" \\")
        L.append(f"{dname}, Cov.\\ 90 & " + " & ".join(f"{100 * get[c]['pit90']:.0f}" for c in ctxs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(SUPP, "real_aurora_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    aurora_table()
    with open(os.path.join(RES, "robust.json")) as f:
        rows = json.load(f)
    fm_path = os.path.join(RES, "real_fm.json")
    if os.path.exists(fm_path):                          # foundation models of run_fm.py
        with open(fm_path) as f:
            rows += json.load(f)
    table(rows)
    for ds in DS:
        print(f"== {ds}")
        for key, name in PRE + SMALL:
            rr = rows_of(rows, ds, key)
            if rr:
                extra = (f" cov80n={100*mean(rr,'cov80_native'):.1f} lin={100*mean(rr,'cov90_linear'):.1f} "
                         f"gau={100*mean(rr,'cov90_gaussian'):.1f} cla={100*mean(rr,'cov90_clamped'):.1f}"
                         if key.startswith("bolt") else "")
                print(f"  {key:22s} crps={mean(rr,'crps'):.3f} fair={mean(rr,'crps_fair'):.3f} "
                      f"es={mean(rr,'es'):.3f} esf={mean(rr,'es_fair'):.3f} ql={mean(rr,'qloss'):.3f} "
                      f"pit80={100*mean(rr,'pit80'):.1f} pit90={100*mean(rr,'pit90'):.1f}" + extra)
    pw_path = os.path.join(RES, "robust_perwindow.npz")
    if os.path.exists(pw_path):
        pw = dict(np.load(pw_path))
        fm_pw = os.path.join(RES, "real_fm_perwindow.npz")
        if os.path.exists(fm_pw):                        # same test windows, so the blocks agree
            fm = dict(np.load(fm_pw))
            for ds in DS:
                assert np.array_equal(fm[f"{ds}|sid"], pw[f"{ds}|sid"])
            pw.update(fm)
        block_ci(PerWindow(pw))


if __name__ == "__main__":
    main()
