"""Figures and the results tables for the simulation study.

Reads results/{main,steps,data,pit}.json and the training-budget results and writes
  ../outputs/figures/sim_cost_data.pdf    CRPS against network calls, training size and training budget
  ../outputs/figures/sim_calibration.pdf  reliability curves from PIT histograms (needs calib.py)
  ../outputs/tables/sim_table.tex       Table of main results (mean and s.d. over seeds)
  ../outputs/tables/sim_steps_table.tex   sampling-step sweep
  ../outputs/tables/sim_budget_table.tex  training budget
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle  # noqa: E402  (shared style of the data figures)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SIM_RESULTS", os.path.join(HERE, "results"))
FIG = os.path.join(HERE, "..", "outputs", "figures")
SEC = os.path.join(HERE, "..", "outputs", "tables")
SUPP = os.path.join(HERE, "..", "outputs", "tables")

MODELS = ["ar", "direct", "ardiff", "fulldiff", "flow", "masked"]
LABEL = {"ar": "AR, Gaussian head", "direct": "Direct multi-step",
         "ardiff": "AR + diffusion head", "fulldiff": "Full-window diffusion",
         "flow": "Full-window flow matching",
         "masked": "Masked diffusion (tokens)", "masked_rand": "Masked diffusion, random order"}
TABLE_MODELS = MODELS + ["masked_rand"]
SHORT = {"ar": "Autoregressive", "direct": "Direct", "ardiff": "Hybrid",
         "fulldiff": "Diffusion", "flow": "Flow matching", "masked": "Masked"}
# colour and bar hatch follow the type of generation (figstyle.py)
COLOR = {m: figstyle.MODEL_COLOR[m] for m in MODELS}
HATCH = figstyle.MODEL_HATCH
MARKER = {"ar": "o", "direct": "s", "ardiff": "^", "fulldiff": "D", "flow": "P", "masked": "v"}
DSNAME = {"seasonal": "Seasonal", "bimodal": "Regime switch (bimodal)", "lorenz": "Lorenz (chaotic)"}
INK, INK2, MUTED, GRID, AXIS = figstyle.INK, figstyle.INK2, figstyle.MUTED, figstyle.GRID, figstyle.AXIS

figstyle.apply()


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def agg(rows, key, **filt):
    vals = [r[key] for r in rows if all(r.get(k) == v for k, v in filt.items())]
    a = np.array(vals, dtype=float)
    return a.mean(0), a.std(0), len(vals)


def legend_row(fig, models=MODELS, y=1.02):
    handles = [plt.Line2D([0], [0], color=COLOR[m], marker=MARKER[m], lw=1.5, ms=4,
                          markeredgecolor="black", markeredgewidth=0.6) for m in models]
    fig.legend(handles, [LABEL[m] for m in models], loc="lower center", ncol=(len(models) + 1) // 2,
               frameon=False, bbox_to_anchor=(0.5, y), handlelength=1.8, columnspacing=1.2)


# ---------------------------------------------------------------------
# Figure: sample paths
# ---------------------------------------------------------------------

def fig_paths(ex):
    rows = ["bimodal", "lorenz"]
    fig, axes = plt.subplots(2, len(MODELS), figsize=(6.3, 2.75), sharex=True)
    for r, ds in enumerate(rows):
        ctx = np.array(ex[ds]["context"]); tru = np.array(ex[ds]["truth"])
        if ds == "bimodal":
            i = int(np.argmax(np.abs(tru[:, -1] - ctx[:, -1])))
        else:
            i = int(np.argmax(tru.std(1)))
        c, t = ctx[i], tru[i]
        xs_c = np.arange(-len(c) + 1, 1)
        xs_f = np.arange(1, len(t) + 1)
        paths_all = [np.array(ex[ds][m])[i] for m in MODELS]
        lo = min(c.min(), t.min(), min(p[:25].min() for p in paths_all))
        hi = max(c.max(), t.max(), max(p[:25].max() for p in paths_all))
        pad = 0.08 * (hi - lo)
        for j, m in enumerate(MODELS):
            ax = axes[r, j]
            paths = paths_all[j][:25]
            for p in paths:
                ax.plot(xs_f, p, color=COLOR[m], lw=0.5, alpha=0.45, solid_capstyle="round")
            ax.plot(xs_c[-24:], c[-24:], color=MUTED, lw=1.0)
            ax.plot(xs_f, t, color=INK, lw=1.1)
            ax.axvline(0.5, color=AXIS, lw=0.5)
            ax.set_ylim(lo - pad, hi + pad)
            ax.grid(False)
            if r == 0:
                ax.set_title(SHORT[m], fontsize=7)
            if j == 0:
                ax.set_ylabel(DSNAME[ds].split(" (")[0])
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("Step")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(os.path.join(FIG, "sim_paths.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure: CRPS against forecast step
# ---------------------------------------------------------------------

def load_h96():
    """Rows of horizon.py (H = 96), one results file per dataset; empty if not run."""
    rows = []
    for ds in ["seasonal", "bimodal", "lorenz"]:
        path = os.path.join(HERE, "results_h96", ds, "main.json")
        if os.path.exists(path):
            with open(path) as f:
                rows += json.load(f)
    return rows


def load_budget():
    """Learning-curve rows: budget.py plus the runs of main.json and robust.json it reuses."""
    rows = []
    for ds in ["bimodal", "lorenz"]:
        path = os.path.join(HERE, "results_budget", f"budget_{ds}.json")
        if os.path.exists(path):
            with open(path) as f:
                rows += json.load(f)
    if not rows:
        return []
    for r in load("main.json"):
        if r["dataset"] in ("bimodal", "lorenz"):
            rows.append(dict(r, train_steps=3000))
    reuse = {"fulldiff_9k": ("fulldiff", 9000), "fulldiff_30k": ("fulldiff", 30000), "flow_30k": ("flow", 30000)}
    path = os.path.join(HERE, "results_robust", "robust.json")
    if os.path.exists(path):
        with open(path) as f:
            for r in json.load(f):
                if r["model"] in reuse:
                    m, s = reuse[r["model"]]
                    rows.append(dict(r, model=m, train_steps=s))
    return rows


def _horizon_row(axes, rows, H):
    for ax, ds in zip(axes, ["seasonal", "bimodal", "lorenz"]):
        for m in MODELS:
            if not [r for r in rows if r["dataset"] == ds and r["model"] == m]:
                continue
            mu, sd, _ = agg(rows, "crps_h", dataset=ds, model=m)
            h = np.arange(1, len(mu) + 1)
            ax.fill_between(h, mu - sd, mu + sd, color=COLOR[m], alpha=0.12, lw=0)
            ax.plot(h, mu, color=COLOR[m], lw=1.5 if H == 24 else 1.2, marker=MARKER[m],
                    markevery=4 if H == 24 else 16, ms=3.5, markeredgecolor="black", markeredgewidth=0.5,
                    solid_capstyle="round")
        ax.set_title(f"{DSNAME[ds].split(' (')[0]}, $H$ = {H}")
        ax.set_xlabel("Forecast step")
        ax.set_xlim(0.5, H + 0.5)
        ax.set_xticks([1, 6, 12, 18, 24] if H == 24 else [1, 24, 48, 72, 96])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("CRPS (lower is better)")


def fig_horizon(main, long_rows=None):
    """CRPS against forecast step for H = 24 and, if available, for H = 96 (horizon.py)."""
    nrow = 2 if long_rows else 1
    fig, axes = plt.subplots(nrow, 3, figsize=(6.3, 2.25 * nrow), squeeze=False)
    _horizon_row(axes[0], main, 24)
    if long_rows:
        _horizon_row(axes[1], long_rows, 96)
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    legend_row(fig, y=0.98 if nrow == 1 else 0.99)
    fig.savefig(os.path.join(FIG, "sim_horizon.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure: cost-quality and data efficiency
# ---------------------------------------------------------------------

def _budget_row(axes, budget):
    """CRPS against training steps (two datasets) and coverage against training steps (Lorenz)."""
    panels = [("bimodal", "crps", "d"), ("lorenz", "crps", "e"), ("lorenz", "cov90", "f")]
    short = {"bimodal": "Regime switch", "lorenz": "Lorenz"}
    for ax, (ds, key, letter) in zip(axes, panels):
        for m in MODELS + ["masked_rand"]:
            sel = [r for r in budget if r["dataset"] == ds and r["model"] == m]
            xs = sorted({r["train_steps"] for r in sel})
            if not xs:
                continue
            ys = np.array([np.mean([r[key] for r in sel if r["train_steps"] == s]) for s in xs])
            es = np.array([np.std([r[key] for r in sel if r["train_steps"] == s]) for s in xs])
            scale = 100 if key == "cov90" else 1
            hollow = m == "masked_rand"
            base = "masked" if hollow else m
            ax.errorbar(xs, ys * scale, yerr=es * scale, color=COLOR[base], lw=1.0 if hollow else 1.5,
                        ls="--" if hollow else "-", marker=MARKER[base], ms=4,
                        markerfacecolor="white" if hollow else COLOR[base],
                        markeredgecolor=COLOR[base] if hollow else "black",
                        markeredgewidth=0.9 if hollow else 0.5, capsize=0, elinewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks([1000, 3000, 9000, 30000])
        ax.set_xticklabels(["1k", "3k", "9k", "30k"])
        ax.minorticks_off()
        ax.set_xlabel("Training steps")
        if key == "cov90":
            ax.axhline(86.5, color=MUTED, lw=0.8, ls=":")
            ax.text(30000, 85.8, "calibrated", fontsize=6, color=MUTED, ha="right", va="top")
            ax.set_title(f"{letter}  {short[ds]}: 90% coverage")
            ax.set_ylabel("Coverage (%)")
        else:
            ax.set_title(f"{letter}  {short[ds]}: training steps")
            ax.set_ylabel("CRPS (lower is better)")
            if ds == "lorenz":
                ax.set_yscale("log")
                ticks = [0.08, 0.1, 0.15, 0.2, 0.3, 0.4]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:g}" for t in ticks])
                ax.minorticks_off()


def fig_cost_data(steps, data, main, budget=None):
    nrow = 2 if budget else 1
    fig, axes2 = plt.subplots(nrow, 3, figsize=(6.3, 2.35 * nrow), squeeze=False)
    axes = axes2[0]
    if budget:
        _budget_row(axes2[1], budget)
    for ax, ds in zip(axes[:2], ["seasonal", "lorenz"]):
        for m in MODELS:
            for order in ["confidence", "random"]:
                sel = [r for r in steps if r["dataset"] == ds and r["model"] == m
                       and r.get("order", "confidence") == order]
                if not sel:
                    continue
                settings = sorted({r["setting"] for r in sel}, key=lambda v: -1 if v is None else v)
                xs, ys, es = [], [], []
                for s in settings:
                    rows = [r for r in sel if r["setting"] == s]
                    xs.append(rows[0]["calls"])
                    v = np.array([r["crps"] for r in rows])
                    ys.append(v.mean()); es.append(v.std())
                hollow = order == "random"
                ax.errorbar(xs, ys, yerr=es, color=COLOR[m], lw=1.0 if hollow else 1.5,
                            ls="--" if hollow else "-", marker=MARKER[m], ms=4,
                            markerfacecolor="white" if hollow else COLOR[m],
                            markeredgecolor=COLOR[m] if hollow else "white",
                            markeredgewidth=0.9 if hollow else 0.5, capsize=0, elinewidth=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{'a' if ds == 'seasonal' else 'b'}  {DSNAME[ds].split(' (')[0]}: cost")
        ax.set_xlabel("Sequential network calls per forecast")
        ax.grid(True, axis="both", which="major")
    axes[0].set_ylabel("CRPS (lower is better)")

    ax = axes[2]
    sizes = sorted({r["n_train"] for r in data})
    for m in MODELS:
        xs, ys, es = [], [], []
        for n in sizes:
            v = np.array([r["crps"] for r in data if r["model"] == m and r["n_train"] == n])
            xs.append(n); ys.append(v.mean()); es.append(v.std())
        v = np.array([r["crps"] for r in main if r["dataset"] == "lorenz" and r["model"] == m])
        xs.append(16384); ys.append(v.mean()); es.append(v.std())
        ax.errorbar(xs, ys, yerr=es, color=COLOR[m], lw=1.5, marker=MARKER[m], ms=4,
                    markeredgecolor="black", markeredgewidth=0.5, capsize=0, elinewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 1024, 4096, 16384])
    ax.set_xticklabels(["256", "1,024", "4,096", "16,384"])
    ax.set_title("c  Lorenz: training data")
    ax.set_xlabel("Training windows")
    fig.tight_layout(w_pad=1.0, h_pad=1.4)
    legend_row(fig, y=0.98 if nrow == 1 else 0.99)
    fig.savefig(os.path.join(FIG, "sim_cost_data.pdf"))
    plt.close(fig)


def _table_rows(rows, ds, m, metrics):
    sel = [r for r in rows if r["dataset"] == ds and r["model"] == m]
    return [_ms([r[k] for r in sel], nd, sc) if sel else "--" for k, nd, sc in metrics]


def supp_table_h96(long_rows, main):
    """Supplementary table: H = 96 against H = 24 (mean and s.d. over seeds)."""
    metrics = [("mse", 3, 1), ("crps", 3, 1), ("es", 2, 1), ("cov90", 1, 100)]
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.08}",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{\textbf{Longer horizon.} The main experiment repeated with a horizon of $H = 96$ instead "
         r"of 24, with the same context, networks, training budget and evaluation; masked diffusion unmasks one "
         r"token per round. Mean (standard deviation) over three seeds. Series are standardized with the "
         r"statistics of the whole training windows, which on the regime-switch data grow with the drift over "
         r"the longer horizon (standard deviation 6.78 against 1.90 for $H = 24$), so the scores of that dataset "
         r"are in different units at the two horizons; for the seasonal and Lorenz data the two scales differ "
         r"by less than 3\%. The energy score grows with the length of the path and is not comparable across "
         r"horizons. Cov.\ 90: coverage (\%) of sample-quantile intervals (about 86.5\% expected for a "
         r"calibrated forecaster with 50 samples).}",
         r"\label{stab:simh96}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{@{}l l r r r r@{}}", r"\toprule",
         r"Data & Forecaster & MSE & CRPS & ES & Cov.\ 90 \\", r"\midrule"]
    for ds in ["seasonal", "bimodal", "lorenz"]:
        for i, m in enumerate(TABLE_MODELS):
            L.append(f"{DSNAME[ds] if i == 0 else ''} & {LABEL[m]} & "
                     + " & ".join(_table_rows(long_rows, ds, m, metrics)) + r" \\")
        if ds != "lorenz":
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(SUPP, "sim_h96_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def supp_table_budget(budget):
    """Supplementary table: scores against the training budget (mean over seeds)."""
    budgets = sorted({r["train_steps"] for r in budget})
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.05}",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{\textbf{Training budget.} Every forecaster was trained for 1,000 to 30,000 steps, each with a "
         r"cosine learning-rate schedule over its own budget. Mean (standard deviation) over three seeds. The "
         r"3,000-step rows repeat Table~\ref{tab:sim}, and the 9,000- and 30,000-step rows of full-window diffusion "
         r"and the 30,000-step row of flow matching come from earlier runs of the same code.}",
         r"\label{stab:simbudget}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{@{}l r r r r r r r@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Regime switch (bimodal)} & \multicolumn{3}{c}{Lorenz (chaotic)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
         r"Forecaster & Steps & CRPS & ES & Cov.\ 90 & CRPS & ES & Cov.\ 90 \\", r"\midrule"]
    metrics = [("crps", 3, 1), ("es", 2, 1), ("cov90", 1, 100)]
    for j, m in enumerate(TABLE_MODELS):
        for i, s in enumerate(budgets):
            sub = [r for r in budget if r["train_steps"] == s]
            cells = _table_rows(sub, "bimodal", m, metrics) + _table_rows(sub, "lorenz", m, metrics)
            L.append(f"{LABEL[m] if i == 0 else ''} & {s:,} & " + " & ".join(cells) + r" \\")
        if j < len(TABLE_MODELS) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(SUPP, "sim_budget_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def summary_extra(long_rows, main, budget):
    """Print the numbers of the longer-horizon and training-budget experiments quoted in the text."""
    if long_rows:
        print("== H = 96 (mean over seeds) vs H = 24 ==")
        for ds in ["seasonal", "bimodal", "lorenz"]:
            for m in TABLE_MODELS:
                if not [r for r in long_rows if r["dataset"] == ds and r["model"] == m]:
                    continue
                c96 = agg(long_rows, "crps", dataset=ds, model=m)[0]
                c24 = agg(main, "crps", dataset=ds, model=m)[0]
                ch = agg(long_rows, "crps_h", dataset=ds, model=m)[0]
                cv = agg(long_rows, "cov90", dataset=ds, model=m)[0]
                es = agg(long_rows, "es", dataset=ds, model=m)[0]
                print(f"{ds:9s} {m:12s} crps96={c96:.3f} (crps24={c24:.3f}) es96={es:.2f} cov90={cv:.2f} "
                      f"crps@1-24={ch[:24].mean():.3f} crps@25-48={ch[24:48].mean():.3f} "
                      f"crps@73-96={ch[72:].mean():.3f}")
            ar = agg(long_rows, "crps_h", dataset=ds, model="ar")[0]
            for m in ["fulldiff", "flow", "direct", "ardiff", "masked_rand"]:
                if [r for r in long_rows if r["dataset"] == ds and r["model"] == m]:
                    x = agg(long_rows, "crps_h", dataset=ds, model=m)[0]
                    print(f"   {ds} {m}/ar CRPS ratio by quarter: "
                          + " ".join(f"{x[i:i + 24].mean() / ar[i:i + 24].mean():.2f}" for i in range(0, 96, 24)))
    if budget:
        print("== training budget (mean over seeds) ==")
        for ds in ["bimodal", "lorenz"]:
            for m in TABLE_MODELS:
                out = []
                for s in sorted({r["train_steps"] for r in budget}):
                    sel = [r for r in budget if r["dataset"] == ds and r["model"] == m and r["train_steps"] == s]
                    if sel:
                        out.append(f"{s // 1000}k: {np.mean([r['crps'] for r in sel]):.3f}/"
                                   f"{np.mean([r['es'] for r in sel]):.2f}/{100 * np.mean([r['cov90'] for r in sel]):.0f}%"
                                   f"(n={len(sel)})")
                print(f"{ds:8s} {m:12s} " + "  ".join(out))


# ---------------------------------------------------------------------
# Figure: calibration (reliability curves from PIT histograms)
# ---------------------------------------------------------------------

LEVELS = np.round(np.arange(0.1, 0.91, 0.1), 2)


def coverage_curve(hist):
    """Empirical coverage of central intervals at LEVELS from a 100-bin PIT histogram."""
    h = np.asarray(hist, dtype=float)
    h = h / h.sum()
    nb = len(h)
    out = []
    for a in LEVELS:
        lo, hi = int(round((1 - a) / 2 * nb)), int(round((1 + a) / 2 * nb))
        out.append(h[lo:hi].sum())
    return np.array(out)


def fig_calibration(pit):
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), sharey=True)
    for ax, ds in zip(axes, ["seasonal", "bimodal", "lorenz"]):
        ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for m in MODELS + ["masked_rand"]:
            curves = [coverage_curve(r["hist"]) for r in pit if r["dataset"] == ds and r["model"] == m]
            if not curves:
                continue
            mu = 100 * np.mean(curves, 0)
            base = "masked" if m == "masked_rand" else m
            hollow = m == "masked_rand"
            ax.plot(100 * LEVELS, mu, color=COLOR[base], lw=1.0 if hollow else 1.5,
                    ls="--" if hollow else "-", marker=MARKER[base], ms=3.5,
                    markerfacecolor="white" if hollow else COLOR[base],
                    markeredgecolor=COLOR[base] if hollow else "black",
                    markeredgewidth=0.8 if hollow else 0.5, zorder=3)
        ax.set_title(DSNAME[ds])
        ax.set_xlabel("Nominal coverage (%)")
        ax.set_xlim(5, 95)
        ax.set_ylim(0, 100)
        ax.set_xticks([10, 30, 50, 70, 90])
        ax.grid(True, axis="both")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Empirical coverage (%)")
    fig.tight_layout(w_pad=1.2)
    handles = [plt.Line2D([0], [0], color=COLOR[m], marker=MARKER[m], lw=1.5, ms=4,
                          markeredgecolor="black", markeredgewidth=0.6) for m in MODELS]
    handles.append(plt.Line2D([0], [0], color=COLOR["masked"], marker=MARKER["masked"], lw=1.0, ls="--",
                              ms=4, markerfacecolor="white", markeredgecolor=COLOR["masked"]))
    labels = [LABEL[m] for m in MODELS] + ["Masked, random order"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.01), handlelength=2.2, columnspacing=1.2)
    fig.savefig(os.path.join(FIG, "sim_calibration.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------

def table(main):
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize",
             r"\renewcommand{\arraystretch}{1.1}", r"\setlength{\tabcolsep}{4.5pt}",
             r"\caption{\textbf{Simulation results.} Mean (standard deviation) over three training seeds on "
             r"500 test windows with 50 samples each; lower is better for MSE, CRPS and ES. "
             r"Cov.\ 90\%: coverage of the interval between the 5\% and 95\% sample quantiles, about 86.5 for a "
             r"calibrated forecaster with 50 samples. Calls: sequential network evaluations per forecast; time: "
             r"sampling time for the 500 windows. Best MSE, CRPS and ES of each dataset in bold. Naive: seasonal "
             r"naive forecast (last value for Lorenz), a reference for the MSE.}",
             r"\label{tab:sim}",
             r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
             r"Model & MSE & CRPS & ES & Cov.\ 90\% & Calls & Time (s) \\"]
    for ds in ["seasonal", "bimodal", "lorenz"]:
        stats = {}
        for m in TABLE_MODELS:
            stats[m] = {k: agg(main, k, dataset=ds, model=m)[:2]
                        for k in ["mse", "crps", "es", "cov90", "time_s", "calls"]}
        best = {k: min(TABLE_MODELS, key=lambda m: stats[m][k][0]) for k in ["mse", "crps", "es"]}
        naive = [r["naive_mse"] for r in main if r["dataset"] == ds][0]
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{DSNAME[ds]}}}}} \\")
        for m in TABLE_MODELS:
            def fmt(k, scale=1.0, nd=3):
                mu, sd = stats[m][k]
                s = f"{mu*scale:.{nd}f} ({sd*scale:.{nd}f})"
                return rf"\textbf{{{s}}}" if best.get(k) == m else s
            lines.append(f"{LABEL[m]} & {fmt('mse')} & {fmt('crps')} & {fmt('es', nd=2)} & "
                         f"{fmt('cov90', 100, 1)} & {int(stats[m]['calls'][0]):,} & "
                         f"{stats[m]['time_s'][0]:.2f} \\\\")
        lines.append(f"Naive & {naive:.3f} & & & & & \\\\")
    lines.append(r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(SEC, "sim_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Multivariate experiment (sim_multi.py)
# ---------------------------------------------------------------------

MULTI_NAME = {"coupled": "Coupled regime switch", "lorenz3": "Lorenz, three coordinates"}


def _dot_panel(ax, rows, key, scale=1.0, label_models=True):
    order = TABLE_MODELS[::-1]
    for i, m in enumerate(order):
        v = scale * np.array([r[key] for r in rows if r["model"] == m])
        base = "masked" if m == "masked_rand" else m
        hollow = m == "masked_rand"
        ax.plot([v.min(), v.max()], [i, i], color=COLOR[base], lw=1.2, solid_capstyle="round")
        ax.plot(v.mean(), i, MARKER[base], ms=5, color=COLOR[base],
                markerfacecolor="white" if hollow else COLOR[base],
                markeredgecolor=COLOR[base] if hollow else "black", markeredgewidth=0.8 if hollow else 0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[m] for m in order] if label_models else [])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)


def bar_style(m):
    """Fill colour and hatch of a forecaster's bars; random-order masking is a lighter tint."""
    base = "masked" if m == "masked_rand" else m
    fill = figstyle.lighter(COLOR[base], 0.55) if m == "masked_rand" else COLOR[base]
    return fill, HATCH[base]


def _bar_panel(ax, rows, key, scale=1.0, fmt="{:.2f}"):
    x = np.arange(len(TABLE_MODELS))
    for i, m in enumerate(TABLE_MODELS):
        v = scale * np.array([r[key] for r in rows if r["model"] == m])
        fill, hatch = bar_style(m)
        figstyle.bars(ax, [i], [v.mean()], 0.78, fill, hatch, yerr=[v.std()], fmt=fmt, fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(TABLE_MODELS) - 0.4)


def fig_multi(multi):
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5))
    co = [r for r in multi if r["dataset"] == "coupled"]
    ax = axes[0]
    truth = 100 * np.mean([r["agree_truth"] for r in co])
    _bar_panel(ax, co, "agree", 100, fmt="{:.0f}")
    ax.axhline(truth, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(25, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(len(TABLE_MODELS) - 0.5, 27, "independent channels", ha="right", va="bottom", fontsize=6,
            color=INK2, zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Paths with all channels agreeing (%)")
    ax.set_title("a  Coupled: channel agreement")
    for ax, ds, t in [(axes[1], "coupled", "b  Coupled: joint score"),
                      (axes[2], "lorenz3", "c  Three-coordinate Lorenz")]:
        sel = [r for r in multi if r["dataset"] == ds]
        _bar_panel(ax, sel, "es", fmt="{:.2f}")
        top = max(np.mean([r["es"] for r in sel if r["model"] == m]) for m in TABLE_MODELS)
        ax.set_ylim(0, 1.18 * top)
        ax.set_ylabel("Energy score (lower is better)")
        ax.set_title(t)
    handles = [figstyle.patch(*bar_style(m), LABEL[m]) for m in TABLE_MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.98), columnspacing=1.2)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIG, "sim_multi.pdf"))
    plt.close(fig)


def supp_table_multi(multi):
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{\textbf{Multivariate simulation.} Three channels forecast jointly; mean (standard deviation) "
         r"over three training seeds, 500 test windows and 50 samples per window. ES and VS are the energy score "
         r"and the variogram score (order 0.5, averaged over pairs of the 72 forecast components) of the joint "
         r"forecast. Agree is the share of sample paths in which all three channels move in the same direction; "
         r"in the data it is close to 100\%, and for independent channels it would be 25\%.}",
         r"\label{stab:simmulti}",
         r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
         r"Model & CRPS & ES & VS & Cov.\ 90\% & Agree (\%) & Calls \\"]
    for ds in ["coupled", "lorenz3"]:
        L += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{MULTI_NAME[ds]}}}}} \\"]
        for m in TABLE_MODELS:
            rows = [r for r in multi if r["dataset"] == ds and r["model"] == m]
            g = lambda k: [r[k] for r in rows]
            agree = _ms(g("agree"), 1, 100) if ds == "coupled" else "--"
            L.append(f"{LABEL[m]} & {_ms(g('crps'))} & {_ms(g('es'), 2)} & {_ms(g('vs'), 4)} & "
                     f"{_ms(g('cov90'), 1, 100)} & {agree} & {rows[0]['calls']:,} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(SUPP, "sim_multi_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def _ms(vals, nd=3, scale=1.0):
    a = scale * np.asarray(vals, dtype=float)
    return f"{a.mean():.{nd}f} ({a.std():.{nd}f})"


def supp_tables(main, steps, data):
    """Supplementary table: sampling-step sweep, written to sim_steps_table.tex."""
    os.makedirs(SUPP, exist_ok=True)
    L = []
    # --- sampling-step sweep -------------------------------------------------
    L += [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
          r"\caption{\textbf{Sampling cost and quality.} Results behind Fig.~\ref{sfig:simcost}a,b. "
          r"The setting is the number of denoising steps for the diffusion models, the number of Euler steps for flow matching, and the number of "
          r"unmasking rounds for masked diffusion. Calls are sequential network evaluations per forecast; "
          r"time is wall-clock sampling time for 500 test windows. Mean (standard deviation) over three seeds.}",
          r"\label{stab:simsteps}",
          r"\begin{fitwidth}", r"\begin{tabular}{@{}l l r r r r r@{}}", r"\toprule",
          r"Model & Setting & Calls & CRPS & ES & Cov.\ 90\% & Time (s) \\"]
    for ds in ["seasonal", "lorenz"]:
        L += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{DSNAME[ds]}}}}} \\"]
        for m in MODELS:
            keys = sorted({(r["setting"], r.get("order", "confidence")) for r in steps
                           if r["dataset"] == ds and r["model"] == m},
                          key=lambda k: (k[1], -1 if k[0] is None else k[0]))
            for s, o in keys:
                v = [r for r in steps if r["dataset"] == ds and r["model"] == m and r["setting"] == s
                     and r.get("order", "confidence") == o]
                if s is None:
                    setting = "--"
                elif m == "masked":
                    setting = f"{s} round{'' if s == 1 else 's'}" + (", random" if o == "random" else "")
                else:
                    setting = f"{s} steps"
                g = lambda k: [r[k] for r in v]
                L.append(f"{LABEL[m] if m != 'masked' else 'Masked diffusion (tokens)'} & {setting} & "
                         f"{v[0]['calls']:,} & {_ms(g('crps'))} & {_ms(g('es'), 2)} & "
                         f"{_ms(g('cov90'), 1, 100)} & {np.mean(g('time_s')):.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{fitwidth}", r"\end{table}", ""]
    with open(os.path.join(SUPP, "sim_steps_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def summary(main, steps, data):
    """Print numbers quoted in the text."""
    print("== main (mean over seeds) ==")
    for ds in ["seasonal", "bimodal", "lorenz"]:
        for m in TABLE_MODELS:
            s = {k: agg(main, k, dataset=ds, model=m)[0] for k in ["mse", "crps", "es", "cov50", "cov90", "time_s"]}
            p = [r["params"] for r in main if r["dataset"] == ds and r["model"] == m][0]
            ch = agg(main, "crps_h", dataset=ds, model=m)[0]
            print(f"{ds:9s} {m:8s} mse={s['mse']:.3f} crps={s['crps']:.3f} es={s['es']:.2f} "
                  f"cov50={s['cov50']:.2f} cov90={s['cov90']:.2f} t={s['time_s']:.2f}s "
                  f"crps@1={ch[0]:.3f} crps@24={ch[-1]:.3f} params={p}")
    if steps:
        print("== steps ==")
        for ds in ["seasonal", "lorenz"]:
            for m in MODELS:
                keys = sorted({(r['setting'], r.get('order', 'confidence')) for r in steps
                               if r['dataset'] == ds and r['model'] == m},
                              key=lambda k: (k[1], -1 if k[0] is None else k[0]))
                for s, o in keys:
                    v = [r for r in steps if r['dataset'] == ds and r['model'] == m and r['setting'] == s
                         and r.get('order', 'confidence') == o]
                    print(f"{ds} {m} {o} setting={s} calls={v[0]['calls']} crps={np.mean([x['crps'] for x in v]):.3f} "
                          f"es={np.mean([x['es'] for x in v]):.2f} cov90={np.mean([x['cov90'] for x in v]):.2f} "
                          f"t={np.mean([x['time_s'] for x in v]):.2f}")
    if data:
        print("== data ==")
        for m in MODELS:
            for n in sorted({r['n_train'] for r in data}):
                v = [r['crps'] for r in data if r['model'] == m and r['n_train'] == n]
                print(f"{m} n={n} crps={np.mean(v):.3f} ({np.std(v):.3f})")


def main():
    """Writes only the outputs used in the Supplementary Material: Table S31 (sim_table), Figs. S5 and S6
    (calibration, cost and training budget) and the sampling-step and training-budget tables."""
    os.makedirs(FIG, exist_ok=True)
    main_rows = load("main.json")
    steps = load("steps.json") if os.path.exists(os.path.join(RES, "steps.json")) else []
    data = load("data.json") if os.path.exists(os.path.join(RES, "data.json")) else []
    budget = load_budget()
    if steps and data:
        fig_cost_data(steps, data, main_rows, budget)
        supp_tables(main_rows, steps, data)
    if budget:
        supp_table_budget(budget)
    pit_path = os.path.join(RES, "pit.json")
    if os.path.exists(pit_path):
        pit = load("pit.json")
        fig_calibration(pit)
        print("== PIT coverage (mean over seeds) ==")
        for ds in ["seasonal", "bimodal", "lorenz"]:
            for m in TABLE_MODELS:
                cs = [coverage_curve(r["hist"]) for r in pit if r["dataset"] == ds and r["model"] == m]
                if cs:
                    print(f"{ds:9s} {m:12s} " + " ".join(f"{100*v:5.1f}" for v in np.mean(cs, 0)))
    table(main_rows)
    summary(main_rows, steps, data)


if __name__ == "__main__":
    main()
