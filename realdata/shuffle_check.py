"""Calibration of the shuffle test and of the rank dependence measures on forecasts whose dependence is known.

The observed 24-step path is drawn from N(0, R(phi0)), where R(phi) has entries phi^|i-j|, as the conditional
distribution of an autoregressive process (phi0 = 0.9, or 0.97 for stronger dependence). Each forecaster draws
100 paths from N(b_n, s_n^2 R(phi)) with its own dependence phi and spread s_n, and a location b_n shared by all
steps of window n (b_n = 0 unless a bias is set: b_n ~ N(0, tau^2); s_n = s unless the spread varies between
windows: s_n = s exp(0.5 xi_n), xi_n ~ N(0, 1)).

Part 1 (Table S5, one run of 300 windows): the shuffle gain, the relative increase of the variogram score when the
paths are shuffled independently at each step, with a bootstrap interval over windows, and the two rank
correlations. The first seven cases are those of the first version of this script, with the same random numbers.

Part 2 (Table S6, R replications of 300 windows): Monte Carlo means and standard errors of
  rho model     Spearman correlation of adjacent steps across the paths of a window, averaged over steps and windows
  rho observed  Spearman correlation of adjacent steps across windows of the randomized PIT
                u = (#paths below + V) / (S + 1), V ~ U(0, 1), averaged over steps (as in the benchmark)
  outside       share of outcomes below or above all S paths; their u are tied at the ends up to the noise V
  inside        rho observed over the windows whose outcome lies inside the range of the paths at both steps
  mid-PIT       rho observed from the non-randomized mid-PIT (#below + (#equal + 1) / 2) / (S + 1), with tied
                ranks averaged (#equal = 0 for continuous forecasts)
  cov80         share of PIT values in (0.1, 0.9)

Usage:  python shuffle_check.py      writes ../outputs/tables/shuffle_check_table.tex, shuffle_dep_table.tex (read by
                                     the former) and results/shuffle_check_ext.json (results/shuffle_check.json,
                                     the first version, is not overwritten); about 12 minutes on a CPU
        python shuffle_check.py --tables     rewrites the two tables from results/shuffle_check_ext.json
"""

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU only (an empty value is dropped on Windows)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

import run  # noqa: E402

H, S, N, PHI = 24, 100, 300, 0.9
R_REP = 1000
HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = os.path.join(HERE, "..", "outputs", "tables")
# name, phi0 (truth), phi (forecast), s (spread, or median spread), tau (location bias), sig (log-sd of the spread)
CASES = [("Correct dependence and spread", 0.9, 0.9, 1.0, 0, 0), ("No dependence", 0.9, 0.0, 1.0, 0, 0),
         ("Weaker dependence", 0.9, 0.5, 1.0, 0, 0), ("Stronger dependence", 0.9, 0.99, 1.0, 0, 0),
         ("Correct dependence, intervals too narrow", 0.9, 0.9, 0.6, 0, 0),
         ("Correct dependence, intervals too wide", 0.9, 0.9, 1.5, 0, 0),
         ("No dependence, intervals too narrow", 0.9, 0.0, 0.6, 0, 0),
         # added in the revision
         ("Correct dependence, intervals far too narrow", 0.9, 0.9, 0.4, 0, 0),
         ("Correct dependence, intervals far too narrow", 0.9, 0.9, 0.28, 0, 0),
         ("Correct dependence, intervals far too narrow", 0.9, 0.9, 0.2, 0, 0),
         (r"Location bias, $\tau = 0.5$", 0.9, 0.9, 1.0, 0.5, 0),
         (r"Location bias, $\tau = 1$", 0.9, 0.9, 1.0, 1.0, 0),
         (r"Location bias, $\tau = 0.5$, narrow", 0.9, 0.9, 0.4, 0.5, 0),
         ("Window-varying spread", 0.9, 0.9, 1.0, 0, 0.5),
         ("Window-varying spread, narrow", 0.9, 0.9, 0.4, 0, 0.5),
         ("Stronger true dependence, correct", 0.97, 0.97, 1.0, 0, 0),
         ("Stronger true dependence, correct, narrow", 0.97, 0.97, 0.4, 0, 0),
         ("Stronger true dependence, correct, narrow", 0.97, 0.97, 0.28, 0, 0)]
N_ORIGINAL = 7


def chol(phi):
    i = torch.arange(H, dtype=torch.float64)
    R = phi ** (i[:, None] - i[None, :]).abs()
    return torch.linalg.cholesky(R + 1e-9 * torch.eye(H, dtype=torch.float64))


def ranks(a, dim):
    return a.argsort(dim).argsort(dim).float()


def rank_dependence(x, y, g):
    """Lag-one rank correlation of the forecast paths (within each window, across samples, averaged) and
    of the observations' PIT values (across windows). They agree for a forecaster with correct dependence,
    whatever its spread, as long as few outcomes fall outside the paths."""
    rx = ranks(x, 1)
    rx = (rx - rx.mean(1, keepdim=True)) / rx.std(1, keepdim=True)
    rho_m = float(((rx[:, :, :-1] * rx[:, :, 1:]).sum(1) / (x.shape[1] - 1)).mean())
    below = (x < y[:, None]).sum(1).float()
    u = (below + torch.rand(below.shape, generator=g)) / (x.shape[1] + 1)
    ru = ranks(u, 0)
    ru = (ru - ru.mean(0)) / ru.std(0)
    rho_o = float((ru[:, :-1] * ru[:, 1:]).sum(0).mean() / (u.shape[0] - 1))
    return rho_m, rho_o


# ------------------------------------------------------------------------------- Part 1: one run per case
def part1():
    g = torch.Generator().manual_seed(0)
    y0 = {PHI: (chol(PHI) @ torch.randn(H, N, generator=g, dtype=torch.float64)).T.float()}       # (N, H)
    rows = []
    for ci, (name, phi0, phi, s, tau, sig) in enumerate(CASES):
        if phi0 not in y0:                                    # observations with another true dependence
            y0[phi0] = (chol(phi0) @ torch.randn(H, N, generator=g, dtype=torch.float64)).T.float()
        y = y0[phi0]
        z = (chol(phi) @ torch.randn(H, N * S, generator=g, dtype=torch.float64)).T.float().reshape(N, S, H)
        if sig:                                               # spread of each window: s exp(sig xi_n)
            z = z * (s * torch.exp(sig * torch.randn(N, generator=g, dtype=torch.float64))).float()[:, None, None]
        else:
            z = s * z                                         # as in the first version
        x = z + tau * torch.randn(N, generator=g, dtype=torch.float64).float()[:, None, None] if tau else z
        perm = torch.argsort(torch.rand(x.shape, generator=g), dim=1)
        vs = run.variogram_score(x, y).numpy()
        vss = run.variogram_score(x.gather(1, perm), y).numpy()
        rng = np.random.default_rng(1)
        boot = [vss[i].sum() / vs[i].sum() - 1 for i in (rng.integers(0, N, N) for _ in range(2000))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rho_m, rho_o = rank_dependence(x, y, g)
        rows.append(dict(case=name, phi0=phi0, phi=phi, s=s, tau=tau, sig=sig, original=ci < N_ORIGINAL,
                         gain=float(100 * (vss.mean() / vs.mean() - 1)), lo=float(100 * lo), hi=float(100 * hi),
                         rho_model=rho_m, rho_obs=rho_o))
        print(f"{name:45s} phi0={phi0:4.2f} phi={phi:4.2f} s={s:4.2f} tau={tau:3.1f} sig={sig:3.1f} "
              f"gain={rows[-1]['gain']:+6.1f}% [{100*lo:+.1f}, {100*hi:+.1f}] "
              f"model rho={rho_m:.3f} observed rho={rho_o:.3f}", flush=True)
    return rows


# --------------------------------------------------------------------- Part 2: replicated dependence measures
def _spearman_steps(r, n):
    """Spearman correlation of adjacent steps from tie-free ranks r (..., n, H) along axis -2."""
    d = r[..., 1:] - r[..., :-1]
    return 1 - 6 * (d ** 2).sum(-2) / (n * (n ** 2 - 1))


def replicate(phi0, phi, s, tau, sig, rng):
    L0 = np.linalg.cholesky(phi0 ** np.abs(np.subtract.outer(np.arange(H), np.arange(H))))
    L1 = np.linalg.cholesky(phi ** np.abs(np.subtract.outer(np.arange(H), np.arange(H))) + 1e-9 * np.eye(H))
    y = rng.standard_normal((N, H)) @ L0.T
    x = (rng.standard_normal((N * S, H)) @ L1.T).reshape(N, S, H)
    sn = s * np.exp(sig * rng.standard_normal(N)) if sig else np.full(N, s)
    bn = tau * rng.standard_normal(N) if tau else np.zeros(N)
    x = bn[:, None, None] + sn[:, None, None] * x
    rx = x.argsort(1).argsort(1).astype(np.float64)
    rho_m = _spearman_steps(rx, S).mean()                                  # (N, H-1) -> mean
    below = (x < y[:, None]).sum(1)
    equal = (x == y[:, None]).sum(1)
    u = (below + rng.random(below.shape)) / (S + 1)
    ru = u.argsort(0).argsort(0).astype(np.float64)                       # (N, H): ranks across windows
    rho_o = _spearman_steps(ru, N).mean()
    out = (below == 0) | (below == S)
    inside = []
    for h in range(H - 1):
        m = ~out[:, h] & ~out[:, h + 1]
        if m.sum() > 3:
            inside.append(stats.spearmanr(u[m, h], u[m, h + 1]).statistic)
    mid = (below + (equal + 1) / 2) / (S + 1)
    rm = stats.rankdata(mid, axis=0)
    rm = (rm - rm.mean(0)) / rm.std(0)
    rho_mid = (rm[:, :-1] * rm[:, 1:]).mean(0).mean()
    cov80 = ((u > 0.1) & (u < 0.9)).mean()
    return dict(rho_model=rho_m, rho_obs=rho_o, outside=out.mean(), inside=np.mean(inside) if inside else np.nan,
                mid=rho_mid, cov80=cov80)


def part2(n_rep=R_REP):
    res = []
    seen = {}
    for ci, (name, phi0, phi, s, tau, sig) in enumerate(CASES):
        if ci in (1, 2, 3, 6):          # cases that differ only in phi: rho observed does not depend on phi
            continue
        key = (phi0, phi, s, tau, sig)
        if key in seen:
            continue
        seen[key] = True
        rng = np.random.default_rng([2026, ci])
        reps = [replicate(phi0, phi, s, tau, sig, rng) for _ in range(n_rep)]
        row = dict(case=name, phi0=phi0, phi=phi, s=s, tau=tau, sig=sig, n_rep=n_rep)
        for k in reps[0]:
            v = np.array([r[k] for r in reps], dtype=float)
            row[k], row[k + "_se"] = float(np.nanmean(v)), float(np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v))))
        res.append(row)
        print(f"rep {name:45s} phi0={phi0:4.2f} phi={phi:4.2f} s={s:4.2f} tau={tau:3.1f} sig={sig:3.1f} "
              + " ".join(f"{k}={row[k]:.4f}({row[k + '_se']:.4f})" for k in reps[0]), flush=True)
    return res


# ------------------------------------------------------------------------------------------------- tables
def s_cell(c):
    t = f"{c['s']:.2f}".rstrip("0")
    return t + "0" if t.endswith(".") else t


def label(c, original_names=True):
    """Row label: the first seven cases keep their names in Table S5; the others are named by their parameters
    (the phi0 column tells the cases with stronger true dependence apart)."""
    if original_names and c.get("original"):
        return c["case"]
    if c["tau"]:
        return rf"Location bias, $\tau = {c['tau']:g}$" + (", narrow" if c["s"] < 1 else "")
    if c["sig"]:
        return "Window-varying spread" + (", narrow" if c["s"] < 1 else "")
    if c["phi"] != c["phi0"]:
        return c["case"]
    return {1.0: "Correct dependence and spread", 0.6: "Correct dependence, too narrow",
            1.5: "Correct dependence, too wide"}.get(c["s"], "Correct dependence, far too narrow")


def tables(rows, rep):
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{\textbf{Calibration of the shuffle test.} The observed 24-step path is Gaussian with correlation "
         r"$\phi_0^{|i-j|}$ between steps $i$ and $j$. Each forecaster draws 100 Gaussian paths with correlation "
         r"parameter $\phi$ and $s$ times the true standard deviation around the true mean. Location bias: the paths "
         r"of window $n$ are shifted by $b_n \sim \mathcal{N}(0, \tau^2)$. Window-varying spread: the standard "
         r"deviation of window $n$ is $s\,e^{0.5\xi_n}$ with $\xi_n \sim \mathcal{N}(0, 1)$. Shuffle gain, "
         r"Eq.~(\ref{eq:shuffle}), over 300 windows with a 95\% bootstrap interval over windows; $\rho$ model and "
         r"$\rho$ observed as in Eqs.~(\ref{eq:rhomodel}) and~(\ref{eq:rhoobs}).}",
         r"\label{stab:shufflecheck}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
         r"Forecaster & $\phi_0$ & $\phi$ & $s$ & Shuffle gain (\%) & $\rho$ model & $\rho$ observed \\",
         r"\midrule"]
    for i, c in enumerate(rows):
        if i == N_ORIGINAL or (i and c["phi0"] != rows[i - 1]["phi0"]):
            L.append(r"\addlinespace")
        L.append(f"{label(c)} & {c['phi0']:.2f} & {c['phi']:.2f} & {s_cell(c)} & ${c['gain']:+.1f}$ "
                 f"(${c['lo']:+.1f}$, ${c['hi']:+.1f}$) & {c['rho_model'] if round(c['rho_model'], 2) else 0.0:.2f} & {c['rho_obs']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "", r"\input{supplementary/shuffle_dep_table}"]
    with open(os.path.join(SUPP, "shuffle_check_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    se = {k: max(r[k + "_se"] for r in rep) for k in ["rho_model", "rho_obs", "inside", "mid", "outside", "cov80"]}
    rho_se = max(se["rho_model"], se["rho_obs"], se["mid"])
    pop = {p: 6 / np.pi * np.arcsin(p / 2) for p in (0.9, 0.97)}
    D = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{\textbf{Rank dependence of the paths and of the PIT values on synthetic forecasts.} Forecasts "
         r"and observations as in Table~\ref{stab:shufflecheck}, with Monte Carlo means over "
         rf"{rep[0]['n_rep']:,} replications. The true copula has a rank correlation of {pop[0.9]:.3f} for "
         rf"$\phi_0 = 0.9$ and {pop[0.97]:.3f} for $\phi_0 = 0.97$. Cov.\ 80: coverage (\%) of the central 80\% "
         r"interval from the PIT; Outside: share (\%) of outcomes below or above all 100 paths. Rand.: $\rho$ "
         r"observed from the randomized PIT, Eq.~(\ref{eq:pit}). Inside: $\rho$ observed over the windows whose "
         r"outcome lies inside the paths at both steps. Mid-PIT: $\rho$ observed from the mid-PIT "
         r"$(\#\text{below} + (\#\text{equal} + 1)/2)/101$ with tied ranks averaged. Monte Carlo standard errors "
         rf"are at most {max(rho_se, se['inside']):.4f} for the correlations and "
         rf"{100 * max(se['outside'], se['cov80']):.2f} percentage points for the shares.}}",
         r"\label{stab:shuffledep}", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}l r r r r r r r r r@{}}", r"\toprule",
         r" & & & & & & & \multicolumn{3}{c}{$\rho$ observed} \\", r"\cmidrule(l){8-10}",
         r"Forecaster & $\phi_0$ & $\phi$ & $s$ & Cov.\ 80 & Outside & $\rho$ model & Rand. & Inside & "
         r"Mid-PIT \\", r"\midrule"]
    for i, c in enumerate(rep):
        if i and c["phi0"] != rep[i - 1]["phi0"]:
            D.append(r"\addlinespace")
        D.append(f"{label(c, False)} & {c['phi0']:.2f} & {c['phi']:.2f} & {s_cell(c)} & {100 * c['cov80']:.1f} & "
                 f"{100 * c['outside']:.1f} & {c['rho_model']:.3f} & {c['rho_obs']:.3f} & {c['inside']:.3f} & "
                 f"{c['mid']:.3f} \\\\")
    D += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(SUPP, "shuffle_dep_table.tex"), "w") as f:
        f.write("\n".join(D) + "\n")


def main():
    path = os.path.join(HERE, "results", "shuffle_check_ext.json")
    if "--tables" in sys.argv[1:]:                  # only rewrite the tables from the stored results
        with open(path) as f:
            res = json.load(f)
        tables(res["cases"], res["replicated"])
        return
    rows = part1()
    rep = part2()
    tables(rows, rep)
    with open(path, "w") as f:
        json.dump(dict(cases=rows, replicated=rep), f, indent=1)


if __name__ == "__main__":
    main()
