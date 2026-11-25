"""Controlled simulation study: autoregressive vs diffusion forecasters.

Five forecasters that differ mainly in how they generate the forecast are
trained on three synthetic datasets and compared on accuracy, probabilistic
quality, calibration and cost.

  ar        autoregressive, Gaussian head, recursive sampling   (DeepAR-style)
  direct    direct multi-step, independent Gaussian per step    (direct baseline)
  ardiff    autoregressive with a diffusion head per step       (TimeGrad-style)
  fulldiff  diffusion over the whole forecast window            (CSDI-style)
  flow      flow matching over the whole forecast window        (Sundial/KiT-style loss)
  masked    masked discrete diffusion over value tokens         (LEAF-style)

Usage:
  python sim.py --exp smoke      quick check of the full pipeline
  python sim.py --exp main       3 datasets x 5 models x 3 seeds
  python sim.py --exp steps      sampling-step sweep (needs main)
  python sim.py --exp data       training-set-size sweep (needs main)
  python sim.py --exp all        main, steps and data in order
  python sim.py --exp all --models flow
                                 run only the listed forecasters and merge their
                                 rows into the existing results files
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L, H = 48, 24                      # context length, forecast horizon
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
MODEL_DIR = os.path.join(OUT, "models")
MODELS = ["ar", "direct", "ardiff", "fulldiff", "flow", "masked"]
DATASETS = ["seasonal", "bimodal", "lorenz"]

CFG = dict(n_train=16384, n_test=500, n_samples=50, train_steps=3000,
           batch=256, k_train=100, k_sample=50, rounds=24, seeds=[0, 1, 2],
           data_sizes=[256, 1024, 4096])


# =====================================================================
# Data
# =====================================================================

def seasonal_windows(rng, n, drift=False):
    """Two seasonal components, level, small trend and AR(1) noise.

    With drift=True the future gets a linear drift whose sign is +1 or -1
    with probability 1/2 and cannot be inferred from the context, so the
    true forecast distribution is bimodal.
    """
    T = L + H
    t = np.arange(T)[None, :]
    level = rng.normal(0, 1, (n, 1))
    a1 = rng.uniform(0.5, 2.0, (n, 1))
    a2 = rng.uniform(0.0, 1.0, (n, 1))
    p1 = rng.uniform(0, 2 * np.pi, (n, 1))
    p2 = rng.uniform(0, 2 * np.pi, (n, 1))
    trend = rng.normal(0, 0.005, (n, 1))
    sig = rng.uniform(0.1, 0.4, (n, 1))
    eps = rng.normal(0, 1, (n, T)) * sig
    noise = np.zeros((n, T))
    noise[:, 0] = eps[:, 0]
    for i in range(1, T):
        noise[:, i] = 0.7 * noise[:, i - 1] + eps[:, i]
    x = (level + trend * t + a1 * np.sin(2 * np.pi * t / 24 + p1)
         + a2 * np.sin(2 * np.pi * t / 12 + p2) + noise)
    if drift:
        sign = rng.choice([-1.0, 1.0], size=(n, 1))
        slope = rng.uniform(0.08, 0.2, (n, 1))
        steps = np.clip(t - (L - 1), 0, None)
        x = x + sign * slope * steps
    return x


def lorenz_windows(rng, n):
    """x-component of Lorenz-63, sampled every 0.05 time units, with noise."""
    dt, sub, burn = 0.01, 5, 1500
    s = rng.normal(0, 1, (n, 3)) + np.array([0.0, 0.0, 25.0])

    def f(u):
        x, y, z = u[:, 0], u[:, 1], u[:, 2]
        return np.stack([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z], 1)

    def rk4(u):
        k1 = f(u); k2 = f(u + dt / 2 * k1); k3 = f(u + dt / 2 * k2); k4 = f(u + dt * k3)
        return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    for _ in range(burn):
        s = rk4(s)
    out = np.zeros((n, L + H))
    for i in range((L + H) * sub):
        s = rk4(s)
        if i % sub == sub - 1:
            out[:, i // sub] = s[:, 0]
    return out + rng.normal(0, 0.5, out.shape)


def make_dataset(name):
    """Return global-z-scored train and test windows as float32 tensors."""
    gen = {"seasonal": lambda r, n: seasonal_windows(r, n, False),
           "bimodal": lambda r, n: seasonal_windows(r, n, True),
           "lorenz": lorenz_windows}[name]
    tr = gen(np.random.default_rng(100), CFG["n_train"])
    te = gen(np.random.default_rng(200), CFG["n_test"])
    mu, sd = tr.mean(), tr.std()
    tr, te = (tr - mu) / sd, (te - mu) / sd
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    return to(tr), to(te)


def inorm(ctx):
    mu = ctx.mean(1, keepdim=True)
    sd = ctx.std(1, keepdim=True).clamp_min(0.1)
    return mu, sd


# =====================================================================
# Building blocks
# =====================================================================

def mlp(i, h, o, depth):
    layers, d = [], i
    for _ in range(depth - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def temb(k, dim=32):
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=k.device) / half)
    a = k.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


def cosine_abar(K, s=0.008):
    t = torch.linspace(0, K, K + 1)
    f = torch.cos(((t / K) + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).clamp(1e-5, 1.0).to(DEV)          # abar[0] = 1


ABAR = None


def ddim_sample(eps_fn, shape, steps, eta=1.0):
    """DDIM sampler on a sub-sequence of the training noise levels.

    eta = 1 gives ancestral (DDPM-like) sampling; steps = K_train recovers DDPM.
    """
    K = CFG["k_train"]
    ts = np.linspace(K, 0, steps + 1).round().astype(int)
    y = torch.randn(shape, device=DEV)
    for t, tp in zip(ts[:-1], ts[1:]):
        a_t, a_p = ABAR[t], ABAR[tp]
        kk = torch.full((shape[0],), int(t), device=DEV)
        e = eps_fn(y, kk)
        x0 = ((y - (1 - a_t).sqrt() * e) / a_t.sqrt()).clamp(-12, 12)
        sig = eta * ((1 - a_p) / (1 - a_t)).sqrt() * (1 - a_t / a_p).sqrt()
        y = a_p.sqrt() * x0 + (1 - a_p - sig ** 2).clamp_min(0).sqrt() * e
        if tp > 0:
            y = y + sig * torch.randn_like(y)
    return y


def diffusion_loss(eps_fn, y0):
    K = CFG["k_train"]
    k = torch.randint(1, K + 1, (y0.shape[0],), device=DEV)
    e = torch.randn_like(y0)
    a = ABAR[k].view(-1, *([1] * (y0.dim() - 1)))
    yk = a.sqrt() * y0 + (1 - a).sqrt() * e
    return F.mse_loss(eps_fn(yk, k), e)


def random_step_pairs(zn):
    """Pick one forecast step per window: (context of length L, next value)."""
    B = zn.shape[0]
    h = torch.randint(0, H, (B,), device=DEV)
    idx = h[:, None] + torch.arange(L, device=DEV)[None]
    ctx = zn.gather(1, idx)
    nxt = zn.gather(1, (h + L)[:, None])
    return ctx, nxt


def normalize_window(w):
    mu, sd = inorm(w[:, :L])
    return (w - mu) / sd


# =====================================================================
# Models
# =====================================================================

class ARGauss(nn.Module):
    calls = lambda self, **kw: H

    def __init__(self):
        super().__init__()
        self.net = mlp(L, 256, 2, 4)

    def head(self, ctx):
        o = self.net(ctx)
        return o[:, :1], o[:, 1:].clamp(-7, 3)

    def loss(self, w):
        ctx, nxt = random_step_pairs(normalize_window(w))
        mu, ls = self.head(ctx)
        return (0.5 * ((nxt - mu) / ls.exp()) ** 2 + ls).mean()

    @torch.no_grad()
    def sample(self, ctx_n, S, **kw):
        buf = ctx_n.repeat_interleave(S, 0)
        out = []
        for _ in range(H):
            mu, ls = self.head(buf)
            y = mu + ls.exp() * torch.randn_like(mu)
            out.append(y)
            buf = torch.cat([buf[:, 1:], y], 1)
        return torch.cat(out, 1).view(-1, S, H)


class DirectGauss(nn.Module):
    calls = lambda self, **kw: 1

    def __init__(self):
        super().__init__()
        self.net = mlp(L, 256, 2 * H, 4)

    def head(self, ctx):
        o = self.net(ctx)
        return o[:, :H], o[:, H:].clamp(-7, 3)

    def loss(self, w):
        zn = normalize_window(w)
        mu, ls = self.head(zn[:, :L])
        return (0.5 * ((zn[:, L:] - mu) / ls.exp()) ** 2 + ls).mean()

    @torch.no_grad()
    def sample(self, ctx_n, S, **kw):
        mu, ls = self.head(ctx_n)
        z = torch.randn(ctx_n.shape[0], S, H, device=DEV)
        return mu[:, None] + ls.exp()[:, None] * z


class ARDiff(nn.Module):
    """Autoregressive backbone, small diffusion model for each next value."""
    calls = lambda self, steps=None, **kw: H * (steps or CFG["k_sample"])

    def __init__(self):
        super().__init__()
        self.enc = mlp(L, 256, 128, 3)
        self.den = mlp(1 + 32 + 128, 256, 1, 4)

    def eps(self, yk, k, c):
        return self.den(torch.cat([yk, temb(k), c], 1))

    def loss(self, w):
        ctx, nxt = random_step_pairs(normalize_window(w))
        c = self.enc(ctx)
        return diffusion_loss(lambda yk, k: self.eps(yk, k, c), nxt)

    @torch.no_grad()
    def sample(self, ctx_n, S, steps=None, **kw):
        steps = steps or CFG["k_sample"]
        buf = ctx_n.repeat_interleave(S, 0)
        out = []
        for _ in range(H):
            c = self.enc(buf)
            y = ddim_sample(lambda yk, k: self.eps(yk, k, c), (buf.shape[0], 1), steps)
            out.append(y)
            buf = torch.cat([buf[:, 1:], y], 1)
        return torch.cat(out, 1).view(-1, S, H)


class FullDiff(nn.Module):
    """Diffusion over the whole forecast window, conditioned on the context."""
    calls = lambda self, steps=None, **kw: steps or CFG["k_sample"]

    def __init__(self):
        super().__init__()
        self.enc = mlp(L, 256, 256, 3)
        self.den = mlp(H + 32 + 256, 512, H, 4)

    def eps(self, yk, k, c):
        return self.den(torch.cat([yk, temb(k), c], 1))

    def loss(self, w):
        zn = normalize_window(w)
        c = self.enc(zn[:, :L])
        return diffusion_loss(lambda yk, k: self.eps(yk, k, c), zn[:, L:])

    @torch.no_grad()
    def sample(self, ctx_n, S, steps=None, **kw):
        steps = steps or CFG["k_sample"]
        c = self.enc(ctx_n).repeat_interleave(S, 0)
        y = ddim_sample(lambda yk, k: self.eps(yk, k, c), (c.shape[0], H), steps)
        return y.view(-1, S, H)


class FlowFull(nn.Module):
    """Flow matching over the whole forecast window, with the networks of FullDiff.

    Straight paths y_t = (1 - t) z + t y from noise z to data y; the network
    regresses the velocity y - z and sampling integrates it with Euler steps.
    """
    calls = lambda self, steps=None, **kw: steps or CFG["k_sample"]

    def __init__(self):
        super().__init__()
        self.enc = mlp(L, 256, 256, 3)
        self.vel = mlp(H + 32 + 256, 512, H, 4)

    def v(self, yt, t, c):
        return self.vel(torch.cat([yt, temb(t * CFG["k_train"]), c], 1))

    def loss(self, w):
        zn = normalize_window(w)
        c = self.enc(zn[:, :L])
        y1 = zn[:, L:]
        y0 = torch.randn_like(y1)
        t = torch.rand(y1.shape[0], device=DEV)
        yt = (1 - t[:, None]) * y0 + t[:, None] * y1
        return F.mse_loss(self.v(yt, t, c), y1 - y0)

    @torch.no_grad()
    def sample(self, ctx_n, S, steps=None, **kw):
        steps = steps or CFG["k_sample"]
        c = self.enc(ctx_n).repeat_interleave(S, 0)
        y = torch.randn(c.shape[0], H, device=DEV)
        for i in range(steps):
            t = torch.full((y.shape[0],), i / steps, device=DEV)
            y = y + self.v(y, t, c) / steps
        return y.view(-1, S, H)


NB, LO, HI = 128, -8.0, 8.0          # value bins for the token model
MASK = NB


def tokenize(z):
    return ((z - LO) / (HI - LO) * NB).floor().clamp(0, NB - 1).long()


def detokenize(idx):
    u = torch.rand_like(idx, dtype=torch.float32)            # uniform dequantization
    return LO + (idx.float() + u) * (HI - LO) / NB


class MaskedDiff(nn.Module):
    """Masked discrete diffusion over value tokens with the same MLP backbone as the others.

    The network sees the continuous context and the (partly masked) horizon tokens
    and returns a categorical distribution over NB bins for every horizon position.
    Sampling unmasks tokens over a number of rounds, either in order of confidence
    (as in LLaDA and LEAF) or in random order.
    """
    calls = lambda self, rounds=None, **kw: rounds or CFG["rounds"]

    def __init__(self, e=16, h=256):
        super().__init__()
        self.tok = nn.Embedding(NB + 1, e)
        self.net = mlp(L + H * e, h, H * NB, 4)

    def logits(self, ctx_n, htok):
        x = torch.cat([ctx_n, self.tok(htok).flatten(1)], 1)
        return self.net(x).view(-1, H, NB)

    def loss(self, w):
        zn = normalize_window(w)
        tgt = tokenize(zn[:, L:])
        B = tgt.shape[0]
        s = torch.rand(B, 1, device=DEV) * 0.95 + 0.05
        mask = torch.rand(B, H, device=DEV) < s
        none = ~mask.any(1)
        mask[none, torch.randint(0, H, (int(none.sum()),), device=DEV)] = True
        inp = tgt.masked_fill(mask, MASK)
        ce = F.cross_entropy(self.logits(zn[:, :L], inp).reshape(-1, NB), tgt.reshape(-1),
                             reduction="none").view(B, H)
        return ((ce * mask).sum(1) / (s[:, 0] * H)).mean()

    @torch.no_grad()
    def sample(self, ctx_n, S, rounds=None, order="confidence", **kw):
        R = rounds or CFG["rounds"]
        ctx = ctx_n.repeat_interleave(S, 0)
        tok = torch.full((ctx.shape[0], H), MASK, device=DEV)
        revealed = 0
        for r in range(R):
            target = round(H * (r + 1) / R)
            n_new = target - revealed
            if n_new <= 0:
                continue
            probs = F.softmax(self.logits(ctx, tok), -1)
            samp = torch.multinomial(probs.reshape(-1, NB), 1).view(-1, H)
            if order == "confidence":
                score = probs.gather(2, samp[..., None])[..., 0]
            else:
                score = torch.rand(tok.shape, device=DEV)
            score = score.masked_fill(tok != MASK, -1.0)
            pick = score.topk(n_new, dim=1).indices
            tok.scatter_(1, pick, samp.gather(1, pick))
            revealed = target
        return detokenize(tok).view(-1, S, H)


CLASSES = {"ar": ARGauss, "direct": DirectGauss, "ardiff": ARDiff,
           "fulldiff": FullDiff, "flow": FlowFull, "masked": MaskedDiff}
LR = {}


# =====================================================================
# Training and evaluation
# =====================================================================

def train(name, data, seed, steps):
    torch.manual_seed(seed)
    model = CLASSES[name]().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR.get(name, 1e-3), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n = data.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (CFG["batch"],), device=DEV)
        loss = model.loss(data[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
    model.eval()
    return model


def crps_samples(x, y):
    """x: (N, S, H) samples, y: (N, H). Returns CRPS per (N, H)."""
    S = x.shape[1]
    t1 = (x - y[:, None]).abs().mean(1)
    xs = x.sort(1).values
    w = (2 * torch.arange(1, S + 1, device=x.device) - S - 1).float()
    t2 = 2.0 / (S * S) * (xs * w[None, :, None]).sum(1)
    return t1 - 0.5 * t2


def energy_score(x, y):
    t1 = (x - y[:, None]).norm(dim=2).mean(1)
    t2 = torch.cdist(x, x).mean((1, 2))
    return t1 - 0.5 * t2


@torch.no_grad()
def evaluate(model, test, S, **kw):
    ctx, y = test[:, :L], test[:, L:]
    mu, sd = inorm(ctx)
    ctx_n = (ctx - mu) / sd
    torch.manual_seed(1234)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    xs = model.sample(ctx_n, S, **kw)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    xs = xs * sd[:, None] + mu[:, None]
    mean = xs.mean(1)
    q = torch.quantile(xs, torch.tensor([0.05, 0.25, 0.75, 0.95], device=DEV), dim=1)
    crps = crps_samples(xs, y)
    return dict(
        mse=float(((mean - y) ** 2).mean()),
        mse_h=((mean - y) ** 2).mean(0).tolist(),
        crps=float(crps.mean()),
        crps_h=crps.mean(0).tolist(),
        es=float(energy_score(xs, y).mean()),
        cov50=float(((y >= q[1]) & (y <= q[2])).float().mean()),
        cov90=float(((y >= q[0]) & (y <= q[3])).float().mean()),
        time_s=elapsed,
        calls=int(model.calls(**kw)),
    ), xs


def naive_mse(name, test):
    ctx, y = test[:, :L], test[:, L:]
    pred = ctx[:, -1:].expand(-1, H) if name == "lorenz" else ctx[:, L - 24:L - 24 + H]
    return float(((pred - y) ** 2).mean())


def model_path(ds, name, seed, n_train):
    return os.path.join(MODEL_DIR, f"{ds}_{name}_s{seed}_n{n_train}.pt")


def get_model(ds, name, seed, data, n_train, steps):
    path = model_path(ds, name, seed, n_train)
    if os.path.exists(path):
        m = CLASSES[name]().to(DEV)
        m.load_state_dict(torch.load(path, map_location=DEV))
        m.eval()
        return m, 0.0
    t0 = time.perf_counter()
    m = train(name, data[:n_train], seed, steps)
    torch.save(m.state_dict(), path)
    return m, time.perf_counter() - t0


def save_json(obj, fname):
    with open(os.path.join(OUT, fname), "w") as f:
        json.dump(obj, f, indent=1)


def load_json(fname, default):
    path = os.path.join(OUT, fname)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def merge_rows(fname, rows, models):
    """Replace the rows of `models` in a results file and keep all other rows."""
    drop = set(models) | ({"masked_rand"} if "masked" in models else set())
    keep = [r for r in load_json(fname, []) if r["model"] not in drop]
    save_json(keep + rows, fname)


# =====================================================================
# Experiments
# =====================================================================

def exp_main(models):
    results, examples = [], load_json("examples.json", {})
    for ds in DATASETS:
        train_w, test_w = make_dataset(ds)
        base = naive_mse(ds, test_w)
        for seed in CFG["seeds"]:
            for name in models:
                m, ttrain = get_model(ds, name, seed, train_w, CFG["n_train"], CFG["train_steps"])
                met, xs = evaluate(m, test_w, CFG["n_samples"])
                met.update(dataset=ds, model=name, seed=seed, train_s=ttrain,
                           params=sum(p.numel() for p in m.parameters()), naive_mse=base)
                results.append(met)
                print(f"[main] {ds:9s} {name:8s} seed={seed} mse={met['mse']:.3f} "
                      f"crps={met['crps']:.3f} es={met['es']:.3f} cov90={met['cov90']:.2f} "
                      f"t={met['time_s']:.2f}s train={ttrain:.0f}s", flush=True)
                if seed == 0:
                    ids = [0, 1, 2, 3, 4, 5]
                    examples.setdefault(ds, {"context": test_w[ids, :L].tolist(),
                                             "truth": test_w[ids, L:].tolist()})
                    examples[ds][name] = xs[ids, :40].tolist()
                if name == "masked":
                    # ablation: same trained model, tokens unmasked in random order
                    met2, _ = evaluate(m, test_w, CFG["n_samples"], order="random")
                    met2.update(dataset=ds, model="masked_rand", seed=seed, train_s=0.0,
                                params=met["params"], naive_mse=base)
                    results.append(met2)
                    print(f"[main] {ds:9s} masked (random order) seed={seed} mse={met2['mse']:.3f} "
                          f"crps={met2['crps']:.3f} es={met2['es']:.3f} cov90={met2['cov90']:.2f}", flush=True)
                del m
                torch.cuda.empty_cache()
        merge_rows("main.json", results, models)
        save_json(examples, "examples.json")


SWEEP = {"ardiff": [{"steps": v} for v in [5, 10, 20, 50]],
         "fulldiff": [{"steps": v} for v in [5, 10, 20, 50, 100]],
         "flow": [{"steps": v} for v in [2, 5, 10, 20, 50]],
         "masked": [{"rounds": v} for v in [1, 2, 4, 8, 24]]
                   + [{"rounds": v, "order": "random"} for v in [4, 24]],
         "ar": [{}], "direct": [{}]}


def exp_steps(models):
    results = []
    for ds in ["seasonal", "lorenz"]:
        train_w, test_w = make_dataset(ds)
        for seed in CFG["seeds"]:
            for name in models:
                m, _ = get_model(ds, name, seed, train_w, CFG["n_train"], CFG["train_steps"])
                for kw in SWEEP[name]:
                    v = kw.get("steps", kw.get("rounds"))
                    met, _ = evaluate(m, test_w, CFG["n_samples"], **kw)
                    met.update(dataset=ds, model=name, seed=seed, setting=v,
                               order=kw.get("order", "confidence"))
                    for k in ("mse_h", "crps_h"):
                        met.pop(k)
                    results.append(met)
                    print(f"[steps] {ds} {name:8s} seed={seed} setting={v} calls={met['calls']} "
                          f"crps={met['crps']:.3f} es={met['es']:.3f} t={met['time_s']:.2f}s", flush=True)
                del m
                torch.cuda.empty_cache()
        merge_rows("steps.json", results, models)


def exp_data(models):
    results = []
    ds = "lorenz"
    train_w, test_w = make_dataset(ds)
    for n in CFG["data_sizes"]:        # the full-size point (16384) comes from main.json
        for seed in CFG["seeds"]:
            for name in models:
                m, _ = get_model(ds, name, seed, train_w, n, CFG["train_steps"])
                met, _ = evaluate(m, test_w, CFG["n_samples"])
                met.update(dataset=ds, model=name, seed=seed, n_train=n)
                for k in ("mse_h", "crps_h"):
                    met.pop(k)
                results.append(met)
                print(f"[data] n={n:5d} {name:8s} seed={seed} crps={met['crps']:.3f} "
                      f"es={met['es']:.3f} mse={met['mse']:.3f}", flush=True)
                del m
                torch.cuda.empty_cache()
        merge_rows("data.json", results, models)


def main():
    global ABAR, OUT, MODEL_DIR, DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["smoke", "main", "steps", "data", "all"])
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS,
                    help="forecasters to run; their rows replace those in existing results files")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.device:
        DEV = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.exp == "smoke":
        CFG.update(n_train=1024, n_test=64, n_samples=16, train_steps=100, seeds=[0])
        OUT = os.path.join(HERE, "results_smoke")
        MODEL_DIR = os.path.join(OUT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    ABAR = cosine_abar(CFG["k_train"])
    models = [m for m in MODELS if m in args.models]
    print("device:", DEV, "models:", models, "config:", CFG, flush=True)
    t0 = time.perf_counter()
    if args.exp in ("smoke", "main", "all"):
        exp_main(models)
    if args.exp in ("smoke", "steps", "all"):
        exp_steps(models)
    if args.exp in ("smoke", "data", "all"):
        exp_data(models)
    save_json(CFG, "config.json")
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
