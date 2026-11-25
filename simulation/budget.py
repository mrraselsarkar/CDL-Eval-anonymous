"""Learning curves: accuracy, probabilistic quality and coverage against the training budget.

Every forecaster of sim.py is trained for 1,000, 3,000, 9,000 and 30,000 steps, each with a
cosine learning-rate schedule over its own budget, on the regime-switch and Lorenz data.
Runs that already exist are not repeated: the 3,000-step runs are those of results/main.json,
and the 9,000- and 30,000-step full-window runs are those of results_robust/robust.json.
budget_report.py merges all sources.

Usage:  python budget.py --datasets bimodal          (one worker per dataset; resumable)
        python budget.py --datasets lorenz --threads 4
"""
import argparse
import json
import os
import time

import torch

import sim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_budget")
MODEL_DIR = os.path.join(OUT, "models")
MODELS = ["ar", "direct", "ardiff", "fulldiff", "flow", "masked"]
EXISTING = {("fulldiff", 9000), ("fulldiff", 30000), ("flow", 30000)}   # in results_robust
TODO = [(m, b) for b in (1000, 9000, 30000) for m in MODELS if (m, b) not in EXISTING]


def out_file(ds):
    return os.path.join(OUT, f"budget_{ds}.json")


def load(ds):
    path = out_file(ds)
    return json.load(open(path)) if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["bimodal", "lorenz"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    sim.DEV = torch.device(args.device)
    sim.ABAR = sim.cosine_abar(sim.CFG["k_train"])
    os.makedirs(MODEL_DIR, exist_ok=True)
    for ds in args.datasets:
        rows = load(ds)
        done = {(r["model"], r["train_steps"], r["seed"]) for r in rows}
        train_w, test_w = sim.make_dataset(ds)
        for name, steps in TODO:
            for seed in sim.CFG["seeds"]:
                if (name, steps, seed) in done:
                    continue
                path = os.path.join(MODEL_DIR, f"{ds}_{name}_s{seed}_t{steps}.pt")
                t0 = time.perf_counter()
                if os.path.exists(path):
                    m = sim.CLASSES[name]().to(sim.DEV)
                    m.load_state_dict(torch.load(path, map_location=sim.DEV))
                    m.eval()
                else:
                    m = sim.train(name, train_w, seed, steps)
                    torch.save(m.state_dict(), path)
                tt = time.perf_counter() - t0
                evals = [(name, {})] + ([("masked_rand", {"order": "random"})] if name == "masked" else [])
                for label, kw in evals:
                    met, _ = sim.evaluate(m, test_w, sim.CFG["n_samples"], **kw)
                    for k in ("mse_h", "crps_h"):
                        met.pop(k)
                    met.update(dataset=ds, model=label, seed=seed, train_steps=steps, train_s=tt,
                               device=args.device)
                    rows.append(met)
                    print(f"[budget] {ds} {label:11s} steps={steps:5d} seed={seed} crps={met['crps']:.3f} "
                          f"es={met['es']:.3f} cov90={met['cov90']:.3f} train={tt:.0f}s", flush=True)
                json.dump(rows, open(out_file(ds), "w"), indent=1)
    print("BUDGET_DONE", flush=True)


if __name__ == "__main__":
    main()
