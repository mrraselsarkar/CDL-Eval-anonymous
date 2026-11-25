"""Sundial through its released generation routine, to check the forward-pass call used elsewhere.

Runs in an environment with transformers 4.40.1, the version that Sundial's model card requires, and
scores the forecasts exactly as bench.py does, on the same test windows (held-out data, H = 24).

Usage:  python sundial_official.py      writes results/sundial_official.json
"""

import json
import os

import numpy as np
import torch

import bench

H, SEEDS = 24, [0, 1, 2]


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM
    bench.sim.ABAR = bench.sim.cosine_abar(bench.sim.CFG["k_train"])
    model = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True).to(bench.DEV).eval()
    rows = []
    for ds, d in bench.prepare("heldout", H, 300).items():
        ctx = d["ctx"][:, -bench.LF:]
        for seed in SEEDS:
            torch.manual_seed(seed)
            out = [model.generate(ctx[i:i + 25], max_new_tokens=H, num_samples=bench.S)[..., :H].float()
                   for i in range(0, ctx.shape[0], 25)]
            x = torch.cat(out)
            gen = torch.Generator(device=bench.DEV).manual_seed(7 + seed)
            win, _ = bench.window_scores("samples", x, None, d, gen)
            row = {k: float(v.mean()) for k, v in win.items() if k != "pit"}
            row.update(dataset=ds, seed=seed, transformers=__import__("transformers").__version__)
            rows.append(row)
            print(f"[sundial-official] {ds:8s} s{seed} crps={row['crps']:.3f} ql={row['ql']:.3f} "
                  f"cov80={row['cov80']:.2f} cov90={row['cov90']:.2f}", flush=True)
    with open(os.path.join(bench.OUT, "sundial_official.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
