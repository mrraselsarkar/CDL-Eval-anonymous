"""Aurora (token length 48) with context lengths from 96 to 2,880 steps on the test windows of run.py.

Usage:  python aurora_context.py      writes results/aurora_context.json
"""
import json
import os

import torch

import run


def main():
    from aurora import load_model
    run.LF = 2880                                        # longest context; shorter ones are its suffixes
    model = load_model().to(run.DEV).eval()
    out = []
    for ds in run.DATASETS:
        d = run.prepare(ds)
        for L in [96, 192, 336, 512, 1024, 2880]:
            torch.manual_seed(0)
            ctx = d["ctx"][:, -L:]
            with torch.no_grad():
                xs = torch.cat([model.generate(inputs=ctx[i:i + 25], max_output_length=run.H, num_samples=run.S,
                                               inference_token_len=48)[..., :run.H].float()
                                for i in range(0, ctx.shape[0], 25)])
            gen = torch.Generator(device=run.DEV).manual_seed(7)
            met, hist = run.score(xs, d, gen)
            h = torch.tensor(hist, dtype=torch.float64)
            met.update(dataset=ds, context=L, pit90=float(h[5:95].sum() / h.sum()))
            out.append(met)
            print(ds, L, {k: round(v, 3) for k, v in met.items() if isinstance(v, float)}, flush=True)
    with open(os.path.join(run.OUT, "aurora_context.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
