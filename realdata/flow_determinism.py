"""Where the randomness of the public flow models enters: rerun Sundial and Aurora on the same windows
with the same random seed and with a different seed. Identical output for the same seed shows that a
forecast is fixed by the initial noise, so in the time--noise plane every step is decided before the
first network call (Section 7.3 of the Review).

Usage:  python flow_determinism.py      prints the largest absolute differences
"""
import torch

import bench


@torch.no_grad()
def main():
    from aurora import load_model
    from transformers import AutoModelForCausalLM
    d = bench.prepare("heldout", 24, 300)["ercot"]
    ctx = d["ctx"][:25, -bench.LF:]
    aurora = load_model().to(bench.DEV).eval()
    sundial = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True).to(bench.DEV).eval()

    def run_aurora(seed):
        torch.manual_seed(seed)
        return aurora.generate(inputs=ctx, max_output_length=24, num_samples=20, inference_token_len=48)

    def run_sundial(seed):
        torch.manual_seed(seed)
        mu, sd = ctx.mean(1, keepdim=True), ctx.std(1, keepdim=True, unbiased=False).clamp_min(1e-2)
        return sundial(input_ids=(ctx - mu) / sd, revin=False, num_samples=20, max_output_length=24,
                       use_cache=False, return_dict=True).logits[..., :24]

    for name, f in [("Aurora", run_aurora), ("Sundial", run_sundial)]:
        a, b, c = f(0), f(0), f(1)
        print(f"{name}: same seed max |diff| = {float((a - b).abs().max()):.2e}; "
              f"different seed mean |diff| = {float((a - c).abs().mean()):.3f}; sample sd = {float(a.std(1).mean()):.3f}")


if __name__ == "__main__":
    main()
