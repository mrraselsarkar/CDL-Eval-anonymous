"""Runs the experiments of the paper, stage by stage, with the exact command lines used for the stored results.

  python scripts/reproduce.py --list                      stages and their jobs
  python scripts/reproduce.py tables                      all tables and figures from the stored results (CPU)
  python scripts/reproduce.py benchmark --dry-run         print the commands of a stage without running them
  python scripts/reproduce.py benchmark --python-toto .venv-toto/bin/python

Stages, in the order of a full reproduction:
  data         download the datasets and run the data checks (CPU, network)
  benchmark    foundation models and seasonal naive baseline at 24, 192 and 720 steps (GPU; main and Toto environments)
  first-round  small forecasters trained on each dataset, context-length and decoding checks (GPU)
  windows      calendar weeks of the test windows, for the cluster bootstrap (CPU; after benchmark)
  revision     experiments of Section S2 with full forecasts stored: new datasets, flow sampling steps and samplers,
               spread correction, copulas, chunked rollout, longer contexts and horizons (GPU, then CPU)
  baselines    exponential smoothing, DeepAR, Lag-Llama, small forecasters with 512 context steps (baselines environment)
  checks       synthetic calibration of the dependence measures, Gaussian check of the flow samplers, determinism of the
               flow heads, Sundial's released generation routine (CPU and GPU; Sundial environment for the last)
  simulation   controlled simulation of Section S3 (GPU or CPU)
  tables       every table and figure of the paper into outputs/ (CPU)
  smoke        a few-minute check of the pipeline on tiny settings

Every job runs in the folder of its script and writes its output to logs/<stage>/<number>_<name>.log as well as to the
console. Jobs on the GPU should not share it with other processes; the stages run their jobs one after the other.
Interpreters: --python (default: the one running this script), --python-toto, --python-extra and --python-sundial, or
the environment variables CDL_PYTHON, CDL_PYTHON_TOTO, CDL_PYTHON_EXTRA and CDL_PYTHON_SUNDIAL.
"""

import argparse
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV2 = os.path.join(ROOT, "realdata", "results", "rev2")

H24 = ["--horizon", "24"]
LONG = ["--n_test", "150", "--seeds", "0"]
FAST = ["naive", "bolt", "bolt_base", "chronos2", "timesfm", "sundial", "aurora"]
NEW_DATA = ["aemo_mid", "ausel_season", "ercot2026"]
ALL7 = ["ercot", "traffic", "ausel", "aemo", "ausel_season", "aemo_mid", "ercot2026"]
HELD = ["ercot", "traffic", "ausel"]


def bench(tag, *args):
    """bench.py job with its result tag (results/bench_<tag>.json)."""
    return ["bench.py", *args, "--tag", tag]


def rev2(tag, *args):
    """bench.py job that also stores the full forecasts (results/rev2/)."""
    return ["bench.py", *args, "--store", "rev2", "--tag", tag]


def extra(*args):
    return ["bench_extra.py", *args, "--skip_existing"]


# Each job: (name, environment, folder, arguments, device). Environments: main, toto, extra, sundial.
STAGES = {
    "data": [
        ("download", "main", "scripts", ["download_data.py"], "cpu"),
        ("data_checks", "main", "realdata", ["fresh_data.py", "check"], "cpu"),
    ],
    "benchmark": [
        ("heldout_24", "main", "realdata", bench("main", "--data", "heldout", *H24, "--models", *FAST, "chronos"), "gpu"),
        ("aemo_24", "main", "realdata", bench("main", "--data", "aemo", *H24, "--models", *FAST), "gpu"),
        ("aemo_24_chronos", "main", "realdata",
         bench("main", "--data", "aemo", *H24, "--seeds", "0", "--models", "chronos"), "gpu"),
        ("heldout_192", "main", "realdata", bench("main", "--data", "heldout", "--horizon", "192", *LONG, "--models",
                                                  "naive", "bolt", "chronos2", "timesfm", "sundial", "aurora", "chronos"),
         "gpu"),
        ("heldout_720", "main", "realdata", bench("long", "--data", "heldout", "--horizon", "720", *LONG, "--models",
                                                  "naive", "bolt", "chronos2", "timesfm", "sundial", "aurora"), "gpu"),
        ("heldout_24_chronos_base", "main", "realdata",
         bench("base", "--data", "heldout", *H24, "--seeds", "0", "--models", "chronos_base"), "gpu"),
        # a re-run of three models; bench_report.py reads the result files in name order, so these rows replace
        # the rows of the same models in bench_main.json
        ("heldout_24_rerun", "main", "realdata",
         bench("main_fix", "--data", "heldout", *H24, "--models", "naive", "sundial", "aurora"), "gpu"),
        ("toto_heldout_24", "toto", "realdata", bench("toto", "--data", "heldout", *H24, "--models", "toto"), "gpu"),
        ("toto_aemo_24", "toto", "realdata", bench("toto", "--data", "aemo", *H24, "--models", "toto"), "gpu"),
        ("toto_heldout_192", "toto", "realdata",
         bench("toto", "--data", "heldout", "--horizon", "192", *LONG, "--models", "toto"), "gpu"),
        ("toto_heldout_720", "toto", "realdata",
         bench("totolong", "--data", "heldout", "--horizon", "720", *LONG, "--models", "toto"), "gpu"),
    ],
    "first-round": [
        ("small_forecasters", "main", "realdata", ["run.py"], "gpu"),
        ("fm_quantile_paths", "main", "realdata", ["run_fm.py", "aurora", "timesfm", "chronos2"], "gpu"),
        ("fm_quantile_paths_toto", "toto", "realdata", ["run_fm.py", "toto"], "gpu"),
        ("robustness", "main", "realdata", ["robust.py"], "gpu"),
        ("aurora_context", "main", "realdata", ["aurora_context.py"], "gpu"),
    ],
    "windows": [
        ("calendar_weeks", "main", "realdata", ["cluster_stats.py"], "cpu"),
    ],
    "revision": [
        *[(f"rerun_{d}_fast", "main", "realdata", rev2(f"rev2_{d}_fast", "--data", d, *H24, "--models", *FAST), "gpu")
          for d in ["heldout", "aemo"]],
        *[(f"rerun_{d}_toto", "toto", "realdata", rev2(f"rev2_{d}_toto", "--data", d, *H24, "--models", "toto"), "gpu")
          for d in ["heldout", "aemo"]],
        ("rerun_heldout_chronos", "main", "realdata",
         rev2("rev2_heldout_chronos", "--data", "heldout", *H24, "--models", "chronos"), "gpu"),
        ("rerun_aemo_chronos", "main", "realdata",
         rev2("rev2_aemo_chronos", "--data", "aemo", *H24, "--seeds", "0", "--models", "chronos"), "gpu"),
        *[(f"rerun_{d}_chronos_base", "main", "realdata",
           rev2(f"rev2_{d}_chronos_base", "--data", d, *H24, "--seeds", "0", "--models", "chronos_base"), "gpu")
          for d in ["heldout", "aemo"]],
        *[job for d in NEW_DATA for job in [
            (f"{d}_fast", "main", "realdata", rev2(f"rev2_{d}_fast", "--data", d, *H24, "--models", *FAST), "gpu"),
            (f"{d}_toto", "toto", "realdata", rev2(f"rev2_{d}_toto", "--data", d, *H24, "--models", "toto"), "gpu"),
            (f"{d}_chronos", "main", "realdata",
             rev2(f"rev2_{d}_chronos", "--data", d, *H24, "--seeds", "0", "--models", "chronos"), "gpu")]],
        *[job for d in ["heldout", "aemo"] for job in [
            *[(f"euler{k}_{d}", "main", "realdata",
               rev2(f"rev2_euler{k}_{d}", "--data", d, *H24, "--seeds", "0", "--models", "sundial", "aurora",
                    "--flow_steps", k), "gpu") for k in ["5", "10", "25", "100", "200"]],
            *[(f"{s}50_{d}", "main", "realdata",
               rev2(f"rev2_{s}50_{d}", "--data", d, *H24, "--seeds", "0", "--models", "sundial", "aurora",
                    "--flow_sampler", s), "gpu") for s in ["sde", "ode"]]]],
        ("calib_fast", "main", "realdata",
         rev2("rev2_calib_fast", "--data", "heldout_calib", *H24, *LONG, "--models", "naive", "sundial", "aurora"), "gpu"),
        ("calib_toto", "toto", "realdata",
         rev2("rev2_calib_toto", "--data", "heldout_calib", *H24, *LONG, "--models", "toto"), "gpu"),
        ("calib_chronos", "main", "realdata",
         rev2("rev2_calib_chronos", "--data", "heldout_calib", *H24, *LONG, "--models", "chronos"), "gpu"),
        ("h720_chronos2", "main", "realdata",
         rev2("rev2_h720_chronos2", "--data", "heldout", "--horizon", "720", *LONG, "--models", "chronos2"), "gpu"),
        ("chunk64_median", "main", "realdata", ["revision_runs.py", "chunked", "--chunk", "64", "--feedback", "median"],
         "gpu"),
        ("chunk24_median", "main", "realdata", ["revision_runs.py", "chunked", "--chunk", "24", "--feedback", "median"],
         "gpu"),
        ("chunk64_paths", "main", "realdata", ["revision_runs.py", "chunked", "--chunk", "64", "--feedback", "paths"],
         "gpu"),
        *[(f"ctx{c}_{d}", "main", "realdata",
           rev2(f"rev2_ctx{c}_{d}", "--data", d, *H24, "--models", "timesfm", "chronos2", "--context", c), "gpu")
          for c in ["2048", "4096"] for d in ["heldout", "aemo"]],
        *[job for h in ["192", "720"] for job in [
            (f"h{h}_fast", "main", "realdata",
             rev2(f"rev2_h{h}_fast", "--data", "heldout", "--horizon", h, *LONG, "--models", "naive", "bolt", "timesfm",
                  "sundial", "aurora"), "gpu"),
            (f"h{h}_toto", "toto", "realdata",
             rev2(f"rev2_h{h}_toto", "--data", "heldout", "--horizon", h, *LONG, "--models", "toto"), "gpu")]],
        ("h192_chronos2", "main", "realdata",
         rev2("rev2_h192_chronos2", "--data", "heldout", "--horizon", "192", *LONG, "--models", "chronos2"), "gpu"),
        ("copula", "main", "realdata", ["revision_runs.py", "copula"], "cpu"),
        ("spread", "main", "realdata", ["revision_runs.py", "spread"], "cpu"),
    ],
    "baselines": [
        ("ets_24", "extra", "realdata", extra("--cpu", "--models", "ets", *H24, "--data", *ALL7), "cpu"),
        *[(f"ets_{h}", "extra", "realdata", extra("--cpu", "--models", "ets", "--horizon", h, *LONG, "--data", *HELD),
           "cpu") for h in ["192", "720"]],
        ("small512_24", "extra", "realdata",
         extra("--require_gpu", "--models", "small512_ar", "small512_flow", *H24, "--data", *ALL7), "gpu"),
        ("deepar_24", "extra", "realdata", extra("--require_gpu", "--models", "deepar", *H24, "--data", *ALL7), "gpu"),
        ("lagllama_24", "extra", "realdata",
         extra("--require_gpu", "--models", "lagllama", "lagllama_ctx512", *H24, "--data", *ALL7), "gpu"),
        *[(f"deepar_{h}", "extra", "realdata",
           extra("--require_gpu", "--models", "deepar", "--horizon", h, *LONG, "--data", *HELD), "gpu")
          for h in ["192", "720"]],
        ("summary", "extra", "realdata", ["bench_extra.py", "--report"], "cpu"),
    ],
    "checks": [
        ("shuffle_test_calibration", "main", "realdata", ["shuffle_check.py"], "cpu"),
        ("gaussian_flow_sampler", "main", "realdata", ["sampler_gauss.py"], "cpu"),
        ("flow_determinism", "main", "realdata", ["flow_determinism.py"], "gpu"),
        ("sundial_generate", "sundial", "realdata", ["sundial_official.py"], "gpu"),
    ],
    "simulation": [
        ("main", "main", "simulation", ["sim.py", "--exp", "main"], "gpu"),
        ("steps", "main", "simulation", ["sim.py", "--exp", "steps"], "gpu"),
        ("data", "main", "simulation", ["sim.py", "--exp", "data"], "gpu"),
        ("calibration", "main", "simulation", ["calib.py"], "gpu"),
        ("longer_training", "main", "simulation", ["sim_robust.py"], "gpu"),
        ("budget_bimodal", "main", "simulation", ["budget.py", "--datasets", "bimodal"], "cpu"),
        ("budget_lorenz", "main", "simulation", ["budget.py", "--datasets", "lorenz"], "cpu"),
    ],
    "tables": [
        ("benchmark", "main", "realdata", ["bench_report.py"], "cpu"),
        ("ranks_and_tests", "main", "realdata", ["rank_stats.py"], "cpu"),
        ("first_round", "main", "realdata", ["robust_report.py"], "cpu"),
        ("shuffle_test_calibration", "main", "realdata", ["shuffle_check.py", "--tables"], "cpu"),
        ("gaussian_flow_sampler", "main", "realdata", ["sampler_gauss.py"], "cpu"),
        ("revision", "main", "realdata", ["revision_report.py"], "cpu+forecasts"),
        ("baselines", "main", "realdata", ["extra_report.py"], "cpu+forecasts"),
        ("simulation", "main", "simulation", ["plots.py"], "cpu"),
    ],
    "smoke": [
        ("simulation", "main", "simulation", ["sim.py", "--exp", "smoke"], "any"),
        ("benchmark", "main", "realdata", ["bench.py", "--data", "heldout", *H24, "--models", "naive", "bolt",
                                           "--seeds", "0", "--max_windows", "4", "--paths", "8",
                                           "--outdir", os.path.join(ROOT, "logs", "smoke", "results"),
                                           "--tag", "smoke"], "any"),
    ],
}
ORDER = ["data", "benchmark", "first-round", "windows", "revision", "baselines", "checks", "simulation", "tables"]


def have_forecasts():
    """True if the full forecasts of results/rev2 (data archive or the revision stage) are present."""
    return bool(glob.glob(os.path.join(REV2, "*__H*__*.npz")))


def run_job(stage, i, job, exe, dry):
    name, env, folder, args, device = job
    cwd = os.path.join(ROOT, folder)
    cmd = [exe, *args]
    shown = f"(cd {folder} && {' '.join(['python' if env == 'main' else f'<{env} python>', *args])})"
    if device == "cpu+forecasts" and not have_forecasts():
        print(f"[{stage} {i:02d}] skip {name}: needs the full forecasts in realdata/results/rev2 (see README, Data)")
        return True
    print(f"[{stage} {i:02d}] {name}: {shown}", flush=True)
    if dry:
        return True
    env_vars = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if device.startswith("cpu"):
        env_vars["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU jobs never touch the GPU ("-1": an empty value is dropped)
    log_dir = os.path.join(ROOT, "logs", stage)
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(log_dir, f"{i:02d}_{name}.log"), "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=cwd, env=env_vars, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = p.wait()
    status = "ok" if code == 0 else f"FAILED (exit {code})"
    print(f"[{stage} {i:02d}] {name}: {status} after {(time.time() - t0) / 60:.1f} min", flush=True)
    return code == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", help=f"stages to run ({', '.join(ORDER)}, smoke, or all)")
    ap.add_argument("--list", action="store_true", help="list the jobs of every stage")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--only", nargs="+", default=None, help="run only the jobs with these names")
    ap.add_argument("--from", dest="start", type=int, default=1, help="start at this job number of each stage")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed job")
    ap.add_argument("--python", default=os.environ.get("CDL_PYTHON", sys.executable))
    ap.add_argument("--python-toto", default=os.environ.get("CDL_PYTHON_TOTO"))
    ap.add_argument("--python-extra", default=os.environ.get("CDL_PYTHON_EXTRA"))
    ap.add_argument("--python-sundial", default=os.environ.get("CDL_PYTHON_SUNDIAL"))
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")       # console encodings without every character of the logs
    exes ={"main": a.python, "toto": a.python_toto, "extra": a.python_extra, "sundial": a.python_sundial}
    if a.list or not a.stages:
        for s in ORDER + ["smoke"]:
            print(f"{s}:")
            for i, (name, env, folder, args, device) in enumerate(STAGES[s], 1):
                print(f"  {i:02d} {name:26s} {env:7s} {device:13s} {folder}/{' '.join(args)}")
        return
    stages = ORDER if a.stages == ["all"] else a.stages
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s): {', '.join(unknown)}")
    if "tables" in stages:
        for d in ["tables", "figures"]:
            os.makedirs(os.path.join(ROOT, "outputs", d), exist_ok=True)
    failed = []
    for s in stages:
        for i, job in enumerate(STAGES[s], 1):
            if i < a.start or (a.only and job[0] not in a.only):
                continue
            exe = exes[job[1]]
            if exe is None:
                print(f"[{s} {i:02d}] {job[0]}: no interpreter for the {job[1]} environment "
                      f"(--python-{job[1]} or CDL_PYTHON_{job[1].upper()})")
                failed.append(f"{s}/{job[0]}")
                if not a.keep_going and not a.dry_run:
                    sys.exit(1)
                continue
            if not run_job(s, i, job, exe, a.dry_run):
                failed.append(f"{s}/{job[0]}")
                if not a.keep_going:
                    sys.exit(1)
    if failed:
        print("failed or not run:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
