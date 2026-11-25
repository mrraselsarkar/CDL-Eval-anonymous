"""Downloads the datasets of the benchmark into realdata/data/; files that exist are skipped.

  chronos    ercot, monash_traffic, monash_australian_electricity from the Hugging Face Hub collection
             autogluon/chronos_datasets (the three held-out datasets)
  aemo2026   AEMO price-and-demand files of the five NEM regions, December 2025 to September 2026 (dataset aemo)
  fresh      AEMO 2015-11 to 2019-12 and 2014-01 to 2015-04 (datasets aemo_mid and the definition check of the Monash
             series), and the ERCOT rows of the EIA-930 subregion files (ercot2026 and the zone validation);
             realdata/fresh_data.py

Usage:  python scripts/download_data.py [chronos] [aemo2026] [fresh] [--dry-run] [--refresh]   (default: all three)

The EIA-930 file of the current half-year grows every month, so a later download adds months to the ercot2026 dataset
and changes its random test windows. The runs of the paper used the files downloaded on 28 October 2026, which end on
30 September 2026; the data snapshot of the results archive (README, Data) holds these files.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "realdata"))
import fresh_data  # noqa: E402  (download helpers, URLs and folders)

DATA = fresh_data.DATA
CHRONOS_URL = "https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/{name}/train-00000-of-00001.parquet"
CHRONOS = {"ercot": "ercot.parquet", "monash_traffic": "monash_traffic.parquet",
           "monash_australian_electricity": "monash_australian_electricity.parquet"}
AEMO_2026 = ("2025-12", "2026-09")


def download_chronos(dry):
    os.makedirs(DATA, exist_ok=True)
    for name, fname in CHRONOS.items():
        dest = os.path.join(DATA, fname)
        url = CHRONOS_URL.format(name=name)
        if os.path.exists(dest):
            print(f"[chronos] {fname} exists")
        elif dry:
            print(f"[chronos] {url} -> {dest}")
        else:
            size = fresh_data._get(url, dest, timeout=600)
            print(f"[chronos] {fname} {size} bytes", flush=True)


def download_aemo2026(dry):
    dest = os.path.join(DATA, "aemo_2026")
    if dry:
        months = fresh_data.months(*AEMO_2026)
        print(f"[aemo2026] {len(months) * len(fresh_data.REGIONS)} files "
              f"{fresh_data.AEMO_URL.format(ym=months[0], region=fresh_data.REGIONS[0])} ... -> {dest}")
        return
    n = fresh_data.download_aemo(*AEMO_2026, dest)
    print(f"[aemo2026] {n} files downloaded", flush=True)


def download_fresh(dry, refresh=False):
    """The files of fresh_data.download_all. EIA-930 files that exist are kept unless refresh is set (download_all
    itself replaces the files of 2026, which would change the ercot2026 windows; see above)."""
    aemo = [("2015-11", "2019-12", os.path.join(DATA, "aemo_2015_2019")),
            ("2014-01", "2015-04", os.path.join(DATA, "aemo_2015_2019", "overlap"))]
    eia = [(fresh_data.ERCOT_PERIODS, os.path.join(DATA, "ercot_2026")),
           (fresh_data.ERCOT_VALID, os.path.join(DATA, "ercot_2026", "validation"))]
    for first, last, dest in aemo:
        if dry:
            print(f"[fresh] AEMO {first} to {last} -> {dest}")
        else:
            fresh_data.download_aemo(first, last, dest)
    for periods, dest in eia:
        todo = [p for p in periods
                if refresh or not os.path.exists(os.path.join(dest, f"EIA930_SUBREGION_{p}_ERCO.csv"))]
        for p in sorted(set(periods) - set(todo)):
            print(f"[fresh] EIA930_SUBREGION_{p}_ERCO.csv exists (--refresh replaces it)")
        if dry:
            for p in todo:
                print(f"[fresh] {fresh_data.EIA_URL.format(period=p)} (ERCO rows) -> {dest}")
        elif todo:
            fresh_data.download_eia(todo, dest)


def main():
    dry = "--dry-run" in sys.argv
    parts = [a for a in sys.argv[1:] if not a.startswith("--")] or ["chronos", "aemo2026", "fresh"]
    steps = {"chronos": download_chronos, "aemo2026": download_aemo2026, "fresh": download_fresh}
    for p in parts:
        if p not in steps:
            sys.exit(f"unknown part {p!r}; choose from {', '.join(steps)}")
        if p == "fresh":
            download_fresh(dry, refresh="--refresh" in sys.argv)
        else:
            steps[p](dry)


if __name__ == "__main__":
    main()
