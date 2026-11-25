"""Data added in the second revision: AEMO 2015-2019, the definition check of the Monash Australian series,
and ERCOT 2026.

  AEMO 2015-11 to 2019-12   data/aemo_2015_2019/PRICE_AND_DEMAND_<YYYYMM>_<REGION>.csv (AEMO price-and-demand files)
  AEMO 2014-01 to 2015-04   data/aemo_2015_2019/overlap/...   (overlap with the end of the Monash series, for the check)
  ERCOT 2025-07 to 2026     data/ercot_2026/EIA930_SUBREGION_<period>_ERCO.csv
  ERCOT 2021                data/ercot_2026/validation/EIA930_SUBREGION_2021_<half>_ERCO.csv   (overlap with ercot.parquet)

ERCOT weather-zone load: ercot.com (Hourly Load Data Archives, Native_Load_<YYYY>) refuses automated downloads
(HTTP 403). The same eight weather-zone series are published by the U.S. Energy Information Administration in the
EIA-930 "subregion" bulk files (balancing authority ERCO; demand reported by ERCOT), which are public. The ERCO rows
of these files are stored unchanged (header plus the rows whose balancing authority is ERCO). The 2021 files are
compared with the last months of ercot.parquet to validate the mapping of the zones (check_ercot).

Every loader returns a list of pandas Series with a DatetimeIndex (time stamp = end of the interval, local time:
AEMO market time, Central Prevailing Time for ERCOT), in the series order of the corresponding dataset.

Usage:
  python fresh_data.py download      downloads all files above (skips files that exist)
  python fresh_data.py check         Monash vs AEMO and ERCOT mapping; writes results/rev2/data_checks.json
"""

import glob
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
AEMO_URL = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_{region}.csv"
EIA_URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_SUBREGION_{period}.csv"
UA = {"User-Agent": "CDL-Eval research data download (python-requests)"}
# ERCOT weather zones: names in ercot.parquet (series order) and EIA-930 subregion codes
ERCOT_ZONES = ["COAST", "EAST", "FAR_WEST", "NORTH", "NORTH_C", "SOUTHERN", "SOUTH_C", "WEST"]
EIA_CODE = {"COAST": "COAS", "EAST": "EAST", "FAR_WEST": "FWES", "NORTH": "NRTH", "NORTH_C": "NCEN",
            "SOUTHERN": "SOUT", "SOUTH_C": "SCEN", "WEST": "WEST"}
ERCOT_PERIODS = ["2025_Jul_Dec", "2026_Jan_Jun", "2026_Jul_Dec"]
ERCOT_VALID = ["2021_Jan_Jun", "2021_Jul_Dec"]
STORM = (pd.Timestamp("2021-02-10"), pd.Timestamp("2021-02-22"))   # excluded from the monthly ratios of check_ercot


def months(first, last):
    """['YYYYMM', ...] from first to last inclusive (strings 'YYYY-MM')."""
    return [p.strftime("%Y%m") for p in pd.period_range(first, last, freq="M")]


# =====================================================================
# Download
# =====================================================================

def _get(url, dest, timeout=120):
    import requests
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    return len(r.content)


def download_aemo(first, last, dest, regions=REGIONS):
    os.makedirs(dest, exist_ok=True)
    n = 0
    for ym in months(first, last):
        for region in regions:
            f = os.path.join(dest, f"PRICE_AND_DEMAND_{ym}_{region}.csv")
            if os.path.exists(f):
                continue
            size = _get(AEMO_URL.format(ym=ym, region=region), f)
            n += 1
            print(f"[aemo] {os.path.basename(f)} {size} bytes", flush=True)
    return n


def download_eia(periods, dest, ba="ERCO"):
    """Downloads EIA-930 subregion six-month files and keeps the header and the rows of one balancing authority."""
    import requests
    os.makedirs(dest, exist_ok=True)
    for p in periods:
        out = os.path.join(dest, f"EIA930_SUBREGION_{p}_{ba}.csv")
        if os.path.exists(out) and not p.startswith("2026"):        # the current half-year is refreshed
            continue
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "raw.csv")
            with requests.get(EIA_URL.format(period=p), headers=UA, timeout=600, stream=True) as r:
                r.raise_for_status()
                with open(raw, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            kept = 0
            with open(raw, "r", encoding="utf-8-sig") as fi, open(out + ".part", "w", encoding="utf-8") as fo:
                fo.write(fi.readline())
                for line in fi:
                    if line.startswith(ba + ",") or line.startswith(f'"{ba}",'):
                        fo.write(line)
                        kept += 1
            os.replace(out + ".part", out)
        print(f"[eia] {os.path.basename(out)}: {kept} rows", flush=True)


def download_all():
    download_aemo("2015-11", "2019-12", os.path.join(DATA, "aemo_2015_2019"))
    download_aemo("2014-01", "2015-04", os.path.join(DATA, "aemo_2015_2019", "overlap"))
    download_eia(ERCOT_PERIODS, os.path.join(DATA, "ercot_2026"))
    download_eia(ERCOT_VALID, os.path.join(DATA, "ercot_2026", "validation"))


# =====================================================================
# Loaders
# =====================================================================

def aemo_half_hourly(files):
    """Half-hourly TOTALDEMAND from AEMO price-and-demand files, processed exactly as bench.load_aemo: five-minute
    values (from October 2021) are averaged over the half hour ending at the time stamp; older files are already
    half-hourly, and the same resampling leaves them unchanged."""
    df = pd.concat([pd.read_csv(f) for f in files])
    df["t"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    s = df.set_index("t")["TOTALDEMAND"].sort_index()
    return s[~s.index.duplicated()].resample("30min", label="right", closed="right").mean().interpolate()


def load_aemo_dir(sub, regions=REGIONS):
    out = []
    for region in regions:
        files = sorted(glob.glob(os.path.join(DATA, sub, f"PRICE_AND_DEMAND_*_{region}.csv")))
        if not files:
            raise FileNotFoundError(f"no AEMO files for {region} in data/{sub}; run: python fresh_data.py download")
        out.append(aemo_half_hourly(files).rename(region))
    return out


def load_aemo_mid():
    """AEMO 2015-11 to 2019-12, five regions (NSW1, QLD1, SA1, TAS1, VIC1)."""
    return load_aemo_dir("aemo_2015_2019")


def load_aemo2026():
    """AEMO December 2025 to September 2026 (the `aemo` dataset of bench.py)."""
    return load_aemo_dir("aemo_2026")


def load_parquet(fname):
    """Series of a chronos_datasets parquet file with their time stamps; missing values interpolated as in
    run.load_series."""
    df = pd.read_parquet(os.path.join(DATA, fname))
    out = []
    for i, (t, v) in enumerate(zip(df["timestamp"], df["target"])):
        x = pd.Series(np.asarray(v, dtype=float)).interpolate(limit_direction="both").to_numpy()
        out.append(pd.Series(x, index=pd.DatetimeIndex(pd.to_datetime(t)), name=str(df["id"].iloc[i])))
    return out


def parquet_timestamps(fname):
    """Time stamps of every series of a chronos_datasets parquet file."""
    df = pd.read_parquet(os.path.join(DATA, fname), columns=["timestamp"])
    return [pd.DatetimeIndex(pd.to_datetime(t)) for t in df["timestamp"]]


DATASET_FILES = {"ercot": "ercot.parquet", "traffic": "monash_traffic.parquet",
                 "ausel": "monash_australian_electricity.parquet"}


def read_eia(files, zones=ERCOT_ZONES):
    """Hourly demand per zone from EIA-930 subregion rows: time stamps converted from UTC to naive Central
    Prevailing Time (hour ending), the repeated hour of the autumn clock change dropped (first kept), the missing
    hour of the spring change and other gaps interpolated, as in ercot.parquet."""
    df = pd.concat([pd.read_csv(f, thousands=",") for f in files])
    df["utc"] = pd.to_datetime(df["UTC Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p")
    df["demand"] = pd.to_numeric(df["Demand (MW)"], errors="coerce")
    out = []
    for z in zones:
        d = df[df["Sub-Region"] == EIA_CODE[z]].sort_values("utc").drop_duplicates("utc")
        t = d["utc"].dt.tz_localize("UTC").dt.tz_convert("America/Chicago").dt.tz_localize(None)
        s = pd.Series(d["demand"].to_numpy(dtype=float), index=pd.DatetimeIndex(t), name=z)
        s = s[~s.index.duplicated(keep="first")]
        full = pd.date_range(s.index[0], s.index[-1], freq="h")
        out.append(s.reindex(full).rename(z))
    return out


def complete_months(series):
    """Cuts all series at the end of the last month that every series covers completely."""
    end = min(s.dropna().index[-1] for s in series)
    last_full = (end + pd.Timedelta(hours=1)).to_period("M").start_time   # first hour of the month after the last full month
    if end + pd.Timedelta(hours=1) != last_full:
        last_full = end.to_period("M").start_time                        # month of `end` is incomplete
    return [s[s.index <= last_full] for s in series]


def load_ercot2026(raw=False):
    """ERCOT weather-zone load from 2025-07-01 01:00 to the end of the latest complete month, hourly, Central
    Prevailing Time (hour ending), in the series order of ercot.parquet. Gaps are interpolated linearly unless
    raw=True."""
    files = [os.path.join(DATA, "ercot_2026", f"EIA930_SUBREGION_{p}_ERCO.csv") for p in ERCOT_PERIODS]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise FileNotFoundError("no ERCOT 2026 files in data/ercot_2026; run: python fresh_data.py download")
    series = complete_months(read_eia(files))
    start = pd.Timestamp("2025-07-01 01:00")
    series = [s[s.index >= start] for s in series]
    return series if raw else [s.interpolate(limit_direction="both") for s in series]


# =====================================================================
# Checks
# =====================================================================

def _compare(a, b, lags=range(-3, 4)):
    """Correlation and differences of two series on their common time stamps, for shifts of b by `lags` steps."""
    res = {}
    for lag in lags:
        bb = b.shift(lag, freq=pd.infer_freq(b.index[:200]))
        j = a.index.intersection(bb.index)
        x, y = a.loc[j].to_numpy(), bb.loc[j].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        slope, icpt = np.polyfit(y, x, 1)
        res[lag] = dict(n=int(len(x)), corr=float(np.corrcoef(x, y)[0, 1]), mad=float(np.abs(x - y).mean()),
                        mean_a=float(x.mean()), mean_b=float(y.mean()), ratio=float(x.mean() / y.mean()),
                        diff_mean=float((y - x).mean()), diff_sd=float((y - x).std()),
                        slope=float(slope), intercept=float(icpt),
                        share_equal=float(np.mean(np.abs(x - y) < 0.01)),
                        median_rel=float(np.median(np.abs(x - y) / np.abs(y))))
    best = max(res, key=lambda k: res[k]["corr"])
    return best, res


def check_monash():
    """Is monash_australian_electricity AEMO TOTALDEMAND? Each Monash series is compared with each region over
    their overlap (AEMO 2014-01 to 2015-04)."""
    mon = load_parquet(DATASET_FILES["ausel"])
    aemo = load_aemo_dir(os.path.join("aemo_2015_2019", "overlap"))
    out = {"overlap": [str(aemo[0].index[0]), str(aemo[0].index[-1])], "pairs": {}, "mapping": {}}
    for s in mon:
        s = s[s.index >= aemo[0].index[0] - pd.Timedelta(hours=2)]
        best_r, best = None, None
        for a in aemo:
            lag, res = _compare(s, a)
            out["pairs"][f"{s.name}|{a.name}"] = dict(best_lag=lag, **res[lag], lag0=res[0])
            if best is None or res[lag]["corr"] > best["corr"]:
                best_r, best = a.name, dict(best_lag=lag, **res[lag])
        out["mapping"][s.name] = dict(region=best_r, **best)
        print(f"[monash] {s.name} ~ {best_r}: lag {best['best_lag']} corr {best['corr']:.5f} "
              f"MAD {best['mad']:.2f} MW ratio {best['ratio']:.4f} equal {best['share_equal']:.3f} "
              f"(n={best['n']})", flush=True)
    return out


def check_ercot():
    """EIA-930 ERCOT zones in 2021 against the last months of ercot.parquet (to 2021-09-01)."""
    par = load_parquet(DATASET_FILES["ercot"])
    files = sorted(glob.glob(os.path.join(DATA, "ercot_2026", "validation", "EIA930_SUBREGION_2021_*_ERCO.csv")))
    eia = read_eia(files)
    out = {"zones": {}, "source": [os.path.basename(f) for f in files]}
    for p, e in zip(par, eia):
        p = p[p.index >= pd.Timestamp("2021-01-01")]
        lag, res = _compare(p, e)
        cross = {}
        for e2 in eia:                                               # the mapping must beat every other zone
            _, r2 = _compare(p, e2, lags=[0])
            cross[e2.name] = round(r2[0]["corr"], 5)
        jj = p.index.intersection(e.index)
        jj = jj[(jj < STORM[0]) | (jj >= STORM[1])]                  # load shedding of the February 2021 storm
        monthly = (e.loc[jj] / p.loc[jj]).resample("MS").median()
        ok = res[lag]["corr"] >= 0.98 and lag == 0 and bool((monthly - 1).abs().max() <= 0.05)
        out["zones"][p.name] = dict(eia=EIA_CODE[p.name], best_lag=lag, **res[lag], lag0=res[0], corr_all=cross,
                                    monthly_ratio={str(k.date()): round(float(v), 4) for k, v in monthly.items()},
                                    validated=ok)
        print(f"[ercot] {p.name:9s} <- {EIA_CODE[p.name]}: lag {lag} corr {res[lag]['corr']:.5f} "
              f"MAD {res[lag]['mad']:.1f} MW ratio {res[lag]['ratio']:.4f} equal {res[lag]['share_equal']:.3f} "
              f"median rel {res[lag]['median_rel']:.4f} (n={res[lag]['n']}); best other zone corr "
              f"{max(v for k, v in cross.items() if k != p.name):.3f}", flush=True)
    out["validated_zones"] = [z for z, v in out["zones"].items() if v["validated"]]
    print(f"[ercot] validated zones, which agree with ercot.parquet (corr >= 0.98, monthly median ratio within 5% "
          f"outside {STORM[0].date()} to {STORM[1].date()}): "
          f"{out['validated_zones']}", flush=True)
    new = load_ercot2026(raw=True)
    out["ercot2026"] = dict(start=str(new[0].index[0]), end=str(new[0].index[-1]), steps=int(len(new[0])),
                            missing={s.name: int(s.isna().sum()) for s in new},
                            nonpositive={s.name: int((s <= 0).sum()) for s in new})
    print(f"[ercot2026] {out['ercot2026']}", flush=True)
    return out


def check_aemo_mid():
    s = load_aemo_mid()
    return {r.name: dict(start=str(r.index[0]), end=str(r.index[-1]), steps=int(len(r)), mean=float(r.mean()))
            for r in s}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "download":
        download_all()
    if cmd in ("download", "check"):
        out = dict(monash_vs_aemo=check_monash(), ercot=check_ercot(), aemo_mid=check_aemo_mid())
        # the 2026 files are processed identically by bench.load_aemo and aemo_half_hourly
        os.makedirs(os.path.join(HERE, "results", "rev2"), exist_ok=True)
        with open(os.path.join(HERE, "results", "rev2", "data_checks.json"), "w") as f:
            json.dump(out, f, indent=1)
        print("written results/rev2/data_checks.json", flush=True)


if __name__ == "__main__":
    main()
