# %%
# In-text numbers not printed by the other scripts: label screening per type (Section 2.4),
# mask library statistics (Section 4.2) and the incompleteness of the test flights.
import glob
import numpy as np
import pandas as pd

import pipeline_tod as P
import experiment_missing_descent as E

import warnings

warnings.filterwarnings("ignore")


def coverage(g):
    g = g.sort_values("timestamp").dropna(subset=["altitude"]).reset_index(drop=True)
    if len(g) < 20:
        return None
    des = g.loc[int(g.altitude.idxmax()):]
    top, bot = des.altitude.max(), des.altitude.min()
    if len(des) < 10 or top - bot < 8000:
        return None
    idx = np.clip(((top - des.altitude) / (top - bot) * E.NB).astype(int), 0, E.NB - 1)
    m = np.zeros(E.NB, bool)
    m[np.unique(idx)] = True
    return m


def share_incomplete(files, ids=None):
    n = inc = 0
    for f in files:
        filters = [("flight_id", "in", ids)] if ids is not None else None
        raw = P.normalise(pd.read_parquet(f, filters=filters))
        for _, g in raw.groupby("flight_id"):
            m = coverage(g)
            if m is not None:
                n += 1
                inc += not m.all()
    return n, inc


#### ----------- label screening ----------- ####
d = pd.read_csv("data/todmass_full.csv", usecols=["aircraft_type", "tod_mass", "tow", "m_tow"]).dropna()
oew = {t: P.get_oew(t) for t in d.aircraft_type.unique()}
below = d.tod_mass < d.aircraft_type.map(oew)
wtc = {t: P.get_wtc(P.get_mtow(t)) for t in d.aircraft_type.unique()}
share = below.groupby(d.aircraft_type).mean().mul(100)
print(f"labelled {len(d):,}; TOW > 1.05 MTOW {int((d.tow > 1.05 * d.m_tow).sum()):,}; below OEW {int(below.sum()):,}")
print("below OEW, heavy types [%]:", share[[t for t in share.index if wtc[t] == "H"]].sort_values(ascending=False).round(1).to_dict())
print(f"below OEW, largest medium type: {share[[t for t in share.index if wtc[t] == 'M']].max():.2f} %")

#### ----------- mask library ----------- ####
M = np.load("results/mask_library.npy")
cov = M.mean(1)
gaps = np.array([int((np.diff(np.r_[1, m.astype(int), 1]) == -1).sum()) for m in M])
print(f"\nmask library: {len(M):,} masks; coverage median {np.median(cov):.2f}, "
      f"quartiles {np.percentile(cov, 25):.2f}/{np.percentile(cov, 75):.2f}; "
      f"one gap {100 * (gaps == 1).mean():.1f} %, two {100 * (gaps == 2).mean():.1f} %")
ret = 100 * M.mean(0)
print(f"band retention below TOD {ret[1]:.1f} %, halfway {ret[9]:.1f} %, near the ground {ret[18]:.1f} %")
n, inc = share_incomplete(sorted(glob.glob("data/24challenge_data/2022-0[37]-*.parquet")))
print(f"descents in the harvest months: {n:,}, incomplete {inc:,} ({100 * inc / n:.1f} %)")

#### ----------- test flights of the missing-descent experiment ----------- ####
every = sorted(glob.glob("data/24challenge_data/*.parquet"))
files = [every[i] for i in dict.fromkeys(np.linspace(0, len(every) - 1, 12).round().astype(int))]
n, inc = share_incomplete(files, E.reference_set().index.tolist())
print(f"test flights already incomplete: {inc:,} of {n:,} ({100 * inc / n:.1f} %)")
