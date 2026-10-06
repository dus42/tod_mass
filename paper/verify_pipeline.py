# %%
# Checks that the released pipeline reproduces the training features (Appendix 1).
import time, numpy as np, pandas as pd
import pipeline_tod as P
import warnings; warnings.filterwarnings("ignore")

DAY = "data/24challenge_data/2022-06-15.parquet"
N = 300
ref = pd.read_parquet("data/traj_tod_full.parquet").set_index("flight_id")
day_ids = pd.read_parquet(DAY, columns=["flight_id"]).flight_id.unique()
pool = ref.index.intersection(pd.Index(day_ids))
ids = pd.Index(pool).to_series().sample(min(N, len(pool)), random_state=0).tolist()

import tempfile, os
sub = os.path.join(tempfile.mkdtemp(), "verify_day.parquet")
df = pd.read_parquet(DAY, filters=[("flight_id", "in", ids)])
types = pd.read_csv("data/todmass_full.csv", usecols=["flight_id", "aircraft_type"])
df = df.merge(types, on="flight_id", how="left")
df.to_parquet(sub, index=False)
print(f"{df.flight_id.nunique()} flights of {len(pool)} available on {DAY[-18:-8]}\n")

ADSB = ["mean_rod","median_rod","min_rod","dur_descent",
        "des_gs_min","des_gs_mean","des_gs_max","tod_alt","tod_gs"]
WX   = ["des_tas_min","des_tas_mean","des_tas_max","tod_tas","des_dist","des_temp"]

for weather in (False, True):
    t0 = time.time()
    rows = P.traj_df_creation(sub, weather, "/tmp/era5-zarr", P.MIN_DESCENT)
    el = time.time() - t0
    d = pd.DataFrame(rows).set_index("flight_id")
    common = d.index.intersection(ref.index)
    tag = "ADS-B + weather" if weather else "ADS-B only"
    print(f"### {tag}: {len(d)} flights in {el:.1f}s = {el/max(len(d),1):.2f}s per flight")
    cols = ADSB + (WX if weather else [])
    for c in cols:
        a, b = d.loc[common, c].astype(float), ref.loc[common, c].astype(float)
        m = a.notna() & b.notna()
        if m.sum() == 0:
            print(f"  {c:<14} no overlap"); continue
        exact = np.isclose(a[m], b[m], rtol=1e-9, atol=1e-9).mean()
        den = b[m].abs().replace(0, np.nan)
        rel = (100 * (a[m] - b[m]).abs() / den).dropna()
        print(f"  {c:<14} n={m.sum():<4} exact {100*exact:5.1f} %   "
              f"mean |rel err| {rel.mean():7.3f} %   max {rel.max():8.3f} %")
    print()
