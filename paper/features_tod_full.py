# %%
# Stage 2: data/processed_tod/ -> data/traj_tod_full.parquet (descent features and TOD mass).
import click
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings

warnings.filterwarnings("ignore")

def calc_duration(df_phase):
    if len(df_phase) == 0:
        return 0
    return (
        (df_phase.timestamp.values[-1] - df_phase.timestamp.values[0]) / 10**9
    ).astype(int)


# %%
#### -----------Feature extraction----------- ####
def rod_sectors(df_descent, n=5):
    top, bottom = df_descent.altitude.max(), df_descent.altitude.min()
    edges = np.linspace(bottom, top, n + 1)
    out = {}
    for i in range(n):
        lo, hi = edges[n - i - 1], edges[n - i]  # sector 1 is the highest band
        band = df_descent[(df_descent.altitude >= lo) & (df_descent.altitude <= hi)]
        out[f"mean_rod_s{i + 1}"] = band.vertical_rate.mean() if len(band) else np.nan
    return out


def descent_features(df1):
    df_cruise = df1.query("cruise==1")
    df_descent = df1.query("phase=='DESCENT'")

    if len(df_descent) == 0:
        return None

    if len(df_cruise) > 0:
        tas_percentiles = np.percentile(df_cruise.tas, [15, 85])
        altitude_percentiles = np.percentile(df_cruise.altitude, [15, 85])
        df_cruise = df_cruise.query(
            f"{tas_percentiles[0]} < tas < {tas_percentiles[1]}"
        )
        df_cruise = df_cruise.query(
            f"{altitude_percentiles[0]} < altitude < {altitude_percentiles[1]}"
        )

    if len(df_cruise) == 0:
        df_cruise = df1.query(
            f"altitude>={np.percentile(df1.altitude,80)}"
        ).reset_index(drop=True)
        start, end = int(len(df_cruise) * 0.2), int(len(df_cruise) * 0.8)
        df_cruise = df_cruise.iloc[start:end] if start != end else df_cruise

    if len(df_cruise) == 0:
        return None

    return {
        "flight_id": df1.flight_id.values[0],
        "mean_rod": df_descent.vertical_rate.mean(),
        "median_rod": df_descent.vertical_rate.median(),
        "min_rod": df_descent.vertical_rate.min(),
        "max_rod": df_descent.vertical_rate.max(),
        **rod_sectors(df_descent),
        "dur_descent": calc_duration(df_descent),
        "tod_mass": df_descent.mass.max(),
        "des_dist": df_descent.airdist.max() - df_descent.airdist.min(),
        "des_temp": df_descent.temperature.max(),
        "des_tas_min": df_descent.tas.min(),
        "des_tas_mean": df_descent.tas.mean(),
        "des_tas_max": df_descent.tas.max(),
        "des_gs_min": df_descent.groundspeed.min(),
        "des_gs_mean": df_descent.groundspeed.mean(),
        "des_gs_max": df_descent.groundspeed.max(),
        "tod_alt": df_cruise.altitude.iloc[-1],
        "tod_tas": df_cruise.tas.iloc[-1],
        "tod_gs": df_cruise.groundspeed.iloc[-1],
    }


def traj_df_creation(f, overwrite=False):
    date = f.split("/")[-1].split(".")[0]
    out = f"data/traj_tod_full/{date}.parquet"
    if Path(out).exists() and not overwrite:
        print(f"{out} already there")
        return

    df = pd.read_parquet(f)

    results = []
    for fid, df1 in df.groupby("flight_id"):
        df1 = df1.reset_index(drop=True).drop_duplicates(subset=["ts"])
        if df1.tas.isna().all() or df1.altitude.isna().all():
            continue
        if "mass" not in df1 or df1.mass.isna().all():
            continue  # no OpenAP model for this type, so no label
        feats = descent_features(df1)
        if feats is not None:
            results.append(feats)

    pd.DataFrame(results).to_parquet(out, index=False)
    return len(results)


# %%
@click.command()
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--limit", default=0, help="process only the first N days (0 = all)")
@click.option("--workers", default=6)
def main(overwrite, limit, workers):
    files = sorted(glob.glob("data/processed_tod/*.parquet"))
    if limit:
        files = files[:limit]
    Path("data/traj_tod_full/").mkdir(exist_ok=True)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        tasks = {
            executor.submit(traj_df_creation, file, overwrite) for file in files
        }
        for future in tqdm(
            as_completed(tasks), total=len(tasks), ncols=0, desc="labelling: "
        ):
            future.result()

    dfs = sorted(glob.glob("data/traj_tod_full/*.parquet"))
    df_all = pd.concat(pd.read_parquet(ff) for ff in dfs)
    df_all.to_parquet("data/traj_tod_full.parquet", index=False)
    print(f"{len(df_all):,} flights written to traj_tod_full.parquet")
    print(f"  with a tod_mass label: {df_all.tod_mass.notna().sum():,}")


# %%
if __name__ == "__main__":
    main()
