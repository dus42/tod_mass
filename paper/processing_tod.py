# %%
# Stage 1: raw trajectories -> data/processed_tod/ (15 s resampling, phases, propagated mass).
import os
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from traffic.core import Traffic, Flight
from openap import aero, FuelFlow, Thrust, Drag
import numpy as np
import openap
import gc
from datetime import datetime
import click

import warnings

warnings.filterwarnings("ignore")


# %%
def time_dups(f: Flight) -> Flight:
    if len(f.data) < 10 or np.percentile(f.data.altitude, 80) < 10_000:
        return None
    dur = f.duration.to_numpy().astype(int) / 10**9
    # res_ = max(1, int(dur / 600))
    f = f.resample("15s")
    # f = f.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    ts_diff = f.data.timestamp.diff().astype(int) / 10**9
    ts_diff[0] = 0
    f = f.assign(ts=ts_diff.cumsum())
    del ts_diff
    return f


# those labels are used for deriving thrust
def label(df):
    fp = openap.FlightPhase()
    fp.set_trajectory(df.ts, df.altitude, df.tas, df.vertical_rate)
    label = fp.phaselabel()
    dc = {
        "CL": "climb",
        "CR": "cruise",
        "DE": "descent_idle",
        "NA": "descent_idle",
        "LVL": "descent_idle",
        "GND": "descent_idle",
    }
    phase = [dc.get(item, item) for item in label]
    df = df.assign(label=phase)
    return df


# more accurate cruise phase extraction
def extract_cruise(f):
    f = f.phases().assign(cruise=0)
    df = f.data
    del f

    cli_ix = df.query("phase=='CLIMB'").index
    if len(cli_ix) == 0:
        id1 = df.index[0]
    else:
        id1 = cli_ix[-1]

    des_ix = df.query("phase=='DESCENT'").index
    if len(des_ix) == 0:
        id2 = df.index[-1]
    else:
        id2 = des_ix[0]

    df1 = df.loc[id1:id2, :].query(
        "(phase=='CRUISE' or phase=='LEVEL') "
        "and 45_000>altitude>20_000 and -500<vertical_rate<500 "
        "and tas==tas and altitude==altitude"
    )
    df.loc[df1.index, "cruise"] = 1
    del df1, cli_ix, des_ix

    return Flight(df)


N_ITER = 5


def assign_fuelflow_mass(f):
    df = f.data
    ac = df.aircraft_type.iloc[0].lower()
    ac_nosyn = ["bcs3", "bcs1"]
    df = df.assign(fuelflow=np.nan, fuel=np.nan, mass=np.nan)
    if ac in ac_nosyn:
        del ac, ac_nosyn
        return Flight(df)
    dT = f.data.query("altitude==altitude.min()").temperature.iloc[0] - aero.T0
    fuel = FuelFlow(ac=ac, use_synonym=True)

    dt = df.ts.diff().bfill().values
    tow = df.tow.values
    mass = tow.copy()
    for _ in range(N_ITER):
        ff = fuel.enroute(
            mass=mass, tas=df.tas, alt=df.altitude, vs=df.vertical_rate, dT=dT
        )
        ff = np.asarray(ff)
        mass = tow - np.cumsum(ff * dt)

    f = f.assign(fuelflow=ff, fuel=tow - mass, mass=mass)
    return f


def assign_thrust(f):
    df = f.data
    ac = df.aircraft_type.iloc[0].lower()
    ac_nosyn = ["bcs3", "bcs1"]
    df = df.assign(thrust=np.nan)
    if ac in ac_nosyn:
        del ac, ac_nosyn
        return Flight(df)
    drag = Drag(ac=ac, wave_drag=True, use_synonym=True)
    thrust = Thrust(ac=ac, use_synonym=True)
    mass = df.mass
    tas = df.tas
    alt = df.altitude
    vs = df.vertical_rate
    dT = df.query("altitude==altitude.min()").temperature.iloc[0] - aero.T0
    D = drag.clean(mass=mass, tas=tas, alt=alt, vs=vs, dT=dT)
    gamma = np.arctan2(vs * aero.fpm, tas * aero.kts)
    T = D + mass * 9.81 * np.sin(gamma)

    T_max = thrust.climb(tas=tas, alt=alt, roc=vs, dT=dT)
    T_idle = thrust.descent_idle(tas=tas, alt=alt, dT=dT)

    T = (
        (
            np.log(1 + np.exp(20 * (T - T_idle * 0.8) / 100_000))
            - np.log(1 + np.exp(20 * (T - T_max * 1.2) / 100_000))
        )
        / (np.log(1 + np.exp(20)))
    ) * 100_000 + T_idle * 0.8

    return f.assign(thrust=T)


def compute_airdist(f):
    vg = f.data.groundspeed * aero.kts
    dt = f.data.timestamp.diff().mean().total_seconds()
    vgx = vg * np.sin(np.radians(f.data.track))
    vgy = vg * np.cos(np.radians(f.data.track))
    vax = vgx - f.data.u_component_of_wind
    vay = vgy - f.data.v_component_of_wind
    va = np.sqrt((vax**2 + vay**2))
    tas = va / aero.kts
    airdist = tas * dt * aero.kts
    f = f.assign(
        tas=tas,
        airdist=airdist.cumsum() / 1000,
    )
    del vg, dt, vgx, vgy, vax, vay, va, tas, airdist
    f = f.assign(distance=f.airdist_max).sort_values(by="timestamp")

    return f


def func(file, overwrite=False):

    date = file.split("/")[1].split(".")[0]
    if os.path.exists(f"data/processed_tod/" + date + ".parquet") and not overwrite:
        print(f"{date} already there")
        return

    df_ac = pd.read_csv("data/flight_list.csv")[["flight_id", "aircraft_type", "tow"]]

    print(f"file: {date} \t {datetime.now()}\n")
    df1 = pd.read_parquet(file)
    df1 = df1.merge(df_ac, on="flight_id", how="inner").query("flight_id!=254165511")
    t = Traffic(df1)
    t = (
        t.filter()
        # .resample(1000)
        .pipe(time_dups)
        .pipe(compute_airdist)
        .pipe(assign_fuelflow_mass)
        .pipe(extract_cruise)
        .eval(6, desc=f"{date}")
    )
    df1 = t.data.dropna(subset=["flight_id"])
    df1["flight_id"] = df1["flight_id"].astype(int)
    df1.to_parquet(f"data/processed_tod/" + date + ".parquet", index=False)
    print(f"file: {date} is done \t {datetime.now()}\n")
    del df1, t, df_ac
    gc.collect()


# %%
@click.command()
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--limit", default=0, help="process only the first N days (0 = all)")
def main(overwrite, limit):
    files = sorted(glob.glob("data/24challenge_data/*.parquet"))
    if limit:
        files = files[:limit]
    Path("data/processed_tod/").mkdir(exist_ok=True)

    for file in files:
        func(file, overwrite)


# %%
if __name__ == "__main__":
    main()
