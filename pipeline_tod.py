# %%
# Raw ADS-B trajectories -> descent features -> TOD mass estimate.
#   python pipeline_tod.py --input trajectories.parquet --model model_tod_noweather_all_nomassrange.txt
import click
import gzip
from pathlib import Path
import re
from pathlib import Path
import numpy as np
import pandas as pd
import openap
from openap import aero
from tqdm import tqdm
from traffic.core import Traffic, Flight
from traffic.data import airports, aircraft
from traffic.data.adsb.opensky import format_history

from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings

warnings.filterwarnings("ignore")

# %%
#### -----------Aircraft properties----------- ####
# OpenAP MTOW, with overrides for the types it has no data for or resolves to a
# poor synonym. Sources are the ones noted in generate_actows.py.
AC_MTOW_OVERRIDES = {
    "a310": 150000,  # eurocontrol
    "at76": 23000,  # skybrary
    "bcs1": 63100,  # eurocontrol
    "bcs3": 69900,  # eurocontrol
    "c56x": 8709,  # eurocontrol
    "crj9": 38330,  # eurocontrol
    "e290": 56400,  # skybrary
}


MODELS = Path(__file__).resolve().parent / "models"


def load_model(name):
    import lightgbm as lgb

    for path in [Path(name), MODELS / name, MODELS / f"{name}.gz"]:
        if not path.exists():
            continue
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                return lgb.Booster(model_str=f.read())
        return lgb.Booster(model_file=str(path))
    raise SystemExit(f"model {name} not found; download the models into {MODELS}")


def get_mtow(ac):
    ac = ac.lower()
    if ac in AC_MTOW_OVERRIDES:
        return AC_MTOW_OVERRIDES[ac]
    try:
        return openap.prop.aircraft(ac, use_synonym=True)["mtow"]
    except Exception:
        return np.nan


# OpenAP OEW, with published values for the types it has no data for. Without these
# OpenAP falls back to a synonym: the AT76 resolves to an Embraer and the A310 to a
# narrowbody at half its real empty weight.
AC_OEW_OVERRIDES = {
    "a310": 80000,  # airbus
    "at76": 13010,  # atr factsheet, technical-specification OEW
    "c56x": 5900,  # cessna
    "e290": 28500,  # embraer
}


def get_oew(ac):
    ac = ac.lower()
    if ac in AC_OEW_OVERRIDES:
        return AC_OEW_OVERRIDES[ac]
    try:
        return openap.prop.aircraft(ac, use_synonym=True)["oew"]
    except Exception:
        return np.nan


def get_wtc(mtow):
    # ICAO wake turbulence category from MTOW; reproduces the challenge-set wtc
    # for all 30 types once the overrides above are applied.
    if mtow != mtow:
        return None
    if mtow <= 7000:
        return "L"
    if mtow < 136000:
        return "M"
    return "H"


# %%
#### -----------Functions here----------- ####
def classify_time(hour, nb_of_partition=24):
    range_of_hour = 24 / nb_of_partition

    return hour // range_of_hour


def calc_duration(df_phase):
    if len(df_phase) == 0:
        return 0
    return (
        (df_phase.timestamp.values[-1] - df_phase.timestamp.values[0]) / 10**9
    ).astype(int)


# drops time duplicates and resamples the flight to a uniform 15 s, as in
# processing_tod.py. Uniform spacing keeps the rate-of-descent statistics comparable
# between short and long flights, and it is the sampling the models are trained on.
def time_dups(f: Flight) -> Flight:
    if len(f.data) < 10 or np.percentile(f.data.altitude, 80) < 10_000:
        return None
    f = f.resample("15s")
    ts_diff = f.data.timestamp.diff().astype(int) / 10**9
    ts_diff[0] = 0
    f = f.assign(ts=ts_diff.cumsum())
    del ts_diff
    return f


# deriving tas from groundspeed and wind data
def compute_airdist(f):
    if "u_component_of_wind" not in f.data.columns:
        # no weather: airspeed is not recoverable, groundspeed features still are
        return f.assign(tas=np.nan, airdist=np.nan)
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
    return f.sort_values(by="timestamp")


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

    query = (
        "(phase=='CRUISE' or phase=='LEVEL') "
        "and 45_000>altitude>20_000 and -500<vertical_rate<500 "
        "and altitude==altitude"
    )
    if df.tas.notna().any():
        query += " and tas==tas"
    df1 = df.loc[id1:id2, :].query(query)
    df.loc[df1.index, "cruise"] = 1
    del df1, cli_ix, des_ix

    return Flight(df)


# %%
#### -----------Input schema----------- ####
def normalise(df):
    if "baroaltitude" not in df.columns:
        return df  # already in traffic naming

    print("  raw OpenSky state vectors detected, formatting with traffic")
    df = format_history(df)

    df = df.drop(
        columns=[
            c
            for c in ["serials", "hour", "squawk", "last_position"]
            if c in df.columns
        ]
    )
    if "onground" in df.columns:
        df = df.query("~onground")
    return df


# %%
#### -----------Weather----------- ####
def add_weather(df, local_store):
    from fastmeteo.source import ArcoEra5

    grid = ArcoEra5(local_store=local_store)
    return grid.interpolate(df)


# %%
#### -----------Feature extraction----------- ####
MIN_DESCENT = 345


def rod_sectors(df_descent, n=5):
    top, bottom = df_descent.altitude.max(), df_descent.altitude.min()
    edges = np.linspace(bottom, top, n + 1)
    out = {}
    for i in range(n):
        lo, hi = edges[n - i - 1], edges[n - i]  # sector 1 is the highest band
        band = df_descent[(df_descent.altitude >= lo) & (df_descent.altitude <= hi)]
        out[f"mean_rod_s{i + 1}"] = band.vertical_rate.mean() if len(band) else np.nan
    return out


def descent_features(df1, min_descent=MIN_DESCENT):
    if df1.altitude.isna().sum() == len(df1):
        return None

    df_cruise = df1.query("cruise==1")
    df_descent = df1.query("phase=='DESCENT'")

    if len(df_descent) == 0 or calc_duration(df_descent) < min_descent:
        return None

    if len(df_cruise) > 0:
        altitude_percentiles = np.percentile(df_cruise.altitude, [15, 85])
        if df_cruise.tas.notna().any():
            tas_percentiles = np.percentile(df_cruise.tas.dropna(), [15, 85])
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

    tod = df_descent.iloc[0]

    ac = df1.aircraft_type.values[0]
    m_tow = get_mtow(ac)
    arrival = pd.Timestamp(df1.timestamp.values[-1]).tz_localize("UTC")

    return {
        "flight_id": df1.flight_id.values[0],
        "icao24": df1.icao24.values[0] if "icao24" in df1.columns else None,
        "callsign": (
            str(df1.callsign.values[0]).strip() if "callsign" in df1.columns else None
        ),
        "arrival_ts": arrival,
        "aircraft_type": ac,
        "wtc": get_wtc(m_tow),
        "m_tow": m_tow,
        "oew": get_oew(ac),
        "mean_rod": df_descent.vertical_rate.mean(),
        "median_rod": df_descent.vertical_rate.median(),
        "min_rod": df_descent.vertical_rate.min(),
        "max_rod": df_descent.vertical_rate.max(),
        **rod_sectors(df_descent),
        "dur_descent": calc_duration(df_descent),
        "des_dist": df_descent.airdist.max() - df_descent.airdist.min(),
        "des_temp": (
            df_descent.temperature.max()
            if "temperature" in df_descent.columns
            else np.nan
        ),
        "des_tas_min": df_descent.tas.min(),
        "des_tas_mean": df_descent.tas.mean(),
        "des_tas_max": df_descent.tas.max(),
        "des_gs_min": df_descent.groundspeed.min(),
        "des_gs_mean": df_descent.groundspeed.mean(),
        "des_gs_max": df_descent.groundspeed.max(),
        "tod_time": pd.Timestamp(tod.timestamp),
        "tod_lat": tod.latitude,
        "tod_lon": tod.longitude,
        "tod_alt": df_cruise.altitude.iloc[-1],
        "tod_tas": df_cruise.tas.iloc[-1],
        "tod_gs": df_cruise.groundspeed.iloc[-1],
        "time_of_day_arrival": int(classify_time(arrival.hour)),
        "day_of_year": int(arrival.dayofyear),
    }


def traj_df_creation(f, weather, local_store, min_descent=MIN_DESCENT):
    df = pd.read_parquet(f) if str(f).endswith(".parquet") else pd.read_csv(f)
    df = normalise(df)
    df = df.astype({c: "float64" for c in df.select_dtypes("number").columns})
    df = df.astype({c: "str" for c in df.select_dtypes(["string", "object"]).columns})
    ts = df["timestamp"]
    if "[pyarrow]" in str(ts.dtype):
        ts = ts.astype(
            "datetime64[ns, UTC]" if "tz=" in str(ts.dtype) else "datetime64[ns]"
        )
    df["timestamp"] = pd.to_datetime(ts, utc=True)

    if "icao24" in df.columns:
        df["icao24"] = df["icao24"].astype(str).str.lower().str.strip()

    if "aircraft_type" not in df.columns:
        if "typecode" in df.columns:
            df = df.rename(columns={"typecode": "aircraft_type"})
        else:
            ac_db = aircraft.data[["icao24", "typecode"]].drop_duplicates("icao24")
            ac_db["icao24"] = ac_db["icao24"].astype(str).str.lower()
            df = df.merge(ac_db, on="icao24", how="left")
            df = df.rename(columns={"typecode": "aircraft_type"})
            if df.aircraft_type.isna().all():
                raise SystemExit(
                    "no icao24 resolved to a typecode -- if the identifiers are "
                    "anonymised, supply an `aircraft_type` column instead"
                )
    df["aircraft_type"] = df["aircraft_type"].str.strip().replace("", np.nan)
    df = df.dropna(subset=["aircraft_type"])
    df["aircraft_type"] = df["aircraft_type"].str.upper()

    if "flight_id" not in df.columns:
        key = df.icao24 + "_" + df.get("callsign", pd.Series("", index=df.index)).fillna("")
        df = df.assign(flight_id=key)

    t = Traffic(df).filter().pipe(time_dups).eval(desc="")
    if t is None:
        return []

    if weather:
        t = Traffic(add_weather(t.data, local_store))

    t = t.pipe(compute_airdist).pipe(extract_cruise).eval(desc="")
    if t is None:
        return []

    results = []
    for flight in t:
        df1 = flight.data.reset_index(drop=True).drop_duplicates(subset=["ts"])
        feats = descent_features(df1, min_descent)
        if feats is None:
            continue
        try:
            ap = flight.infer_airport("landing")
            if not re.fullmatch(r"[A-Z]{4}", str(ap.icao)):
                raise ValueError(ap.icao)
            feats["ades"] = str(ap.icao)
            feats["lat_arr"], feats["lon_arr"] = ap.latlon
        except Exception:
            feats["ades"] = None
            feats["lat_arr"], feats["lon_arr"] = np.nan, np.nan
        results.append(feats)
    return results


# %%
#### -----------Flight information----------- ####
def apply_flight_list(feats, path):
    fl = pd.read_csv(path)
    for col in ["firstseen", "lastseen"]:
        fl[col] = pd.to_datetime(fl[col], utc=True)
    fl["icao24"] = fl.icao24.astype(str).str.lower().str.strip()
    fl["callsign"] = fl.callsign.astype(str).str.strip()
    fl = fl.dropna(subset=["arrival"])

    fl = fl.assign(
        flight_duration=(fl.lastseen - fl.firstseen).dt.total_seconds() / 60
    )

    merged = feats.merge(
        fl[["icao24", "callsign", "departure", "arrival", "lastseen", "flight_duration"]],
        on=["icao24", "callsign"],
        how="left",
        suffixes=("", "_fl"),
    )
    merged["dt"] = (merged.lastseen - merged.arrival_ts).abs()
    merged = merged[merged.dt.isna() | (merged.dt < pd.Timedelta("2h"))]
    merged = (
        merged.sort_values("dt").groupby("flight_id", as_index=False).first()
    )

    matched = merged.arrival.notna()
    print(f"  flight list matched {matched.sum()} of {len(feats)} flight(s)")

    merged.loc[matched, "adep"] = merged.loc[matched, "departure"]
    merged.loc[matched, "ades"] = merged.loc[matched, "arrival"]

    def latlon(code):
        if not isinstance(code, str):
            return (np.nan, np.nan)
        try:
            a = airports[code]
        except (ValueError, TypeError):
            return (np.nan, np.nan)
        return a.latlon if a else (np.nan, np.nan)

    for icao_col, la, lo in [("adep", "lat_dep", "lon_dep"), ("ades", "lat_arr", "lon_arr")]:
        pos = merged[icao_col].map(latlon)
        merged[la] = [p[0] for p in pos]
        merged[lo] = [p[1] for p in pos]

    merged["distance_ap"] = [
        openap.aero.distance(a, b, c, d) if not any(pd.isna([a, b, c, d])) else np.nan
        for a, b, c, d in zip(
            merged.lat_dep, merged.lon_dep, merged.lat_arr, merged.lon_arr
        )
    ]
    return merged.drop(columns=["departure", "arrival", "lastseen", "dt"])


# %%
#### -----------Encoding----------- ####
CT_FEATURES = ["adep", "ades", "wtc", "aircraft_type", "day_of_year", "time_of_day_arrival"]


def as_categorical(df, features):
    df = df.copy()
    for col in [c for c in CT_FEATURES if c in features]:
        df[col] = df[col].astype("category")
    return df


# %%
@click.command()
@click.option("--input", "input_", required=True, help="raw ADS-B .parquet/.csv, or a folder of them")
@click.option("--output", default="tod_features.parquet", help="feature table to write")
@click.option("--weather/--no-weather", default=False, help="interpolate ERA5 with fastmeteo")
@click.option("--local-store", default="/tmp/era5-zarr", help="fastmeteo grid cache")
@click.option("--workers", default=6, help="parallel workers over input files")
@click.option("--model", default=None, help="a model_tod_*.txt to also predict TOD mass with")
@click.option("--min-descent", default=MIN_DESCENT, help="drop flights with a shorter descent, in seconds")
@click.option("--flight-list", default=None, help="OpenSky flight list csv, adds the flight-information features")
def main(input_, output, weather, local_store, workers, model, min_descent, flight_list):
    files = sorted(Path(input_).glob("*.parquet")) if Path(input_).is_dir() else [Path(input_)]
    print(f"#### -----------{len(files)} file(s), weather={weather}----------- ###")

    all_results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        tasks = {
            executor.submit(traj_df_creation, f, weather, local_store, min_descent)
            for f in files
        }
        for future in tqdm(
            as_completed(tasks), total=len(tasks), ncols=0, desc="extracting: "
        ):
            all_results.extend(future.result())

    df = pd.DataFrame(all_results)
    if len(df) == 0:
        print("no flights with a usable descent were found")
        return
    print(f"{len(df)} flight(s) with a usable descent")
    for col, what in [("ades", "no ICAO arrival airport"), ("m_tow", "no OpenAP aircraft data")]:
        n = df[col].isna().sum()
        if n:
            print(f"  {n} flight(s) with {what}")

    if flight_list is not None:
        df = apply_flight_list(df, flight_list)

    df.to_parquet(output, index=False)
    print(f"features written to {output}")

    if model is not None:
        import lightgbm as lgb

        booster = load_model(model)
        features = booster.feature_name()
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise SystemExit(f"model wants features this run did not produce: {missing}")

        empty = [c for c in features if df[c].isna().mean() > 0.5]
        if empty:
            print(f"  WARNING: mostly empty for this run: {empty}")
            print("           use a model without those features, or supply --flight-list")

        ct_in_model = [c for c in features if c in CT_FEATURES]
        for col in ct_in_model:
            levels = booster.pandas_categorical[ct_in_model.index(col)]
            unseen = set(df[col].dropna().unique()) - set(levels)
            if unseen:
                print(f"  {col}: {len(unseen)} unseen level(s) {sorted(unseen)[:5]}")

        X = as_categorical(df[features], features)
        df = df.assign(
            tod_mass_pred=booster.predict(X, num_threads=1) * df["m_tow"]
        )
        df[["flight_id", "aircraft_type", "tod_mass_pred"]].to_csv(
            Path(output).with_suffix(".predictions.csv"), index=False
        )
        print(df[["flight_id", "aircraft_type", "tod_mass_pred"]].head().to_string(index=False))


# %%
if __name__ == "__main__":
    main()
