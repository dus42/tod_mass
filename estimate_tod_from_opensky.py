# %%
# Download arrivals from OpenSky and estimate the TOD mass of each flight.
# Requires an OpenSky Trino account (https://openskynetwork.github.io/opensky-api/trino.html),
# with credentials in ~/.config/pyopensky/settings.conf, see the README.
#   python estimate_tod_from_opensky.py --arrival-airport EHAM \
#       --start "2026-01-01 00:00" --stop "2026-01-01 11:59:59" --bounds 1,48,8,56
import click
from pathlib import Path
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


# %%
def download(start, stop, arrival_airport, bounds, workdir):
    from traffic.data import opensky

    workdir.mkdir(parents=True, exist_ok=True)
    traj_file = workdir / "trajectories.parquet"
    list_file = workdir / "flight_list.csv"

    if traj_file.exists():
        print(f"  {traj_file} already there, skipping the download")
    else:
        print(f"  querying trajectories {start} -> {stop} ...")
        kw = dict(arrival_airport=arrival_airport) if arrival_airport else {}
        if bounds:
            kw["bounds"] = tuple(float(x) for x in bounds.split(","))
        t = opensky.history(start=start, stop=stop, **kw)
        if t is None:
            raise SystemExit("the query returned nothing; widen the time range")
        t.data.to_parquet(traj_file, index=False)
        print(f"  {len(t.data):,} samples written to {traj_file}")

    if list_file.exists():
        print(f"  {list_file} already there, skipping the download")
    else:
        print("  querying the flight list ...")
        fl = opensky.flightlist(
            start=start, stop=stop, arrival_airport=arrival_airport
        )
        if fl is None or len(fl) == 0:
            print("  no flight list returned, continuing without it")
            return traj_file, None
        fl.to_csv(list_file, index=False)
        print(f"  {len(fl):,} flights written to {list_file}")

    return traj_file, list_file


# %%
def pick_model(weather, flight_list):
    suffix1 = "weather" if weather else "noweather"
    suffix3 = "_finfo" if flight_list else ""
    return f"model_tod_{suffix1}_all{suffix3}_nomassrange.txt"


# %%
@click.command()
@click.option("--start", required=True, help='e.g. "2024-06-01 06:00"')
@click.option("--stop", required=True, help='e.g. "2024-06-01 12:00"')
@click.option("--arrival-airport", default=None, help="ICAO code, e.g. EHAM")
@click.option("--bounds", default=None, help="west,south,east,north in degrees")
@click.option("--workdir", default="opensky_data", help="where the download is cached")
@click.option("--out", default="tod_mass_estimates.csv")
@click.option("--weather/--no-weather", default=False, help="interpolate ERA5 with fastmeteo")
@click.option("--local-store", default="/tmp/era5-zarr", help="fastmeteo grid cache")
@click.option("--model", default=None, help="override the automatic model choice")
@click.option("--input", "input_", default=None, help="skip the download, use this file")
@click.option("--flight-list", default=None, help="skip the download, use this flight list")
def main(
    start,
    stop,
    arrival_airport,
    bounds,
    workdir,
    out,
    weather,
    local_store,
    model,
    input_,
    flight_list,
):
    import lightgbm as lgb

    from pipeline_tod import (
        CT_FEATURES,
        MIN_DESCENT,
        apply_flight_list,
        as_categorical,
        load_model,
        traj_df_creation,
    )

    #### -----------1. get the data----------- ####
    if input_ is None:
        print("#### ----------- downloading from OpenSky ----------- ####")
        traj_file, list_file = download(
            start, stop, arrival_airport, bounds, Path(workdir)
        )
    else:
        traj_file, list_file = Path(input_), flight_list

    #### -----------2. trajectories -> one feature row per flight----------- ####
    print("\n#### ----------- extracting descent features ----------- ####")
    rows = traj_df_creation(str(traj_file), weather, local_store, MIN_DESCENT)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise SystemExit(
            "no flight had a usable descent: the trajectories may be too short, "
            "or cover only the cruise"
        )
    print(f"  {len(df):,} flight(s) with a usable descent")

    no_mtow = df.m_tow.isna()
    if no_mtow.any():
        types = ", ".join(sorted(df.aircraft_type[no_mtow].unique()))
        print(f"  {no_mtow.sum()} flight(s) dropped, no MTOW known for: {types}")
        df = df[~no_mtow]
        if len(df) == 0:
            raise SystemExit("no flight left with a known MTOW")

    if list_file is not None:
        df = apply_flight_list(df, str(list_file))

    #### -----------3. estimate the mass----------- ####
    model = model or pick_model(weather, list_file is not None)
    print(f"\n#### ----------- predicting with {model} ----------- ####")

    booster = load_model(model)
    features = booster.feature_name()
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"the model expects features this run did not produce: {missing}")

    ct_in_model = [c for c in features if c in CT_FEATURES]
    seen = {c: set(booster.pandas_categorical[ct_in_model.index(c)]) for c in ct_in_model}
    for col in ct_in_model:
        n = len(set(df[col].dropna().unique()) - seen[col])
        if n:
            print(f"  {col}: {n} value(s) unseen in training")

    df["unseen_type"] = ~df.aircraft_type.isin(seen.get("aircraft_type", set()))
    df["unseen_airport"] = ~df.ades.isin(seen.get("ades", set()))
    if "adep" in seen:
        df["unseen_airport"] |= df.adep.notna() & ~df.adep.isin(seen["adep"])

    df["tod_mass_estimate"] = (
        booster.predict(as_categorical(df[features], features), num_threads=1)
        * df["m_tow"]
    )

    cols = [
        "flight_id",
        "icao24",
        "callsign",
        "aircraft_type",
        "wtc",
        "adep",
        "ades",
        "tod_time",
        "tod_lat",
        "tod_lon",
        "tod_alt",
        "tod_mass_estimate",
        "unseen_type",
        "unseen_airport",
    ]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out, index=False)

    n_flag = int((df.unseen_type | df.unseen_airport).sum())
    print(f"\nwritten to {out}")
    print(
        f"  {len(df)} flight(s); {n_flag} carry an aircraft type or airport the model "
        f"has not seen and are the least reliable rows"
    )
    print(df[cols].head(10).to_string(index=False))


# %%
if __name__ == "__main__":
    main()
