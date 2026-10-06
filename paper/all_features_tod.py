# %%
# Stage 3: flight list + descent features -> data/todmass_full.csv (the modelling frame).
# %% merge trajectory features with the flight information, for the full dataset
import pandas as pd
import openap
from traffic.data import airports

from pipeline_tod import get_mtow

import warnings

warnings.filterwarnings("ignore")

# %%
df = pd.read_csv("data/flight_list.csv")
print(f"flight list: {len(df):,} flights")

ap_df = pd.DataFrame(
    airports.data.iloc[
        :,
        2:5,
    ].values,
    columns=["icao", "lat", "lon"],
)
df = df.merge(ap_df, left_on="adep", right_on="icao", how="left")
df = df.rename(columns={"lat": "lat_dep", "lon": "lon_dep"}).drop(columns="icao")
df = df.merge(ap_df, left_on="ades", right_on="icao", how="left")
df = df.rename(columns={"lat": "lat_arr", "lon": "lon_arr"}).drop(columns="icao")
df["distance_ap"] = df.apply(
    lambda row: openap.aero.distance(
        row.lat_dep, row.lon_dep, row.lat_arr, row.lon_arr
    ),
    axis=1,
)

# %%
# read the trajectory data
df_traj = pd.read_parquet("data/traj_tod_full.parquet")
print(f"trajectory features: {len(df_traj):,} flights")
dff = pd.merge(df, df_traj, on="flight_id", how="inner")

# %%
mtow = {ac: get_mtow(ac) for ac in dff.aircraft_type.unique()}
dff["m_tow"] = dff.aircraft_type.map(mtow)

dff = dff.dropna(subset=["tod_mass"])
# only the columns the models and experiments use
UNUSED = ["callsign", "name_adep", "country_code_adep", "name_ades", "country_code_ades",
          "actual_offblock_time", "airline", "taxiout_time", "flown_distance",
          "max_rod"] + [f"mean_rod_s{i}" for i in range(1, 6)]
dff = dff.drop(columns=[c for c in UNUSED if c in dff])
print(f"with a TOD-mass label: {len(dff):,} flights, {dff.aircraft_type.nunique()} types")
dff.to_csv("data/todmass_full.csv", index=False)
print("written to todmass_full.csv")

# %%
