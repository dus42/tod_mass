# %%
# Stage 4: train the four models (Tables 2-3, Figure 2 predictions).
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor as lgbreg

from pipeline_tod import get_oew

# %%
#### -----------Configuration----------- ####
# challenge-derived / non-descent features removed for every variant
DROP_ALWAYS = ["min_tow_ch", "max_tow_ch", "mean_cruise_altitude"]

# Tried and not kept. features_tod_full.py still extracts these, so they stay in
# traj_tod_full.parquet and can be brought back by deleting a line here, but they earn
# little: the five sector rates carry 2.5 % of the split gain between them and
# max_rod 0.2 %, against 4.8 % for the two groundspeed features that were kept.
DROP_ALWAYS += ["max_rod"] + [f"mean_rod_s{i}" for i in range(1, 6)]

# supplied by fastmeteo in the challenge, needs local weather storage to reproduce
WEATHER_FEATURES = [
    "des_tas_min",
    "des_tas_mean",
    "des_tas_max",
    "tod_tas",
    "des_dist",
    "des_temp",
]

# open, but must be retrieved separately from the trajectory
FINFO_FEATURES = [
    "adep",
    "flight_duration",
    "lat_dep",
    "lon_dep",
    "distance_ap",
]

# aircraft types with fewer labelled flights than this are not modelled: under a
# random split such a type is either never tested, or tested with no training
# examples of its own. Drops 5 types / 11 flights (0.003 % of the data).
MIN_FLIGHTS = 100

TAG = "nomassrange"

# categoricals are kept as their own values (ICAO codes, type designators) and given
# the pandas `category` dtype. LightGBM stores those levels inside the model file, so
# a released model re-encodes new data by itself -- unlike pd.factorize, whose codes
# depend on row order in the training CSV.
CT_FEATURES = [
    "adep",
    "ades",
    "wtc",
    "aircraft_type",
    "day_of_year",
    "time_of_day_arrival",
]


# %%
#### -----------Functions here----------- ####
def classify_time(hour, nb_of_partition=24):
    range_of_hour = 24 / nb_of_partition

    return hour // range_of_hour


# %%
#### -----------Features Processing----------- ####
# built by all_features_tod.py from the full release, without factorising
print("#### -----------Features Processing----------- ###")
dff_all = pd.read_csv("data/todmass_full.csv")

n_released = len(pd.read_csv("data/flight_list.csv", usecols=["flight_id"]))
n_lab = dff_all.tod_mass.notna().sum()
print(f"  {n_lab:,} of {n_released:,} released flights carry a TOD-mass label "
      f"({100 * n_lab / n_released:.1f} %)")

# Convert 'actual_offblock_time' to datetime
dff_all["arrival_time"] = pd.to_datetime(dff_all["arrival_time"])

# Extract the hour
dff_all["time_of_day_arrival"] = dff_all["arrival_time"].dt.hour.apply(classify_time)

# Extract the day
dff_all["date"] = pd.to_datetime(dff_all["date"], format="%Y-%m-%d")
dff_all["day_of_year"] = dff_all["date"].dt.dayofyear

# Convert float64 to float32, might boost the training
dff_all = dff_all.astype(
    {
        col: "float32"
        for col in dff_all.drop(columns="flight_id")
        .select_dtypes(include="float")
        .columns
    }
)

# categorical levels have to line up with what the pipeline produces, so the two
# time features are whole numbers rather than floats
dff_all = dff_all.astype({"day_of_year": "int16", "time_of_day_arrival": "int8"})

DROP_COLS = [
        "date",
        "distance",
        "callsign",
        "name_adep",
        "name_ades",
        "actual_offblock_time",
        "arrival_time",
        "tow",
        "fuel_spent",
        "taxiout_time",
        "country_code_ades",
        "country_code_adep",
        "airline",
        "flown_distance",
]
dff_all["tow_ref"] = dff_all["tow"]
dff_all = dff_all.drop(columns=[c for c in DROP_COLS + DROP_ALWAYS if c in dff_all])

#### -----------Discard labels that cannot be right----------- ####
oew = {ac: get_oew(ac) for ac in dff_all.aircraft_type.unique()}
over_mtow = dff_all.tow_ref > 1.05 * dff_all.m_tow
under_oew = dff_all.tod_mass < dff_all.aircraft_type.map(oew)

print("#### -----------Physical check----------- ###")
print(f"  takeoff weight above MTOW      {int(over_mtow.sum()):>7,}")
print(f"  TOD mass below empty weight    {int(under_oew.sum()):>7,}")
worst = dff_all[under_oew].aircraft_type.value_counts().head(3)
tot = dff_all.aircraft_type.value_counts()
for ac, n in worst.items():
    print(f"    {ac:<6} {n:>6,} of {int(tot[ac]):>7,}  ({100*n/tot[ac]:.0f} %)")
keep = ~(over_mtow | under_oew)
print(f"  dropped {int((~keep).sum()):,} of {len(dff_all):,} "
      f"({100*(~keep).mean():.2f} %), {int(keep.sum()):,} remain")
dff_all = dff_all[keep].drop(columns=["tow_ref"])

#### -----------Retain only the modellable aircraft types----------- ####
counts = dff_all.aircraft_type.value_counts()
dropped = counts[counts < MIN_FLIGHTS]
retained = counts[counts >= MIN_FLIGHTS]

print(f"#### -----------Aircraft types (min {MIN_FLIGHTS} flights)----------- ###")
for ac, n in dropped.items():
    print(f"  dropping {ac:<6} {n:>3} flights")
print(
    f"  retained {len(retained)} of {len(counts)} types, "
    f"{retained.sum():,} of {len(dff_all):,} flights "
    f"({100 * dropped.sum() / len(dff_all):.3f} % dropped)"
)

dff_all = dff_all[dff_all.aircraft_type.isin(retained.index)]

results = []

# paper table order: (weather, flight information)
dff_all["oew"] = dff_all.aircraft_type.map(oew).astype("float32")

for weather, finfo in [(True, True), (True, False), (False, True), (False, False)]:

    suffix1 = "weather" if weather else "noweather"
    suffix3 = "_finfo" if finfo else ""
    name = f"{suffix1}_all{suffix3}_{TAG}"
    print(f"\n#### ----------- {name} ----------- ####")

    dff = dff_all.copy()

    ct_features = list(CT_FEATURES)
    if not weather:
        dff = dff.drop(columns=WEATHER_FEATURES)
    if not finfo:
        dff = dff.drop(columns=FINFO_FEATURES)
        ct_features.remove("adep")

    dff[ct_features] = dff[ct_features].astype("category")

    # %%
    ############### Train-Test check #################
    # initialize data
    labelled = dff.dropna(subset=["tod_mass"])
    X = labelled.drop(columns=[c for c in ["tod_mass", "dataset"] if c in labelled])
    y = labelled["tod_mass"] / labelled["m_tow"]
    mass = labelled["tod_mass"]
    dfeat = pd.DataFrame(
        list(X.drop(columns=["flight_id"]).columns), columns=["col_name"]
    )

    X_train, X_test, y_train, y_test, _, mass_test = train_test_split(
        X,
        y,
        mass,
        test_size=0.2,
        random_state=42,
    )
    X_train = X_train.drop(columns=["flight_id"])

    model = lgbreg(
        reg_alpha=0.17,
        reg_lambda=0.03,
        num_leaves=215,
        learning_rate=0.08,
        n_estimators=800,
        colsample_bytree=0.5,
        importance_type="gain",
        force_col_wise=True,
        # single-threaded on purpose: num_threads is written into the model file, and
        # prediction runs inside traffic's .pipe()/.eval() worker processes, where a
        # multi-threaded booster would oversubscribe the CPU
        n_jobs=1,
    )
    # train the model
    model.fit(
        X_train,
        y_train,
        categorical_feature=ct_features,
    )
    # make the prediction using the resulting model
    ratio_pred = model.predict(
        X_test.drop(columns=["flight_id"]), categorical_feature=ct_features
    )
    preds = ratio_pred * X_test["m_tow"].values
    X_test = X_test.assign(mass_true=mass_test, mass_pred=preds)

    per_ac = (
        X_test.groupby("aircraft_type", observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n_test": len(g),
                    "rmse": rmse(g.mass_true, g.mass_pred),
                    "mape": 100
                    * mean_absolute_percentage_error(g.mass_true, g.mass_pred),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_ac["n_flights"] = per_ac.aircraft_type.map(counts).astype(int)
    per_ac = per_ac.sort_values("n_flights", ascending=False)
    per_ac.to_csv(f"results/results_tod_pertype_{name}.csv", index=False)

    model.booster_.save_model(f"../models/model_tod_{name}.txt")
    features = model.feature_name_
    pd.DataFrame(features, columns=["feature"]).to_csv(
        f"results/features_tod_{name}.csv", index=False
    )

    dfeat = dfeat.assign(imp=model.feature_importances_)
    dfeat = dfeat.assign(pct=100 * dfeat.imp / dfeat.imp.sum())
    print(
        dfeat.sort_values(by="pct", ascending=False)[["col_name", "pct"]].to_string(
            index=False, float_format=lambda v: f"{v:6.2f}"
        )
    )

    # %%
    X_test = X_test.assign(err=X_test.mass_true - X_test.mass_pred)
    X_test[["flight_id", "aircraft_type", "mass_true", "mass_pred"]].to_parquet(
        f"results/results_tod_testpred_{name}.parquet", index=False
    )

    mae = X_test.err.abs().mean()
    mape = 100 * mean_absolute_percentage_error(
        X_test.mass_true.values, X_test.mass_pred.values
    )
    rmse_ = rmse(X_test.mass_true.values, X_test.mass_pred.values)

    print(f"RMSE: {rmse_:.0f} kg   MAPE: {mape:.2f} %   MAE: {mae:.0f} kg")
    results.append(
        {
            "weather": weather,
            "finfo": finfo,
            "n_features": len(features),
            "rmse": rmse_,
            "mape": mape,
            "mae": mae,
        }
    )

# %%
#### -----------Summary (paper Table 1)----------- ####
res = pd.DataFrame(results)
res.to_csv(f"results/results_tod_{TAG}.csv", index=False)

print("\n\n#### ----------- Summary ----------- ####")
print(f"{'ADS-B':^6}{'Weather':^9}{'Flight info':^13}{'RMSE [kg]':>11}{'MAPE [%]':>10}")
for r in results:
    tick = lambda b: "  v  " if b else "  x  "
    print(
        f"{'  v  ':^6}{tick(r['weather']):^9}{tick(r['finfo']):^13}"
        f"{r['rmse']:>11,.0f}{r['mape']:>10.2f}"
    )

# %%
