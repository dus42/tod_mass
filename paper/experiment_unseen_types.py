# %%
# Unfamiliar aircraft types (Tables 4-5).
import click
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor as lgbreg

from pipeline_tod import get_oew

import warnings

warnings.filterwarnings("ignore")

# %%
MIN_FLIGHTS = 100
DROP_ALWAYS = ["min_tow_ch", "max_tow_ch", "mean_cruise_altitude"]
DROP_ALWAYS += ["max_rod"] + [f"mean_rod_s{i}" for i in range(1, 6)]
WEATHER_FEATURES = [
    "des_tas_min",
    "des_tas_mean",
    "des_tas_max",
    "tod_tas",
    "des_dist",
    "des_temp",
]
FINFO_FEATURES = ["adep", "flight_duration", "lat_dep", "lon_dep", "distance_ap"]
CT_FEATURES = [
    "adep",
    "ades",
    "wtc",
    "aircraft_type",
    "day_of_year",
    "time_of_day_arrival",
]
DROP_COLS = [
    "date",
    "distance",
    "callsign",
    "name_adep",
    "name_ades",
    "actual_offblock_time",
    "arrival_time",
    "fuel_spent",
    "taxiout_time",
    "country_code_ades",
    "country_code_adep",
    "airline",
    "flown_distance",
]


def classify_time(hour, nb_of_partition=24):
    range_of_hour = 24 / nb_of_partition

    return hour // range_of_hour


def make_model():
    return lgbreg(
        reg_alpha=0.17,
        reg_lambda=0.03,
        num_leaves=215,
        learning_rate=0.08,
        n_estimators=800,
        colsample_bytree=0.5,
        importance_type="gain",
        force_col_wise=True,
        n_jobs=1,
    )


# %%
@click.command()
@click.option("--frame", default="data/todmass_full.csv")
@click.option("--out", default="results/results_tod_unseen_types.csv")
@click.option("--target", type=click.Choice(["mass", "ratio"]), default="mass",
              help="regress the TOD mass in kg, or its fraction of the MTOW")
def main(frame, out, target):
    #### -----------the same frame the released models are trained on----------- ####
    dff = pd.read_csv(frame)
    dff["arrival_time"] = pd.to_datetime(dff["arrival_time"])
    dff["time_of_day_arrival"] = dff["arrival_time"].dt.hour.apply(classify_time)
    dff["date"] = pd.to_datetime(dff["date"], format="%Y-%m-%d")
    dff["day_of_year"] = dff["date"].dt.dayofyear
    dff = dff.astype(
        {
            col: "float32"
            for col in dff.drop(columns="flight_id")
            .select_dtypes(include="float")
            .columns
        }
    )
    dff = dff.astype({"day_of_year": "int16", "time_of_day_arrival": "int8"})
    dff = dff.dropna(subset=["tod_mass"])

    oew = {ac: get_oew(ac) for ac in dff.aircraft_type.unique()}
    keep = ~(
        (dff.tow > 1.05 * dff.m_tow)
        | (dff.tod_mass < dff.aircraft_type.map(oew))
    )
    dff = dff[keep]
    dff["oew"] = dff.aircraft_type.map(oew).astype("float32")
    counts = dff.aircraft_type.value_counts()
    dff = dff[dff.aircraft_type.isin(counts[counts >= MIN_FLIGHTS].index)]
    dff = dff.drop(columns=[c for c in DROP_COLS + DROP_ALWAYS if c in dff])

    #### -----------split the types, alternating by size----------- ####
    by_size = dff.groupby("aircraft_type").m_tow.first().sort_values()
    seen_types = list(by_size.index[0::2])
    unseen_types = list(by_size.index[1::2])
    print(f"#### ----------- target: {target} ----------- ####")
    print("#### ----------- type split ----------- ####")
    print(f"  trained on {len(seen_types)}: {', '.join(seen_types)}")
    print(f"  held out   {len(unseen_types)}: {', '.join(unseen_types)}")

    seen = dff[dff.aircraft_type.isin(seen_types)]
    unseen = dff[dff.aircraft_type.isin(unseen_types)].copy()
    print(f"  flights: {len(seen):,} trained, {len(unseen):,} held out")

    seen_train, seen_test = train_test_split(seen, test_size=0.2, random_state=42)

    #### -----------baselines, calibrated on the training types only----------- ####
    frac = (seen_train.tod_mass / seen_train.m_tow).mean()
    wtc_mean = seen_train.groupby("wtc").tod_mass.mean()
    for d in (unseen, seen_test):
        d["MTOW fraction"] = frac * d.m_tow
        d["WTC mean"] = d.wtc.map(wtc_mean)
    print(f"\n  calibrated MTOW fraction: {frac:.3f}")

    #### -----------train one model per feature set----------- ####
    rows = []
    for weather, finfo in [(False, False), (True, False), (False, True), (True, True)]:
        label = "ADS-B" + (" + weather" if weather else "") + (" + flight info" if finfo else "")
        drop = ([] if weather else WEATHER_FEATURES) + ([] if finfo else FINFO_FEATURES)
        ct = [c for c in CT_FEATURES if c not in drop]

        def prep(d):
            d = d.drop(columns=[c for c in drop if c in d], errors="ignore")
            d = d.drop(columns=[c for c in ["tod_mass", "flight_id", "tow",
                                            "MTOW fraction", "WTC mean"] if c in d])
            for c in ct:
                d[c] = d[c].astype("category")
            return d

        ytr = (
            seen_train.tod_mass / seen_train.m_tow
            if target == "ratio"
            else seen_train.tod_mass
        )
        Xtr = prep(seen_train.copy())
        model = make_model()
        model.fit(Xtr, ytr, categorical_feature=ct)

        for name, d in [("seen types", seen_test), ("unseen types", unseen)]:
            X = prep(d.copy())[Xtr.columns]
            pred = model.predict(X, num_threads=1)
            d[label] = pred * d.m_tow if target == "ratio" else pred
            rows.append(
                {
                    "subset": name,
                    "method": label,
                    "n": len(d),
                    "rmse": rmse(d.tod_mass, d[label]),
                    "mape": 100 * mape(d.tod_mass, d[label]),
                }
            )
        print(f"  trained {label} ({Xtr.shape[1]} features)")

    for name, d in [("seen types", seen_test), ("unseen types", unseen)]:
        for b in ["MTOW fraction", "WTC mean"]:
            rows.append(
                {
                    "subset": name,
                    "method": b,
                    "n": len(d),
                    "rmse": rmse(d.tod_mass, d[b]),
                    "mape": 100 * mape(d.tod_mass, d[b]),
                }
            )

    #### -----------report----------- ####
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    order = ["MTOW fraction", "WTC mean", "ADS-B", "ADS-B + weather",
             "ADS-B + flight info", "ADS-B + weather + flight info"]
    for name in ["seen types", "unseen types"]:
        sub = res[res.subset == name].set_index("method")
        print(f"\n#### ----------- {name}: {int(sub.n.iloc[0]):,} flights ----------- ####")
        print(f"{'method':<32}{'RMSE [kg]':>11}{'MAPE [%]':>10}")
        for m in order:
            if m in sub.index:
                print(f"{m:<32}{sub.loc[m,'rmse']:>11,.0f}{sub.loc[m,'mape']:>10.2f}")

    #### -----------per held-out type----------- ####
    print(f"\n#### ----------- per held-out type, ADS-B model vs MTOW fraction ----------- ####")
    print(f"{'type':<7}{'n':>8}{'true mean':>11}{'MTOW frac':>11}{'model':>11}{'frac MAPE':>11}{'model MAPE':>12}")
    for ac, g in unseen.groupby("aircraft_type"):
        print(
            f"{ac:<7}{len(g):>8,}{g.tod_mass.mean():>11,.0f}"
            f"{g['MTOW fraction'].mean():>11,.0f}{g['ADS-B'].mean():>11,.0f}"
            f"{100*mape(g.tod_mass, g['MTOW fraction']):>11.2f}"
            f"{100*mape(g.tod_mass, g['ADS-B']):>12.2f}"
        )
    per = []
    for name, d in [("seen types", seen_test), ("unseen types", unseen)]:
        for ac, g in d.groupby("aircraft_type"):
            per.append({"subset": name, "aircraft_type": ac, "n": len(g),
                        "mape": 100 * mape(g.tod_mass, g["ADS-B"]),
                        "mape_mtow_fraction": 100 * mape(g.tod_mass, g["MTOW fraction"])})
    per_out = out.replace(".csv", "_pertype.csv")
    pd.DataFrame(per).to_csv(per_out, index=False)
    print(f"\nwritten to {out} and {per_out}")


# %%
if __name__ == "__main__":
    main()
