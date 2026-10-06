# %%
# Incomplete descents (Table 6), measured and synthetic masks.
import click
from pathlib import Path
import glob
import shutil
import numpy as np
import pandas as pd
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error as mape

import pipeline_tod as P
from pipeline_tod import get_oew

import warnings

warnings.filterwarnings("ignore")

# %%
NB = 20  # bands of the altitude lost
MIN_FLIGHTS = 100  # the type filter the models are trained under
MODEL = "model_tod_noweather_all_nomassrange.txt"
WORK = Path("/tmp/tod_missing")

SUPPLIED = ["ades", "lat_arr", "lon_arr"]

_LIB = []  # mask library, shared with the worker processes


# %%
#### -----------The reference set: test-split flights only----------- ####
def reference_set():
    d = pd.read_csv("data/todmass_full.csv").dropna(subset=["tod_mass"])
    oew = {ac: get_oew(ac) for ac in d.aircraft_type.unique()}
    d = d[~((d.tow > 1.05 * d.m_tow) | (d.tod_mass < d.aircraft_type.map(oew)))]
    counts = d.aircraft_type.value_counts()
    d = d[d.aircraft_type.isin(counts[counts >= MIN_FLIGHTS].index)]
    _, test = train_test_split(d, test_size=0.2, random_state=42)
    return test.set_index("flight_id")


# %%
#### -----------Mask library harvested from real trajectories----------- ####
def harvest_masks(pattern, min_drop=8000):
    """Coverage patterns of real descents, measured on the raw recording.

    Accepts a glob, so the masks are harvested from the raw recordings themselves
    (`24challenge_data/2022-0*.parquet`): they come from the same population as the
    flights being masked, which is what makes them representative."""
    masks = []
    for path in sorted(glob.glob(pattern)) or [pattern]:
        masks += _harvest_one(path, min_drop)
    return masks


def _harvest_one(path, min_drop):
    raw = P.normalise(pd.read_parquet(path))
    raw["timestamp"] = raw["timestamp"].astype("datetime64[ns, UTC]")
    if "flight_id" in raw.columns:
        raw["fid"] = raw["flight_id"]
    else:
        raw["fid"] = (
            raw.icao24.astype(str).str.lower()
            + "_"
            + raw.callsign.astype(str).str.strip()
        )
    masks = []
    for _, g in raw.groupby("fid"):
        g = g.sort_values("timestamp").dropna(subset=["altitude"]).reset_index(drop=True)
        if len(g) < 20:
            continue
        des = g.loc[int(g.altitude.idxmax()) :]
        if len(des) < 10:
            continue
        top, bot = des.altitude.max(), des.altitude.min()
        if top - bot < min_drop:
            continue
        idx = np.clip(((top - des.altitude) / (top - bot) * NB).astype(int), 0, NB - 1)
        m = np.zeros(NB, bool)
        m[np.unique(idx)] = True
        if not m.all():
            masks.append(m)
    del raw
    return masks


def synthetic(kind, share, rng):
    k = int(round(share * NB))
    m = np.ones(NB, bool)
    if k == 0:
        return m
    if kind == "top":
        m[:k] = False
    elif kind == "bottom":
        m[-k:] = False
    else:
        m[rng.choice(NB, k, replace=False)] = False
    return m


# %%
def mask_raw_flight(g, mask):
    """drop raw samples of the descent that fall in an uncovered altitude band.

    The raw recording carries no phase labels, so the descent is taken as everything
    after the highest sample, which is what the mask harvesting also assumes."""
    g = g.sort_values("timestamp").reset_index(drop=True)
    if len(g) < 20:
        return None
    des = g.loc[int(g.altitude.idxmax()) :]
    top, bot = des.altitude.max(), des.altitude.min()
    if top - bot < 5000:
        return None
    band = np.clip(((top - des.altitude) / (top - bot) * NB).astype(int), 0, NB - 1)
    keep = pd.Series(True, index=g.index)
    keep.loc[des.index] = mask[band.values]
    return g[keep]


# %%
def build_day(path, ids, types, spec, seed, outdir, lib):
    """write one masked raw day, ready for the pipeline"""
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(path, filters=[("flight_id", "in", ids)])
    if len(df) == 0:
        return None
    df = df.merge(types, on="flight_id", how="inner")
    out = []
    for _, g in df.groupby("flight_id"):
        if spec is None:
            out.append(g)
            continue
        m = lib[rng.integers(len(lib))] if spec == "lib" else synthetic(*spec, rng)
        gm = mask_raw_flight(g, m)
        if gm is not None and len(gm) > 20:
            out.append(gm)
    if not out:
        return None
    f = Path(outdir) / Path(path).name
    pd.concat(out).to_parquet(f, index=False)
    return str(f)


# %%
@click.command()
@click.option("--days", default=12,
              help="how many raw days to draw test flights from, spread evenly across the "
                   "year; twelve is one per month, which keeps the sample seasonally representative")
@click.option("--only", default=None, help="run a single condition, e.g. 'complete'")
@click.option("--masks-from", default="data/24challenge_data/2022-0[37]-*.parquet",
              help="glob of raw recordings to harvest coverage patterns from")
@click.option("--masks-cache", default="results/mask_library.npy",
              help="load the harvested masks from here if it exists, else write them "
                   "there. Harvesting is the slowest part of the run and its result "
                   "does not change, so repeated runs should reuse it.")
@click.option("--repeats", default=1,
              help="run each condition this many times with different mask draws, to "
                   "separate the effect of the masking from the luck of the draw. The "
                   "complete condition uses no random numbers and is unaffected.")
@click.option("--out", default="results/results_tod_missing_descent_spread.csv")
@click.option("--workers", default=4, help="raw days held in memory at once")
@click.option("--seed", default=42)
def main(days, only, masks_from, masks_cache, repeats, out, workers, seed):
    cache = Path(masks_cache) if masks_cache else None
    if cache and cache.exists():
        lib = list(np.load(cache))
        print(f"#### ----------- setup ----------- ####")
        print(f"  mask library read from {cache}")
    else:
        lib = harvest_masks(masks_from)
        if cache:
            np.save(cache, np.array(lib))
        print("#### ----------- setup ----------- ####")
    print(f"  {len(lib)} incomplete descents harvested, coverage "
          f"{min(m.mean() for m in lib):.2f} .. {max(m.mean() for m in lib):.2f}")

    ref = reference_set()
    every = sorted(glob.glob("data/24challenge_data/*.parquet"))
    idx = np.linspace(0, len(every) - 1, days).round().astype(int)
    files = [every[i] for i in dict.fromkeys(idx)]
    ids = ref.index.tolist()  # int64, matching the raw parquet column
    types = ref.reset_index()[["flight_id", "aircraft_type"]]
    print(f"  test-split reference set: {len(ref):,} flights, drawing from "
          f"{len(files)} raw days spread across the year:")
    print("   ", ", ".join(Path(f).stem for f in files))

    booster = P.load_model(MODEL)
    feats = booster.feature_name()

    conditions = [("complete", None)]
    for share in (0.1, 0.25, 0.5):
        for kind in ("top", "bottom", "random"):
            conditions.append((f"{kind} {int(share * 100)}%", (kind, share)))
    conditions.append(("measured", "lib"))
    if only:
        conditions = [c for c in conditions if c[0] == only]
        if not conditions:
            raise SystemExit(f"no condition named {only!r}")

    rows = []
    for name, spec in conditions:
        for rep in range(1 if spec is None else repeats):
            rep_seed = seed + 1000 * rep
            outdir = WORK / name.replace(" ", "_").replace("%", "")
            shutil.rmtree(outdir, ignore_errors=True)
            outdir.mkdir(parents=True)

            made = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [
                    ex.submit(build_day, f, ids, types, spec, rep_seed + i, str(outdir), lib)
                    for i, f in enumerate(files)
                ]
                for fut in as_completed(futs):
                    r = fut.result()
                    if r:
                        made.append(r)

            res = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [
                    ex.submit(P.traj_df_creation, f, False, "/tmp/era5-zarr", P.MIN_DESCENT)
                    for f in made
                ]
                for fut in as_completed(futs):
                    res.extend(fut.result())

            d = pd.DataFrame(res)
            if len(d) == 0:
                print(f"  {name:<14} no flight survived")
                continue
            d = d.set_index("flight_id")
            for c in SUPPLIED:
                d[c] = ref[c].reindex(d.index)
            d["tod_mass"] = ref["tod_mass"].reindex(d.index)
            d = d.dropna(subset=["tod_mass"] + SUPPLIED)

            X = P.as_categorical(d[feats], feats)
            pred = booster.predict(X, num_threads=1) * d["m_tow"]
            rows.append({"condition": name, "repeat": rep, "seed": rep_seed,
                         "flights": len(d),
                         "rmse": rmse(d.tod_mass, pred),
                         "mape": 100 * mape(d.tod_mass, pred)})
            tag = f" [{rep + 1}/{repeats}]" if repeats > 1 and spec is not None else ""
            print(f"  {name:<14}{tag:<8} {len(d):>5} flights   "
                  f"RMSE {rows[-1]['rmse']:>7,.0f}   MAPE {rows[-1]['mape']:>5.2f} %")
            shutil.rmtree(outdir, ignore_errors=True)

    r = pd.DataFrame(rows)
    r.to_csv(out, index=False)
    print("\n#### ----------- degradation ----------- ####")
    if "complete" in set(r.condition):
        base = r[r.condition == "complete"].iloc[0]
        print(f"{'condition':<16}{'flights':>8}{'RMSE [kg]':>11}"
              f"{'MAPE [%]':>10}{'vs complete':>13}")
        for x in r.itertuples():
            print(f"{x.condition:<16}{x.flights:>8,}{x.rmse:>11,.0f}"
                  f"{x.mape:>10.2f}{x.mape - base.mape:>+13.2f}")
    else:
        print(r.to_string(index=False))

    if repeats > 1:
        g = r.groupby("condition").mape.agg(["mean", "std", "min", "max", "count"])
        print("\n#### ----------- spread over mask draws ----------- ####")
        print(f"{'condition':<16}{'n':>3}{'mean':>8}{'sd':>7}{'min':>8}{'max':>8}")
        for c, x in g.iterrows():
            sd = "  --  " if x["count"] < 2 else f"{x['std']:>7.3f}"
            print(f"{c:<16}{int(x['count']):>3}{x['mean']:>8.2f}{sd}"
                  f"{x['min']:>8.2f}{x['max']:>8.2f}")
    print(f"\nwritten to {out}")


# %%
if __name__ == "__main__":
    main()
