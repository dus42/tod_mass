# %%
# Figure 3: MAPE per aircraft type for the four feature sets.
# %% plots
import click
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline_tod import get_mtow, get_wtc

import warnings

warnings.filterwarnings("ignore")

# %%
FEATURE_SETS = {
    "noweather_all": "ADS-B",
    "weather_all": "ADS-B + weather",
    "noweather_all_finfo": "ADS-B + flight info",
    "weather_all_finfo": "ADS-B + weather + flight info",
}
HUE_ORDER = list(FEATURE_SETS.values())

sns.set_context(
    "notebook",
    rc={
        "axes.labelsize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    },
)


# %%
@click.command()
@click.option("--tag", default="nomassrange")
@click.option("--outdir", default="latex/figures")
def main(tag, outdir):
    frames = []
    for suffix, label in FEATURE_SETS.items():
        f = f"results/results_tod_pertype_{suffix}_{tag}.csv"
        if not Path(f).exists():
            print(f"  missing {f}, skipping")
            continue
        frames.append(pd.read_csv(f).assign(features=label))
    df = pd.concat(frames, ignore_index=True)

    df["ac"] = df["aircraft_type"].str.upper()
    df["wtc"] = df["ac"].map(lambda ac: get_wtc(get_mtow(ac)))
    df = df.sort_values(by="ac").reset_index(drop=True)

    for pl, title in zip(
        ["mape", "rmse"],
        ["MAPE [%]", "RMSE [kg]"],
    ):
        g = sns.catplot(
            data=df,
            kind="bar",
            x="ac",
            y=pl,
            hue="features",
            errorbar=None,
            palette="tab10",
            alpha=0.9,
            height=4,
            aspect=2.4,
            order=sorted(df["ac"].unique()),
            hue_order=HUE_ORDER,
        )
        g.despine(left=True)
        g.ax.set_ylim(0, np.nanmax(df[pl].to_numpy()) * 1.05)
        g.set_xticklabels(g.ax.get_xticklabels(), rotation=45, ha="right", fontsize=13)
        g.set_axis_labels("", title)

        sns.move_legend(
            g,
            "lower center",
            bbox_to_anchor=(0.42, -0.13),
            ncol=4,
            title=None,
            frameon=False,
        )
        g.ax.grid(axis="y", alpha=0.3)
        g.ax.set_axisbelow(True)

        g.tight_layout()
        Path(outdir).mkdir(exist_ok=True)
        plt.savefig(f"{outdir}/{pl}_per_ac.png", dpi=200, bbox_inches="tight")
        plt.close(g.fig)
        print(f"written {outdir}/{pl}_per_ac.png")


# %%
if __name__ == "__main__":
    main()
