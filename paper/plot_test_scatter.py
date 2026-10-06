# %%
# Figure 2: predicted against reference mass.
import click
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error

import warnings

warnings.filterwarnings("ignore")


# %%
def scatter(d, out):
    d = d.assign(err=d.mass_true - d.mass_pred)
    pred, true = np.round(d.mass_pred) / 1000, np.round(d.mass_true) / 1000
    H, xedges, yedges = np.histogram2d(pred, true, bins=17)
    Hmasked = np.ma.masked_where(H == 0, H)

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.grid(True, which="major")
    ax.scatter(pred, true, c="tab:blue", s=5)
    XX, YY = np.meshgrid(0.5 * (xedges[:-1] + xedges[1:]), 0.5 * (yedges[:-1] + yedges[1:]))
    contour = plt.contourf(XX, YY, Hmasked.T, levels=15, alpha=0.5, cmap="viridis")
    cb = plt.colorbar(contour)
    cb.ax.set_title("Number of\npoints", fontsize=10, pad=8)
    lo = min(d.mass_true.min(), d.mass_pred.min()) / 1000
    hi = max(d.mass_true.max(), d.mass_pred.max()) / 1000
    ax.plot([lo, hi], [lo, hi], color="tab:red", label="error = 0")
    ax.set_xlabel("Predicted mass [t]")
    ax.set_ylabel("Reference mass [t]", rotation=0, ha="left")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_label_coords(-0.095, 1.02)

    mape = 100 * mean_absolute_percentage_error(d.mass_true, d.mass_pred)
    stats = (
        f"RMSE:   {int(np.round(rmse(d.mass_true, d.mass_pred)))} kg\n"
        f"MAE:   {int(np.round(d.err.abs().mean()))} kg\n"
        f"ME:    {int(np.round(d.err.median()))} kg\n"
        f"MAPE:   {mape:.2f} % "
    )
    ax.text(0.95, 0.33, stats, ha="right", transform=ax.transAxes, fontsize=12,
            fontfamily="monospace", verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.2"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"written {out}")


# %%
@click.command()
@click.option("--tag", default="nomassrange")
@click.option("--outdir", default="latex/figures")
def main(tag, outdir):
    Path(outdir).mkdir(exist_ok=True)
    for f in sorted(Path("results").glob(f"results_tod_testpred_*_{tag}.parquet")):
        name = f.stem.removeprefix("results_tod_testpred_").removesuffix(f"_{tag}")
        scatter(pd.read_parquet(f), f"{outdir}/test_train_tod_{name}.png")


# %%
if __name__ == "__main__":
    main()
