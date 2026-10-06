# %%
# Figure 4: a sample of the measured masks.
import click
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import warnings

warnings.filterwarnings("ignore")

# %%
RECORDED, MISSING = "#2f5d8a", "#e6e9ee"
FONT = 8.2


# %%
@click.command()
@click.option("--lib", default="results/mask_library.npy")
@click.option("--out", default="latex/figures/mask_coverage.png")
@click.option("--sample", default=2000, help="how many masks to draw")
@click.option("--seed", default=0)
@click.option("--sort/--no-sort", default=True, help="order the masks by completeness")
def main(lib, out, sample, seed, sort):
    M = np.load(lib)
    nb = M.shape[1]
    rng = np.random.default_rng(seed)
    S = M[rng.choice(len(M), min(sample, len(M)), replace=False)]

    cov = S.mean(1)
    gap_end = np.array([np.max(np.where(~m)[0]) if (~m).any() else -1 for m in S])
    if sort:
        S = S[np.lexsort((gap_end, -cov))]

    plt.rcParams.update({"font.size": FONT})
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    ax.imshow(S.T, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([MISSING, RECORDED]), vmin=0, vmax=1,
              extent=[0, len(S), nb + 0.5, 0.5])

    ax.set_yticks([1, 6, 11, 16, 20])
    ax.set_yticklabels(["20  (TOD)", "15", "10", "5", "1  (ground)"])
    ax.set_ylabel("Altitude band", rotation=0, ha="right", va="bottom")
    ax.yaxis.set_label_coords(0.0, 1.03)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(S):,} randomly drawn masks" + (", sorted from most to least complete" if sort else ", in random order"))
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    ax.legend(handles=[Patch(facecolor=RECORDED, label="recorded"),
                       Patch(facecolor=MISSING, edgecolor="#b8bec8", linewidth=0.5,
                             label="missing")],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False,
              handlelength=1.2, columnspacing=1.6, fontsize=FONT)

    fig.tight_layout()
    Path(out).parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"written {out}")


# %%
if __name__ == "__main__":
    main()
