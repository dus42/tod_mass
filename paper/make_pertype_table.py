# %%
# Table 3 as LaTeX.
import click
import pandas as pd

from pipeline_tod import get_mtow, get_wtc

import warnings

warnings.filterwarnings("ignore")


# %%
def _block(rows_df, start):
    """one half of the table, as a minipage"""
    rows = []
    for i, r in enumerate(rows_df.itertuples(), start):
        line = (
            f"    {i} & {r.aircraft_type} & {r.wtc} & {int(r.n_flights):,} "
            f"& {int(r.n_test):,} & {int(round(r.rmse)):,} & {r.mape:.2f} \\\\".replace(",", "{,}")
        )
        rows.append(("    \\myrowcolour\n" if (i - start) % 2 else "") + line)
    return (
        "    \\begin{minipage}[t]{0.48\\textwidth}\n"
        "    \\vspace{0pt}\n"
        "    \\centering\n    \\small\n    \\setlength{\\tabcolsep}{4pt}\n"
        "    \\begin{tabular}{rc|crrrr}\n    \\toprule\n"
        "    && & \\multicolumn{2}{c}{\\textbf{Flights}} "
        "& \\multicolumn{2}{c}{\\textbf{Test-set error}}\\\\\n"
        "    \\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "    & \\thead{Aircraft\\\\type} & \\thead{WTC} & \\thead{total} & \\thead{test} "
        "& \\thead{RMSE\\\\{}[kg]} & \\thead{MAPE\\\\{}[\\%]} \\\\\n"
        "    \\midrule\n" + "\n".join(rows) + "\n"
        "    \\bottomrule\n    \\end{tabular}\n    \\end{minipage}"
    )


def tex_table(d, label, caption):
    half = (len(d) + 1) // 2
    return (
        "\\begin{table}[htbp]\n"
        "    \\centering\n"
        f"    \\caption{{{caption}}}\n"
        + _block(d.iloc[:half], 1)
        + "\n    \\hfill\n"
        + _block(d.iloc[half:], half + 1)
        + f"\n    \\label{{{label}}}\n"
        "\\end{table}"
    )


# %%
@click.command()
@click.option("--model", default="noweather_all_nomassrange", help="which results_tod_pertype_*.csv to read")
@click.option("--out", default=None, help="write the LaTeX here instead of stdout")
def main(model, out):
    d = pd.read_csv(f"results/results_tod_pertype_{model}.csv")
    d["wtc"] = d.aircraft_type.map(lambda ac: get_wtc(get_mtow(ac)))
    d = d.sort_values("aircraft_type")

    tex = tex_table(
        d,
        "tab:pertype",
        "Per-type performance of the ADS-B-only model on the test split.",
    )
    if out:
        open(out, "w").write(tex)
        print(f"written to {out}")
    else:
        print(tex)

    print()
    print(
        d[["aircraft_type", "wtc", "n_flights", "rmse", "mape"]].to_string(
            index=False, float_format=lambda v: f"{v:.2f}"
        )
    )


# %%
if __name__ == "__main__":
    main()
