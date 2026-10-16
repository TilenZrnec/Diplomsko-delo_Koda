"""Povzetek rezultatov zagona: ROC-AUC po naborih, rangi, časi, nasičenost.

Zagon: python -m src.summary <run_id | mapa zagona | results.csv> [--metric M]

--metric izbere metriko: roc_auc (privzeto, glavna metrika, iz results.csv) ali
dodatni accuracy / log_loss (iz metrics.csv, ki ga naredi
scripts/compute_metrics.py). Tabele dodatnih metrik gredo v summary/<metrika>/;
pri log_loss je boljša nižja vrednost, zato rang 1 dobi najnižja. Nasičenost je
vedno določena po ROC-AUC, da je podmnožica nenasičenih naborov pri vseh
metrikah ista.

Pot je OBVEZNA - brez nje ni privzetka, da se nikoli ne povzame napačna
tabela. Prva vrstica izpisa je vedno "Vir: <pot>", da je jasno, kateri
rezultati so povzeti. Poleg izpisa zapiše tabele kot CSV in LaTeX v mapo
summary/ ob rezultatih, da gredo številke v diplomo iz datotek, ne s prepisom.

Pravila, ki jih je treba poznati (in so zapisana v diplomi):

* Rang se računa NA NABOR na povprečnem ROC-AUC čez folde - ne na (nabor, fold),
  ker foldi niso neodvisni in bi statistični test navidezno dobil 5x več vzorcev.
* Neuspeh je kaznovan: če algoritem na naboru na kateremkoli foldu ni dal
  rezultata, je njegov ROC-AUC za ta nabor NaN in dobi najslabši rang
  (izenačeni neuspehi si delijo povprečje najslabših rangov).
* "Povprečni ROC-AUC" se poroča na naborih, kjer so uspeli VSI algoritmi, da
  se povprečja ne računajo na različnih podmnožicah.
* Nasičen nabor: najboljši algoritem doseže >= saturation_threshold (config.yaml).
  Tam razlike niso informativne, zato je vsak povzetek podan še brez njih.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import load_config  # noqa: E402
from src.metrics import HIGHER_IS_BETTER  # noqa: E402

METRIC_LABELS = {"roc_auc": "ROC-AUC", "accuracy": "točnost", "log_loss": "logaritemska izguba"}
METRIC_KEYS = ["run_id", "dataset", "algorithm", "repeat", "fold"]


# --------------------------------------------------------------------------- #
# Vhod                                                                         #
# --------------------------------------------------------------------------- #

def resolve_results(path_or_id, config=None):
    """Vrne (pot do results.csv, mapa za summary/) iz run_id, mape zagona ali CSV-ja."""
    config = config or load_config()
    candidates = [path_or_id, os.path.join(config["results_dir"], path_or_id)]
    for cand in candidates:
        if os.path.isfile(cand) and cand.endswith(".csv"):
            return os.path.abspath(cand), os.path.dirname(os.path.abspath(cand))
        if os.path.isdir(cand):
            csv = os.path.join(cand, "results.csv")
            if not os.path.isfile(csv):
                raise SystemExit(f"{cand}: ni results.csv - najprej scripts/merge_results.py")
            return os.path.abspath(csv), os.path.abspath(cand)
    raise SystemExit(f"Ne najdem rezultatov: {path_or_id}")


def load_results(results_csv, metric="roc_auc"):
    """Prebere CSV in poskrbi za stolpce, ki jih starejši zagoni nimajo.

    Pri dodatni metriki (accuracy, log_loss) pripne njen stolpec iz metrics.csv
    ob results.csv; učenje brez te metrike šteje za neuspelo.
    """
    df = pd.read_csv(results_csv)
    if "error" not in df.columns:
        df["error"] = np.nan
    # Neuspeh = manjka roc_auc ali je zapisana napaka.
    df["failed"] = df["roc_auc"].isna() | df["error"].notna()
    if metric != "roc_auc":
        metrics_csv = os.path.join(os.path.dirname(results_csv), "metrics.csv")
        if not os.path.isfile(metrics_csv):
            raise SystemExit(f"{metrics_csv} ne obstaja - najprej scripts/compute_metrics.py")
        extra = pd.read_csv(metrics_csv)[METRIC_KEYS + [metric]]
        df = df.merge(extra, on=METRIC_KEYS, how="left", validate="one_to_one")
        df["failed"] = df["failed"] | df[metric].isna()
    return df


# --------------------------------------------------------------------------- #
# Tabele                                                                       #
# --------------------------------------------------------------------------- #

def per_dataset_table(df, metric="roc_auc"):
    """Ena vrstica na (nabor, algoritem): mean/std metrike, mediane časov, št. neuspehov."""
    g = df.groupby(["dataset", "algorithm"])
    table = g.agg(
        n_folds=("fold", "count"),
        n_failed=("failed", "sum"),
        **{f"{metric}_mean": (metric, "mean"), f"{metric}_std": (metric, "std")},
        train_time_median_s=("train_time_s", "median"),
        inference_time_median_s=("inference_time_s", "median"),
    ).reset_index()
    # Delen neuspeh (nekateri foldi) bi dal optimistično povprečje uspelih foldov;
    # zato povprečje velja le, če so uspeli vsi foldi.
    table.loc[table["n_failed"] > 0, [f"{metric}_mean", f"{metric}_std"]] = np.nan
    return table


def dataset_pivot(df, metric="roc_auc"):
    """Matrika nabor x algoritem s povprečjem metrike (NaN = neuspeh na tem naboru)."""
    table = per_dataset_table(df, metric)
    return table.pivot(index="dataset", columns="algorithm", values=f"{metric}_mean")


def rank_matrix(pivot, higher_is_better=True):
    """Rangi po vrsticah (1 = najboljši); NaN dobi najslabši rang, izenačeni povprečje."""
    filled = pivot.fillna(-np.inf if higher_is_better else np.inf)
    return filled.rank(axis=1, ascending=not higher_is_better, method="average")


def saturated_mask(pivot, threshold):
    """True za nabore, kjer najboljši algoritem doseže vsaj threshold."""
    return pivot.max(axis=1) >= threshold


def overall_table(df, pivot, ranks, metric="roc_auc"):
    """Ena vrstica na algoritem: povprečna metrika, rang, zmage, neuspehi, časi."""
    complete = pivot.dropna(axis=0, how="any")
    rows = []
    for algo in pivot.columns:
        sub = df[df["algorithm"] == algo]
        rows.append({
            "algorithm": algo,
            f"mean_{metric}_complete": complete[algo].mean(),
            f"mean_{metric}_own": pivot[algo].mean(),
            "mean_rank": ranks[algo].mean(),
            "n_wins": int((ranks[algo] == ranks.min(axis=1)).sum()),
            "n_failed_datasets": int(pivot[algo].isna().sum()),
            "n_datasets": int(len(pivot)),
            "train_time_median_s": sub["train_time_s"].median(),
            "inference_time_median_s": sub["inference_time_s"].median(),
            "train_time_total_s": sub["train_time_s"].sum(),
            "inference_time_total_s": sub["inference_time_s"].sum(),
        })
    out = pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)
    out.attrs["n_complete"] = int(len(complete))
    return out


# --------------------------------------------------------------------------- #
# Izpis in zapis                                                               #
# --------------------------------------------------------------------------- #

def to_latex(df, caption, label, floatfmt=".4f", column_names=None):
    """Preprosta LaTeX tabela v slogu diplome (\\hline, brez booktabs)."""
    cols = list(df.columns)
    names = column_names or cols
    align = "l" + "r" * (len(cols) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(_tex_escape(str(n)) for n in names) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                cells.append("--" if np.isnan(val) else format(val, floatfmt))
            else:
                cells.append(_tex_escape(str(val)))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _tex_escape(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def write_outputs(out_dir, tables):
    os.makedirs(out_dir, exist_ok=True)
    for name, (df, caption, label, floatfmt) in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        with open(os.path.join(out_dir, f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(to_latex(df, caption, label, floatfmt))


def metric_dir(base_dir, metric):
    """summary/ za glavno metriko, summary/<metrika>/ za dodatne."""
    out = os.path.join(base_dir, "summary")
    return out if metric == "roc_auc" else os.path.join(out, metric)


def summarize(path_or_id, config=None, write=True, metric="roc_auc"):
    config = config or load_config()
    results_csv, base_dir = resolve_results(path_or_id, config)
    print(f"Vir: {os.path.relpath(results_csv, REPO_ROOT)}\n")

    df = load_results(results_csv, metric)
    threshold = config["saturation_threshold"]
    label = METRIC_LABELS[metric]

    per_ds = per_dataset_table(df, metric)
    pivot = dataset_pivot(df, metric)
    ranks = rank_matrix(pivot, HIGHER_IS_BETTER[metric])
    # Nasičenost vedno po ROC-AUC (glej docstring modula).
    roc_pivot = pivot if metric == "roc_auc" else dataset_pivot(df, "roc_auc")
    saturated = saturated_mask(roc_pivot, threshold).reindex(pivot.index)

    overall = overall_table(df, pivot, ranks, metric)
    overall_ns = overall_table(
        df[df["dataset"].isin(pivot.index[~saturated])], pivot[~saturated], ranks[~saturated],
        metric,
    )

    show_cols = ["algorithm", f"mean_{metric}_complete", "mean_rank", "n_wins",
                 "n_failed_datasets", "train_time_median_s", "inference_time_median_s"]

    print(f"=== Vsi nabori ({len(pivot)}; povprečna metrika {label} na {overall.attrs['n_complete']} "
          "naborih, kjer so uspeli vsi algoritmi; rang na nabor, neuspeh = najslabši rang) ===")
    print(tabulate(overall[show_cols], headers="keys", floatfmt=".4f", showindex=False))

    print(f"\n=== Brez nasičenih naborov (najboljši ROC-AUC >= {threshold}): "
          f"{int((~saturated).sum())} naborov, {int(saturated.sum())} izločenih ===")
    print(tabulate(overall_ns[show_cols], headers="keys", floatfmt=".4f", showindex=False))

    print(f"\n=== {label} po naborih (povprečje čez folde; -- = neuspeh) ===")
    shown = pivot.round(4).astype(object).where(pivot.notna(), "--").reset_index()
    print(tabulate(shown, headers="keys", floatfmt=".4f", showindex=False))

    failed = df[df["failed"]]
    if not failed.empty:
        print("\n=== Neuspela učenja ===")
        cols = [c for c in ["dataset", "algorithm", "fold", "error"] if c in failed.columns]
        short = failed[cols].copy()
        if "error" in short:
            short["error"] = short["error"].astype(str).str.slice(0, 90)
        print(tabulate(short, headers="keys", showindex=False))

    if write:
        out_dir = metric_dir(base_dir, metric)
        pivot_out = pivot.reset_index()
        pivot_out["saturated"] = saturated.values
        write_outputs(out_dir, {
            "overall": (overall, "Povzetek čez vse nabore.", "tab:overall", ".4f"),
            "overall_nonsaturated": (
                overall_ns, f"Povzetek brez nasičenih naborov (ROC-AUC $\\geq$ {threshold}).",
                "tab:overall-nonsat", ".4f"),
            "per_dataset": (per_ds, f"{label} po naborih in algoritmih.", "tab:per-dataset", ".4f"),
            "pivot": (pivot_out, f"Povprečni {label} po naborih." if metric == "roc_auc"
                      else f"Povprečje metrike {label} po naborih.", "tab:pivot", ".4f"),
            "ranks": (ranks.reset_index(), "Rangi po naborih.", "tab:ranks", ".2f"),
        })
        print(f"\nTabele zapisane v {os.path.relpath(out_dir, REPO_ROOT)}/ (CSV + LaTeX)")

    return {"per_dataset": per_ds, "pivot": pivot, "ranks": ranks, "saturated": saturated,
            "overall": overall, "overall_nonsaturated": overall_ns}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Povzetek rezultatov benchmarka.")
    parser.add_argument("results", help="run_id, mapa zagona ali pot do results.csv")
    parser.add_argument("--metric", default="roc_auc", choices=list(HIGHER_IS_BETTER),
                        help="metrika (privzeto roc_auc; dodatni potrebujeta metrics.csv)")
    parser.add_argument("--no-write", action="store_true", help="Samo izpis, brez zapisa summary/")
    args = parser.parse_args()
    summarize(args.results, write=not args.no_write, metric=args.metric)
