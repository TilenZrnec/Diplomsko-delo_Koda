"""Analiza poskusa uglaševanja: XGBoost z iskanjem po mreži proti privzetim algoritmom.

Zagon cc18_xgb_tuned (algoritem xgboost_tuned, 68 naborov cc18_tuning, 15 zunanjih
foldov; zasnova v njegovem PROVENANCE.md in v CLAUDE.md, What is next, korak 7)
primerja s privzetimi algoritmi diplome (cc18_v4/combined) na istih naborih in
foldih. Vse številke razdelka o uglaševanju v diplomi izpiše ta skript.

Kaj izračuna:
  1. Preverbe: 1020 vrstic brez napak; isti foldi kot v cc18_v4/combined
     (n_train, n_test, n_classes); privzeti XGBoost, ki ga isti proces nauči v
     vsakem foldu (default_test_roc_auc), je enak privzetemu XGBoost iz
     cc18_v4/combined (razlika do zadnjega bita, ~1e-16).
  2. Uglašeni proti privzetemu XGBoost po naborih (povprečje čez 15 foldov):
     število boljših naborov, povprečna in mediana razlika, dvostranski
     Wilcoxonov test (zsplit) - na vseh 68, na nenasičenih in na nasičenih
     (nasičenost iz cc18_v4/combined, kot v diplomi).
  3. Sedem algoritmov (šest privzetih + uglašeni XGBoost): povprečni ROC-AUC in
     povprečni rang na naborih (src/summary.py: rang na nabor); parni Wilcoxonovi
     testi uglašenega XGBoost proti vsakemu od šestih, Holmov popravek čez teh 6.
  4. Zaostanek za TabPFN-3.5 pred uglaševanjem in po njem ter delež, ki ga
     uglaševanje zapre.
  5. Dodatni metriki (točnost, logaritemska izguba) iz metrics.csv obeh zagonov.
  6. Zmagovalne nastavitve (tuning/<id>/xgboost_tuned_r<r>_f<f>.csv, vrstica z
     rangom 1) in robovi mreže: kolikokrat zmaga najšibkejši kot (0,01; 100
     dreves) in koliko je notranji rezultat pridobil zadnji korak z 200 na 100
     dreves; pri zmagovalcih s 1000 drevesi korak s 500 na 1000.
  7. Notranja ocena proti testni (optimizem izbire) in cena: (iskanje + ponovno
     učenje) / eno privzeto učenje na istem vozlišču.

Zagon (iz korena repozitorija):
    python scripts/analyze_tuning.py
    python scripts/analyze_tuning.py --run cc18_xgb_tuned --reference cc18_v4/combined
Izhod: <zagon>/summary/tuning_per_dataset.csv, tuning_overall.csv,
tuning_wilcoxon_holm.csv, tuning_winners.csv, tuning_report.txt.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import load_config  # noqa: E402
from src.metrics import HIGHER_IS_BETTER  # noqa: E402
from src.stats import holm  # noqa: E402
from src.summary import rank_matrix, resolve_results  # noqa: E402

TUNED, DEFAULT = "xgboost_tuned", "xgboost"
KEYS = ["dataset", "repeat", "fold"]
GRID = ["learning_rate", "n_estimators", "min_child_weight", "colsample_bylevel", "subsample"]
NAMES = {"tabpfn": "TabPFN-3.5", "tabicl": "TabICL", "catboost": "CatBoost", "lightgbm": "LightGBM",
         "xgboost": "XGBoost (privzeti)", "random_forest": "Naključni gozd",
         "xgboost_tuned": "XGBoost (uglašeni)"}


class Report:
    """Izpis na zaslon in hkrati v tuning_report.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)


def read(path):
    return pd.read_csv(path, dtype={"dataset_id": str}, float_precision="round_trip")


def with_metrics(results, run_dir):
    """results.csv z dodanima stolpcema accuracy in log_loss iz metrics.csv."""
    m = read(os.path.join(run_dir, "metrics.csv"))
    keys = ["run_id", "dataset", "algorithm", "repeat", "fold"]
    return results.merge(m[keys + ["accuracy", "log_loss"]], on=keys, how="left", validate="one_to_one")


def wilcoxon(a, b):
    """Dvostranski Wilcoxonov test predznačenih rangov (ničle zsplit, kot src/stats.py)."""
    d = a - b
    return 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)


def load_winners(run_dir, tuned):
    """Ena vrstica na fold: zmagovalna nastavitev in notranji rezultati sosednjih točk mreže."""
    rows = []
    for _, r in tuned.iterrows():
        path = os.path.join(run_dir, "tuning", r["dataset_id"],
                            f"{TUNED}_r{int(r['repeat'])}_f{int(r['fold'])}.csv")
        table = pd.read_csv(path, float_precision="round_trip")
        win = table[table["rank"] == 1].iloc[0]
        others = (table[["min_child_weight", "colsample_bylevel", "subsample"]]
                  == win[["min_child_weight", "colsample_bylevel", "subsample"]].to_numpy()).all(axis=1)
        same_lr = table[others & (table["learning_rate"] == win["learning_rate"])].set_index("n_estimators")
        row = {k: r[k] for k in KEYS + ["dataset_id"]}
        row.update({g: win[g] for g in GRID})
        row["inner_mean"] = win["inner_mean"]
        row["inner_at_200"] = same_lr["inner_mean"].get(200, np.nan)
        row["inner_at_500"] = same_lr["inner_mean"].get(500, np.nan)
        row["default_rank"] = int(table.loc[table["order"] == 0, "rank"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="cc18_xgb_tuned", help="Zagon poskusa uglaševanja")
    parser.add_argument("--reference", default="cc18_v4/combined", help="Privzeti algoritmi diplome")
    args = parser.parse_args()

    config = load_config()
    alpha = config["alpha"]
    tuned_csv, run_dir = resolve_results(args.run, config)
    ref_csv, ref_dir = resolve_results(args.reference, config)
    tuned = with_metrics(read(tuned_csv), run_dir)
    ref = with_metrics(read(ref_csv), ref_dir)
    saturated = read(os.path.join(ref_dir, "summary", "pivot.csv")).set_index("dataset")["saturated"].astype(bool)
    report = Report()

    # 1. Preverbe ----------------------------------------------------------------------
    datasets = sorted(tuned["dataset"].unique())
    ref = ref[ref["dataset"].isin(datasets)]
    report(f"Poskus uglaševanja: {os.path.relpath(run_dir, REPO_ROOT)} proti "
           f"{os.path.relpath(ref_dir, REPO_ROOT)}")
    report(f"  vrstic {len(tuned)}, naborov {len(datasets)}, z napako {int(tuned['error'].notna().sum())}")
    hp = tuned["hyperparameters"].map(json.loads)
    tuned["default_test_roc_auc"] = hp.map(lambda h: h["default_test_roc_auc"])
    tuned["chosen_inner_mean"] = hp.map(lambda h: h["chosen_inner_mean"])
    tuned["default_inner_mean"] = hp.map(lambda h: h["default_inner_mean"])
    tuned["cost_ratio"] = hp.map(lambda h: (h["search_time_s"] + h["refit_time_s"]) / h["default_train_time_s"])
    default = ref[ref["algorithm"] == DEFAULT].set_index(KEYS)
    t = tuned.set_index(KEYS)
    same_folds = all((t[c] == default.loc[t.index, c]).all() for c in ("n_train", "n_test", "n_classes"))
    d_check = (t["default_test_roc_auc"] - default.loc[t.index, "roc_auc"]).abs()
    report(f"  isti foldi kot v referenci: {same_folds}; privzeti XGBoost v zagonu = referenca na "
           f"{int((d_check == 0).sum())} od {len(d_check)} foldov, največja razlika {d_check.max():.1e}")

    # 2.-4. ROC-AUC in dodatni metriki po naborih -----------------------------------------
    both = pd.concat([ref, tuned[ref.columns.intersection(tuned.columns)]], ignore_index=True)
    algos = list(NAMES)
    subsets = {"vsi": datasets, "nenasičeni": [d for d in datasets if not saturated[d]],
               "nasičeni": [d for d in datasets if saturated[d]]}
    overall_rows, wilcoxon_rows, per_dataset = [], [], None
    for metric in ("roc_auc", "accuracy", "log_loss"):
        better = 1 if HIGHER_IS_BETTER[metric] else -1
        pivot = both.groupby(["dataset", "algorithm"])[metric].mean().unstack()[algos]
        if metric == "roc_auc":
            per_dataset = pivot.assign(saturated=saturated.reindex(pivot.index),
                                       folds_better=(t["roc_auc"] > default.loc[t.index, "roc_auc"])
                                       .groupby(level="dataset").sum())
        report()
        report(f"== {metric} ==")
        for subset, names in subsets.items():
            p = pivot.loc[names]
            ranks = rank_matrix(p, HIGHER_IS_BETTER[metric])
            for a in algos:
                overall_rows.append({"metric": metric, "subset": subset, "n": len(p), "algorithm": a,
                                     "mean": p[a].mean(), "mean_rank": ranks[a].mean()})
            gain = better * (p[TUNED] - p[DEFAULT])
            report(f"  {subset} ({len(p)}): uglašeni proti privzetemu XGBoost: boljši na "
                   f"{int((gain > 0).sum())}, slabši na {int((gain < 0).sum())}, povprečna razlika "
                   f"{(p[TUNED] - p[DEFAULT]).mean():+.4f}, mediana {(p[TUNED] - p[DEFAULT]).median():+.4f}, "
                   f"Wilcoxon p = {wilcoxon(p[TUNED], p[DEFAULT]):.2g}")
            report("    povprečje / povprečni rang (7 algoritmov): " + ", ".join(
                f"{NAMES[a]} {p[a].mean():.4f} / {ranks[a].mean():.2f}"
                for a in sorted(algos, key=lambda a: ranks[a].mean())))
            if metric == "roc_auc":
                gap_d, gap_t = (p["tabpfn"] - p[DEFAULT]).mean(), (p["tabpfn"] - p[TUNED]).mean()
                report(f"    zaostanek za TabPFN-3.5: privzeti {gap_d:.4f}, uglašeni {gap_t:.4f}, "
                       f"zaprto {100 * (1 - gap_t / gap_d):.0f} %; uglašeni pred TabPFN-3.5 na "
                       f"{int((p[TUNED] > p['tabpfn']).sum())}")
            if subset in ("vsi", "nenasičeni"):
                block = []
                for other in [a for a in algos if a != TUNED]:
                    diff = p[TUNED] - p[other]
                    block.append({"metric": metric, "subset": subset, "n": len(p), "a": TUNED, "b": other,
                                  "mean_diff": diff.mean(), "wins_a": int((better * diff > 0).sum()),
                                  "wins_b": int((better * diff < 0).sum()), "p": wilcoxon(p[TUNED], p[other])})
                for r, ph in zip(block, holm([r["p"] for r in block])):
                    r["p_holm"], r["significant"] = float(ph), bool(ph < alpha)
                wilcoxon_rows += block
                report("    Wilcoxon + Holm (6 primerjav): " + "; ".join(
                    f"{NAMES[r['b']]} {r['wins_a']}:{r['wins_b']} p_Holm {r['p_holm']:.2g}" for r in block))

    # 6. Zmagovalne nastavitve in robovi mreže ------------------------------------------------
    win = load_winners(run_dir, tuned)
    win["saturated"] = win["dataset"].map(saturated)
    report()
    report("== zmagovalne nastavitve (fold = ena izbira) ==")
    winner_rows = []
    for subset, mask in (("vsi", win["dataset"].notna()), ("nenasičeni", ~win["saturated"]),
                         ("nasičeni", win["saturated"])):
        w = win[mask]
        for g in GRID:
            for value, count in w[g].value_counts().sort_index().items():
                winner_rows.append({"subset": subset, "n_folds": len(w), "parameter": g,
                                    "value": value, "count": int(count)})
        report(f"  {subset} ({len(w)} foldov): " + "; ".join(
            f"{g} " + "/".join(f"{v:g}:{c}" for v, c in w[g].value_counts().sort_index().items())
            for g in GRID))
        low = w[(w["learning_rate"] == 0.01) & (w["n_estimators"] == 100)]
        top = w[w["n_estimators"] == 1000]
        is_default = ((w["learning_rate"] == 0.3) & (w["n_estimators"] == 100) & (w["min_child_weight"] == 1)
                      & (w["colsample_bylevel"] == 1) & (w["subsample"] == 1))
        report(f"    privzeta nastavitev zmagala {int(is_default.sum())}-krat, mediana njenega notranjega "
               f"ranga {w['default_rank'].median():.0f} od 128; najmočnejši kot (0,3; 1000) "
               f"{int(((w['learning_rate'] == 0.3) & (w['n_estimators'] == 1000)).sum())}-krat")
        if len(low):
            gain = low["inner_mean"] - low["inner_at_200"]
            report(f"    najšibkejši kot (0,01; 100) {len(low)}-krat; korak z 200 na 100 dreves: mediana "
                   f"notranjega dobička {gain.median():.4f}, največ {gain.max():.4f}")
        if len(top):
            gain = top["inner_mean"] - top["inner_at_500"]
            report(f"    1000 dreves {len(top)}-krat; korak s 500 na 1000: mediana {gain.median():.5f}, "
                   f"90. percentil {gain.quantile(0.9):.4f}")
        common = w.groupby(["learning_rate", "n_estimators"]).size().sort_values(ascending=False)
        report(f"    najpogostejši (stopnja učenja, drevesa): " + ", ".join(
            f"({lr:g}; {int(n)}) {c}" for (lr, n), c in common.head(3).items()))

    # 7. Notranja ocena proti testni in cena --------------------------------------------------
    report()
    report("== notranja ocena proti testni, cena ==")
    t = t.reset_index()
    t["saturated"] = t["dataset"].map(saturated)
    for subset, mask in (("nenasičeni", ~t["saturated"]), ("nasičeni", t["saturated"])):
        s = t[mask]
        inner = (s["chosen_inner_mean"] - s["default_inner_mean"]).groupby(s["dataset"]).mean().mean()
        test = (s["roc_auc"] - s["default_test_roc_auc"]).groupby(s["dataset"]).mean().mean()
        report(f"  {subset}: notranje preverjanje obeta {inner:+.4f} proti privzetemu, test da {test:+.4f}; "
               f"cena = mediana {s['cost_ratio'].median():.0f}-kratnika privzetega učenja "
               f"(razpon {s['cost_ratio'].min():.0f}-{s['cost_ratio'].max():.0f})")
    worst = per_dataset.assign(diff=per_dataset[TUNED] - per_dataset[DEFAULT]).sort_values("diff")
    for name in worst.index[:2]:
        s = t[t["dataset"] == name]
        report(f"  {name}: razlika {worst.loc[name, 'diff']:+.4f}; notranji dobiček > 0 na "
               f"{int((s['chosen_inner_mean'] > s['default_inner_mean']).sum())} od {len(s)} foldov, "
               f"testni < 0 na {int((s['roc_auc'] < s['default_test_roc_auc']).sum())}")
    report(f"  največji dobički: " + ", ".join(
        f"{n} {d:+.4f}" for n, d in worst["diff"].sort_values(ascending=False).head(5).items()))
    report(f"  čas uglaševanja skupaj (train_time_s): {t['train_time_s'].sum() / 3600:.1f} h na 8 jedrih")

    out_dir = os.path.join(run_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)
    per_dataset.reset_index().to_csv(os.path.join(out_dir, "tuning_per_dataset.csv"), index=False)
    pd.DataFrame(overall_rows).to_csv(os.path.join(out_dir, "tuning_overall.csv"), index=False)
    pd.DataFrame(wilcoxon_rows).to_csv(os.path.join(out_dir, "tuning_wilcoxon_holm.csv"), index=False)
    pd.DataFrame(winner_rows).to_csv(os.path.join(out_dir, "tuning_winners.csv"), index=False)
    with open(os.path.join(out_dir, "tuning_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report.lines) + "\n")
    print(f"\nZapisano v {os.path.relpath(out_dir, REPO_ROOT)}/tuning_*")


if __name__ == "__main__":
    main()
