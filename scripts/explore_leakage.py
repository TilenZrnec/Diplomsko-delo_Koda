"""Test uhajanja podatkov: raziskovalna analiza po zagonu (post hoc, NI del vnaprej določenega testa).

Vnaprej določeni test je scripts/analyze_leakage.py (zasnova v
results/runs/cc18_leakage/DESIGN.md) in ga ta skript ne spreminja. Tu so
številke, ki smo jih pogledali šele po rezultatu testa, da bi ocenili velikost
učinka; v diplomi so označene kot raziskovalne:

  1. Prednost temeljnega modela pred CatBoostom (povprečni ROC-AUC čez 5 foldov,
     nato povprečje čez nabore) v vsaki različici in delež prvotne prednosti, ki
     ostane na spremenjenih kopijah; število naborov, kjer je model še vedno
     boljši, in dvostranski Wilcoxonov test (zsplit) te prednosti.
  2. Spearmanov koeficient med delto (padec modela minus padec CatBoosta) in
     prvotno prednostjo ter med delto in velikostjo učne množice.
  3. Nabori z največjo delto.
  4. Isti izračun delte kot v vnaprej določenem testu, a za točnost in
     logaritemsko izgubo (metrics.csv; pri izgubi je padec povečanje izgube),
     enostranski Wilcoxon, Holm znotraj modela.

Zagon (iz korena repozitorija):
    python scripts/explore_leakage.py cc18_leakage --reference medic3_160_leakage
Izhod: <zagon>/summary/leakage_exploratory.csv (prednosti po različicah),
leakage_exploratory_metrics.csv (delta za dodatni metriki), leakage_exploratory_report.txt.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.analyze_leakage import NAMES, complete_datasets, load_run  # noqa: E402
from src.config import load_config  # noqa: E402
from src.metrics import HIGHER_IS_BETTER  # noqa: E402
from src.stats import holm  # noqa: E402


class Report:
    """Izpis na zaslon in hkrati v leakage_exploratory_report.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)


def with_metrics(df, run_dir):
    m = pd.read_csv(os.path.join(run_dir, "metrics.csv"), float_precision="round_trip")
    keys = ["run_id", "dataset", "algorithm", "repeat", "fold"]
    return df.merge(m[keys + ["accuracy", "log_loss"]], on=keys, how="left", validate="one_to_one")


def means(df, metric):
    """Povprečje metrike čez folde: vrstice (nabor, algoritem), stolpci različice."""
    return df.groupby(["base_name", "algorithm", "perturbation"])[metric].mean().unstack("perturbation")


def metric_deltas(df, leakage, metric):
    """delta = padec(model) - padec(kontrola) za poljubno metriko, po naborih."""
    sign = 1 if HIGHER_IS_BETTER[metric] else -1
    m = means(df, metric)
    control = leakage["control"]
    rows = []
    for (name, alg), r in m.iterrows():
        if alg == control:
            continue
        c = m.loc[(name, control)]
        for arm in [a for a in leakage["arms"] if a != "original"]:
            drop_model = sign * (r["original"] - r[arm])
            drop_control = sign * (c["original"] - c[arm])
            rows.append({"dataset": name, "model": alg, "arm": arm, "metric": metric,
                         "delta": drop_model - drop_control})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="Zagon testa (cc18_leakage)")
    parser.add_argument("--reference", default=None, help="Zagon na referenčnem naboru (medic3_160_leakage)")
    args = parser.parse_args()

    config = load_config()
    leakage, alpha = config["leakage"], config["alpha"]
    n_folds = int(leakage["repeats"]) * config["n_splits"]
    control = leakage["control"]
    models = [a for a in leakage["algorithms"] if a != control]
    report = Report()
    report("Raziskovalna analiza po zagonu (post hoc) - ni del vnaprej določenega testa.")

    df_all, run_dir = load_run(args.run, config)
    df = with_metrics(complete_datasets(df_all, leakage, n_folds, report), run_dir)
    ref = None
    if args.reference:
        ref_all, ref_dir = load_run(args.reference, config)
        ref = with_metrics(complete_datasets(ref_all, leakage, n_folds, report), ref_dir)

    # 1. Prednost pred CatBoostom po različicah ----------------------------------------
    auc = means(df, "roc_auc")
    lead_rows = []
    report()
    report(f"Prednost pred kontrolo ({NAMES[control]}), ROC-AUC, {auc.index.get_level_values(0).nunique()} naborov:")
    for model in models:
        original = None
        for arm in leakage["arms"]:
            lead = auc.xs(model, level="algorithm")[arm] - auc.xs(control, level="algorithm")[arm]
            original = lead.mean() if arm == "original" else original
            p = float(stats.wilcoxon(lead, zero_method="zsplit").pvalue)
            row = {"model": model, "arm": arm, "n": len(lead), "mean_lead": lead.mean(),
                   "median_lead": lead.median(), "share_of_original": lead.mean() / original,
                   "n_model_better": int((lead > 0).sum()), "p_wilcoxon": p}
            if ref is not None:
                ra = means(ref, "roc_auc")
                row["reference_lead"] = float((ra.xs(model, level="algorithm")[arm]
                                               - ra.xs(control, level="algorithm")[arm]).iloc[0])
            lead_rows.append(row)
            report(f"  {NAMES[model]:11} {arm:15} povprečna prednost {row['mean_lead']:+.4f} "
                   f"({100 * row['share_of_original']:.0f} % izvirne), boljši na {row['n_model_better']}/"
                   f"{row['n']}, p = {p:.2g}" + (f"; Medic3 {row['reference_lead']:+.4f}" if ref is not None else ""))
    leads = pd.DataFrame(lead_rows)

    # 2.-3. Delta proti prvotni prednosti in velikosti; največje delte ------------------------
    d_auc = metric_deltas(df, leakage, "roc_auc")
    n_train = df.groupby("base_name")["n_train"].mean()
    report()
    report("Spearman: delta (ROC-AUC) proti prvotni prednosti pred kontrolo in proti velikosti učne množice:")
    for model in models:
        orig_lead = auc.xs(model, level="algorithm")["original"] - auc.xs(control, level="algorithm")["original"]
        for arm in [a for a in leakage["arms"] if a != "original"]:
            d = d_auc[(d_auc["model"] == model) & (d_auc["arm"] == arm)].set_index("dataset")["delta"]
            r1 = stats.spearmanr(d, orig_lead.loc[d.index])
            r2 = stats.spearmanr(d, n_train.loc[d.index])
            report(f"  {NAMES[model]:11} {arm:15} rho(prednost) = {r1.statistic:+.2f} (p = {r1.pvalue:.2g}), "
                   f"rho(velikost) = {r2.statistic:+.2f} (p = {r2.pvalue:.2g})")
    top = d_auc[(d_auc["model"] == leakage["primary"]) & (d_auc["arm"] == "monotone")].nlargest(8, "delta")
    report(f"  največja delta ({NAMES[leakage['primary']]}, monotona): " + ", ".join(
        f"{r['dataset']} {r['delta']:.3f}" for _, r in top.iterrows()))

    # 4. Dodatni metriki ---------------------------------------------------------------------
    report()
    report("Delta za dodatni metriki (enostranski Wilcoxon, H1: delta > 0, Holm znotraj modela):")
    metric_rows = []
    for metric in ("accuracy", "log_loss"):
        dm = metric_deltas(df, leakage, metric)
        dref = metric_deltas(ref, leakage, metric) if ref is not None else None
        for model in models:
            block = []
            for arm in [a for a in leakage["arms"] if a != "original"]:
                d = dm[(dm["model"] == model) & (dm["arm"] == arm)]["delta"]
                res = stats.wilcoxon(d, zero_method="zsplit", alternative="greater")
                row = {"metric": metric, "model": model, "arm": arm, "n": len(d),
                       "median_delta": d.median(), "n_positive": int((d > 0).sum()), "p": float(res.pvalue)}
                if dref is not None:
                    row["reference_delta"] = float(dref[(dref["model"] == model) & (dref["arm"] == arm)]["delta"].iloc[0])
                block.append(row)
            for r, ph in zip(block, holm([r["p"] for r in block])):
                r["p_holm"] = float(ph)
                r["significant"] = bool(ph < alpha)
                report(f"  {metric:9} {NAMES[model]:11} {r['arm']:15} mediana delta {r['median_delta']:+.4f}, "
                       f"> 0 na {r['n_positive']}/{r['n']}, p_Holm = {r['p_holm']:.2g}"
                       + (f"; Medic3 {r['reference_delta']:+.4f}" if dref is not None else ""))
            metric_rows += block

    out_dir = os.path.join(run_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)
    leads.to_csv(os.path.join(out_dir, "leakage_exploratory.csv"), index=False)
    pd.DataFrame(metric_rows).to_csv(os.path.join(out_dir, "leakage_exploratory_metrics.csv"), index=False)
    with open(os.path.join(out_dir, "leakage_exploratory_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report.lines) + "\n")
    print(f"\nZapisano v {os.path.relpath(out_dir, REPO_ROOT)}/leakage_exploratory*")


if __name__ == "__main__":
    main()
