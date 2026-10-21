"""Test uhajanja podatkov: vnaprej določena analiza zagona (zasnova, utemeljitve in viri
v results/runs/cc18_leakage/DESIGN.md; parametri v config.yaml, razdelek leakage).

    python scripts/analyze_leakage.py cc18_leakage --reference medic3_160_leakage
    python scripts/analyze_leakage.py leakage_smoke_local          # tehnični preizkus

Kaj izračuna:
  1. Za vsak nabor, algoritem in različico povprečni ROC-AUC čez folde.
  2. padec = ROC-AUC(original) - ROC-AUC(različica), za vsako spremenjeno različico.
  3. delta = padec(temeljni model) - padec(CatBoost). Kontrola odšteje del padca, ki
     ga povzroči že sama sprememba podatkov (šum, razbite vezi); monotona preslikava
     drevesa skoraj ne spremeni, temeljni model pa dobi drugačne vrednosti.
  4. Enostranski Wilcoxonov test predznačenih rangov čez nabore, H1: delta > 0
     (temeljni model pade bolj kot kontrola, kot bi pri pomnjenju); ničle razdeli
     zero_method="zsplit" kot src/stats.py (Demšar 2006). Holmov popravek čez obe
     spremenjeni različici znotraj modela. Glavni model je leakage.primary
     (TabPFN-3.5), ostali (TabICL) so drugotni rezultat.
  5. Referenca (--reference, Medic3-160, nikoli objavljen): enak delta in še po
     foldih. Vnaprej določeno pravilo: značilen rezultat na CC18 je skladen s
     pomnjenjem le, če je delta na referenci MANJŠI od mediane delta na CC18; sicer
     ga pojasni splošna občutljivost modela na preslikavo.

Preverbe (niso del testa): vse pričakovane vrstice brez napak (nabor, kjer katerikoli
algoritem v katerikoli različici ni uspel, je iz testa izločen in izpisan); padec
CatBoosta pri monotoni različici (naj bo ~0); izvirna različica proti diplomskim
rezultatom na istih foldih (--check-against; CatBoost naj bo enak, temeljna modela
do ~1e-3 zaradi GPU).

Izhod: <zagon>/summary/leakage_per_dataset.csv, leakage_tests.csv/.tex,
leakage_reference.csv (z referenco) in leakage_report.txt (to, kar se izpiše).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import load_config  # noqa: E402
from src.stats import corrected_resampled_ttest, holm  # noqa: E402
from src.summary import resolve_results, to_latex  # noqa: E402

# Najmanj naborov za Wilcoxonov test: pri n = 5 je najmanjša enostranska p = 1/32.
MIN_DATASETS = 6
NAMES = {"tabpfn": "TabPFN-3.5", "tabicl": "TabICL", "catboost": "CatBoost",
         "monotone": "monotona preslikava", "monotone_noise": "preslikava s šumom"}


class Report:
    """Izpis na zaslon in hkrati v leakage_report.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)


def load_run(path_or_id, config):
    """results.csv zagona testa uhajanja; doda base_id (nabor brez pripone različice)."""
    csv, run_dir = resolve_results(path_or_id, config)
    df = pd.read_csv(csv, dtype={"dataset_id": str}, float_precision="round_trip")
    if "perturbation" not in df.columns or df["perturbation"].isna().any():
        raise SystemExit(f"{csv}: ni zagon testa uhajanja (manjka stolpec perturbation)")
    unknown = set(df["perturbation"]) - set(config["leakage"]["arms"])
    if unknown:
        raise SystemExit(f"{csv}: neznane različice {sorted(unknown)}")
    df["base_id"] = [i[: -len(a) - 1] for i, a in zip(df["dataset_id"], df["perturbation"])]
    df["base_name"] = [n[: -len(a) - 1] for n, a in zip(df["dataset"], df["perturbation"])]
    return df, run_dir


def complete_datasets(df, leakage, n_folds, report):
    """Vrne vrstice le tistih naborov, kjer je vsak (algoritem, različica) uspel na vseh foldih."""
    good = df[df["error"].isna() & df["roc_auc"].notna()]
    counts = good.groupby(["base_id", "algorithm", "perturbation"])["fold"].nunique()
    keep = []
    for base in sorted(df["base_id"].unique()):
        missing = [(alg, arm) for alg in leakage["algorithms"] for arm in leakage["arms"]
                   if counts.get((base, alg, arm), 0) != n_folds]
        if missing:
            report(f"  IZLOČEN {base}: nepopolno ali z napako {missing}")
        else:
            keep.append(base)
    return good[good["base_id"].isin(keep)]


def deltas(df, leakage):
    """Tabela (nabor, model, različica): povprečni ROC-AUC, padca in delta."""
    control = leakage["control"]
    perturbed = [a for a in leakage["arms"] if a != "original"]
    means = df.groupby(["base_id", "base_name", "algorithm", "perturbation"])["roc_auc"].mean()
    means = means.unstack("perturbation")
    rows = []
    for (base, name, alg), m in means.iterrows():
        if alg == control:
            continue
        c = means.loc[(base, name, control)]
        for arm in perturbed:
            drop_model = m["original"] - m[arm]
            drop_control = c["original"] - c[arm]
            rows.append({
                "dataset_id": base, "dataset": name, "model": alg, "arm": arm,
                "auc_original": m["original"], "auc_arm": m[arm],
                "drop_model": drop_model,
                "control_auc_original": c["original"], "control_auc_arm": c[arm],
                "drop_control": drop_control,
                "delta": drop_model - drop_control,
            })
    return pd.DataFrame(rows)


def fold_deltas(df, leakage, model, arm):
    """delta po foldih (isti foldi v vseh različicah) za en model in eno različico."""
    p = df.pivot_table(index=["algorithm", "fold"], columns="perturbation", values="roc_auc")
    drop = p["original"] - p[arm]
    return (drop.loc[model] - drop.loc[leakage["control"]]).sort_index()


def run_tests(per_dataset, leakage, alpha):
    rows = []
    models = [leakage["primary"]] + [a for a in leakage["algorithms"]
                                      if a not in (leakage["primary"], leakage["control"])]
    for model in models:
        block = []
        for arm in [a for a in leakage["arms"] if a != "original"]:
            d = per_dataset[(per_dataset["model"] == model) & (per_dataset["arm"] == arm)]["delta"]
            row = {"model": model, "role": "glavni" if model == leakage["primary"] else "drugotni",
                   "arm": arm, "n": len(d), "median_delta": d.median(), "mean_delta": d.mean(),
                   "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
                   "statistic": np.nan, "p": np.nan}
            if len(d) >= MIN_DATASETS:
                res = stats.wilcoxon(d, zero_method="zsplit", alternative="greater")
                row["statistic"], row["p"] = float(res.statistic), float(res.pvalue)
            block.append(row)
        ps = [r["p"] for r in block]
        adjusted = holm(ps) if not any(np.isnan(ps)) else [np.nan] * len(ps)
        for r, ph in zip(block, adjusted):
            r["p_holm"] = ph
            r["significant"] = bool(ph < alpha) if not np.isnan(ph) else None
        rows += block
    return pd.DataFrame(rows)


def check_against(df, other, label, report):
    """Izvirna različica proti diplomskemu zagonu na istih (nabor, algoritem, fold)."""
    csv, _ = resolve_results(other)
    thesis = pd.read_csv(csv, dtype={"dataset_id": str}, float_precision="round_trip")
    mine = df[df["perturbation"] == "original"][["base_id", "algorithm", "fold", "roc_auc"]]
    merged = mine.merge(thesis[["dataset_id", "algorithm", "fold", "roc_auc"]],
                        left_on=["base_id", "algorithm", "fold"],
                        right_on=["dataset_id", "algorithm", "fold"], suffixes=("", "_thesis"))
    if merged.empty:
        report(f"  {label}: ni skupnih učenj z {other}")
        return
    diff = (merged["roc_auc"] - merged["roc_auc_thesis"]).abs()
    for alg, d in diff.groupby(merged["algorithm"]):
        report(f"  {label}, {NAMES.get(alg, alg)}: {len(d)} učenj, enakih {int((d == 0).sum())}, "
               f"največja |razlika| {d.max():.2e} (proti {other})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="Zagon testa (npr. cc18_leakage) ali pot do results.csv")
    parser.add_argument("--reference", default=None, help="Zagon na referenčnem naboru (medic3_160_leakage)")
    parser.add_argument("--check-against", default="cc18_v4/combined",
                        help="Diplomski zagon za preverbo izvirne različice (privzeto cc18_v4/combined)")
    parser.add_argument("--reference-check-against", default="medic3_160_v2/combined",
                        help="Diplomski zagon za preverbo reference (privzeto medic3_160_v2/combined)")
    args = parser.parse_args()

    config = load_config()
    leakage, alpha = config["leakage"], config["alpha"]
    n_folds = int(leakage["repeats"]) * config["n_splits"]
    report = Report()

    df_all, run_dir = load_run(args.run, config)
    report(f"Test uhajanja: {os.path.relpath(run_dir, REPO_ROOT)} "
           f"({df_all['base_id'].nunique()} naborov, {len(df_all)} vrstic, {n_folds} foldov)")
    expected = df_all["base_id"].nunique() * len(leakage["arms"]) * len(leakage["algorithms"]) * n_folds
    report(f"  pričakovanih vrstic {expected}, z napako {int(df_all['error'].notna().sum())}")
    df = complete_datasets(df_all, leakage, n_folds, report)
    per_dataset = deltas(df, leakage)
    tests = run_tests(per_dataset, leakage, alpha)

    ref, ref_rows = None, []
    if args.reference:
        ref_all, _ = load_run(args.reference, config)
        ref = complete_datasets(ref_all, leakage, n_folds, report)
        ref_table = deltas(ref, leakage)
        for _, r in ref_table.iterrows():
            folds = fold_deltas(ref, leakage, r["model"], r["arm"])
            t, p = corrected_resampled_ttest(folds.to_numpy(), 1.0 / (config["n_splits"] - 1))
            ref_rows.append({**r.to_dict(), "fold_deltas": " ".join(f"{v:+.4f}" for v in folds),
                             "n_folds_positive": int((folds > 0).sum()),
                             "corrected_t": t, "corrected_p_two_sided": p})
    ref_df = pd.DataFrame(ref_rows)

    # Odločitev po vnaprej določenem pravilu.
    verdicts = []
    for _, r in tests.iterrows():
        cc18 = per_dataset[(per_dataset["model"] == r["model"]) & (per_dataset["arm"] == r["arm"])]
        ref_delta = np.nan
        if len(ref_df):
            sel = ref_df[(ref_df["model"] == r["model"]) & (ref_df["arm"] == r["arm"])]
            ref_delta = float(sel["delta"].iloc[0]) if len(sel) else np.nan
        if r["significant"] is None:
            verdict = f"test ni izveden (manj kot {MIN_DATASETS} naborov)"
        elif not r["significant"]:
            verdict = "ni značilen: ni dokazov za pomnjenje vrednosti"
        elif np.isnan(ref_delta):
            verdict = "značilen; brez reference ga ne moremo ločiti od splošne občutljivosti"
        elif ref_delta < cc18["delta"].median():
            verdict = "značilen in na referenci manjši: skladno s pomnjenjem (ni dokaz)"
        else:
            verdict = "značilen, a na referenci enako velik ali večji: splošna občutljivost na preslikavo"
        verdicts.append((ref_delta, verdict))
    tests["reference_delta"] = [v[0] for v in verdicts]
    tests["verdict"] = [v[1] for v in verdicts]

    report()
    report("Rezultat (enostranski Wilcoxon, H1: delta > 0; Holm znotraj modela; "
           f"alpha {alpha}):")
    for _, r in tests.iterrows():
        report(f"  {NAMES[r['model']]:11} {NAMES[r['arm']]:20} n={r['n']:2d}  mediana delta "
               f"{r['median_delta']:+.4f}  delta>0 na {r['n_positive']}/{r['n']}  p={r['p']:.4g}  "
               f"p_Holm={r['p_holm']:.4g}  referenca {r['reference_delta']:+.4f}")
        report(f"      -> {r['verdict']}")

    report()
    report("Preverbe (niso del testa):")
    ctrl = per_dataset[(per_dataset["model"] == leakage["primary"]) & (per_dataset["arm"] == "monotone")]
    if len(ctrl):
        report(f"  padec kontrole ({NAMES[leakage['control']]}) pri monotoni preslikavi: mediana "
               f"{ctrl['drop_control'].median():+.2e}, največji |padec| {ctrl['drop_control'].abs().max():.2e}")
    if args.check_against:
        check_against(df, args.check_against, "izvirna različica", report)
    if ref is not None and args.reference_check_against:
        check_against(ref, args.reference_check_against, "referenca, izvirna", report)
    if len(ref_df):
        report("  referenca po foldih (delta; popravljeni t-test je le informativen):")
        for _, r in ref_df.iterrows():
            report(f"    {NAMES[r['model']]:11} {NAMES[r['arm']]:20} {r['fold_deltas']}  "
                   f"p={r['corrected_p_two_sided']:.3g}")

    out_dir = os.path.join(run_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)
    per_dataset.to_csv(os.path.join(out_dir, "leakage_per_dataset.csv"), index=False)
    tests.to_csv(os.path.join(out_dir, "leakage_tests.csv"), index=False)
    if len(ref_df):
        ref_df.to_csv(os.path.join(out_dir, "leakage_reference.csv"), index=False)
    tex = tests.assign(model=tests["model"].map(NAMES), arm=tests["arm"].map(NAMES))[
        ["model", "arm", "n", "median_delta", "n_positive", "p", "p_holm", "reference_delta"]]
    with open(os.path.join(out_dir, "leakage_tests.tex"), "w", encoding="utf-8") as f:
        f.write(to_latex(
            tex, "Test uhajanja: delta = padec temeljnega modela minus padec CatBoosta "
                 "(ROC-AUC), enostranski Wilcoxonov test s Holmovim popravkom.",
            "tab:uhajanje", floatfmt=".4f",
            column_names=["Model", "Različica", "n", "Mediana delta", "delta > 0", "p",
                          "p (Holm)", "Medic3 delta"]))
    with open(os.path.join(out_dir, "leakage_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report.lines) + "\n")
    print(f"\nZapisano v {os.path.relpath(out_dir, REPO_ROOT)}/leakage_*")


if __name__ == "__main__":
    main()
