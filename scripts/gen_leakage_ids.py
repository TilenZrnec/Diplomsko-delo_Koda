"""Zapiše sezname naborov za test uhajanja podatkov in (neobvezno) preveri preslikave.

Test uhajanja (CLAUDE.md, What is next, korak 8; zasnova in viri v
results/runs/cc18_leakage/DESIGN.md) vsak nabor požene v treh različicah iz
config.yaml (leakage: arms). Ta skript iz razdelka leakage zapiše:

  scripts/cc18_leakage.json        - 36 nenasičenih CC18 (prvih 36 iz cc18_tuning)
                                     BREZ naborov, ki nimajo nobenega številskega
                                     atributa (preslikava jih ne bi spremenila);
                                     različice istega nabora so zaporedne
  scripts/medic3_160_leakage.json  - referenčni Medic3-160 v istih treh različicah
  scripts/leakage_smoke.json       - sick v treh različicah (tehnični preizkus;
                                     nasičen, zato ni del testa)

Vsak opis je samozadosten (perturbation, noise_sd, perturbation_seed, repeats),
zato manifest zagona zapiše natanko to, kar je teklo.

Z --check za vsak nabor naloži vse tri različice po isti poti kot runner
(src.data.load_dataset) in preveri: foldi in oznake razredov so v vseh različicah
enaki in enaki foldom 0-4 diplomskih zagonov; kategorični atributi in vzorec
manjkajočih vrednosti so nespremenjeni; monotona različica ohrani vrstni red in
vezi vsakega atributa (rangi enaki); šumna različica premakne rang (ECDF) vrednosti
- izpiše povprečni premik. Rezultat zapiše v CSV (privzeto
results/runs/cc18_leakage/diagnostics/perturbation_check.csv). Uporablja le
podatke, nobenega modela - preverba ne razkrije nobenega rezultata testa.

Zagon (iz korena repozitorija; potrebuje predpomnilnik OpenML, za Medic3 še
data/medic3/Medic3.csv):
    python scripts/gen_leakage_ids.py [--check]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import dataset_specs, load_config, spec_id  # noqa: E402
from src.data import load_dataset  # noqa: E402

OUT_CC18 = os.path.join(REPO_ROOT, "scripts", "cc18_leakage.json")
OUT_MEDIC3 = os.path.join(REPO_ROOT, "scripts", "medic3_160_leakage.json")
OUT_SMOKE = os.path.join(REPO_ROOT, "scripts", "leakage_smoke.json")
DEFAULT_CHECK = os.path.join(REPO_ROOT, "results", "runs", "cc18_leakage", "diagnostics",
                             "perturbation_check.csv")


def arm_specs(base, leakage):
    """Opisi vseh različic enega nabora, v vrstnem redu leakage.arms."""
    if isinstance(base, int):
        base = {"source": "openml", "id": base}
    specs = []
    for arm in leakage["arms"]:
        spec = dict(base, perturbation=arm, repeats=int(leakage["repeats"]))
        if arm == "monotone_noise":
            spec.update(noise_sd=float(leakage["noise_sd"]), perturbation_seed=int(leakage["seed"]))
        specs.append(spec)
    return specs


def write_specs(path, specs):
    """En opis na vrstico - pregledno v diffu in v urejevalniku."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join("  " + json.dumps(s, ensure_ascii=False) for s in specs) + "\n]\n")


def load(spec, config):
    return load_dataset(spec, n_splits=config["n_splits"], random_state=config["random_state"],
                        cache_dir=config["cache_dir"], n_repeats=config["n_repeats"])


def check_dataset(specs, config):
    """Preveri vse različice enega nabora (glej docstring modula); vrne vrstico poročila."""
    leakage = config["leakage"]
    n_folds = int(leakage["repeats"]) * config["n_splits"]
    plain = {k: v for k, v in specs[0].items() if k not in ("perturbation", "repeats")}
    reference = load(plain if plain.get("source") == "csv" else int(plain["id"]), config)
    by_arm = {s["perturbation"]: load(s, config) for s in specs}
    orig = by_arm["original"]
    X0 = orig["X"]
    cat = orig["categorical_cols"]
    numeric = [c for c in X0.columns if c not in set(cat)]
    problems = []

    # Foldi: vse različice enake in enake prvim n_folds foldom diplomskih zagonov.
    thesis_folds = reference["folds"][:n_folds]
    for arm, d in by_arm.items():
        if len(d["folds"]) != n_folds:
            problems.append(f"{arm}: {len(d['folds'])} foldov namesto {n_folds}")
        for (tr, te), (tr0, te0) in zip(d["folds"], thesis_folds):
            if not (np.array_equal(tr, tr0) and np.array_equal(te, te0)):
                problems.append(f"{arm}: foldi niso enaki diplomskim")
                break
        if not d["y"].equals(reference["y"]):
            problems.append(f"{arm}: oznake razredov niso enake")
        if not d["X"][cat].equals(X0[cat]):
            problems.append(f"{arm}: kategorični atributi spremenjeni")
        if not d["X"][numeric].isna().equals(X0[numeric].isna()):
            problems.append(f"{arm}: vzorec manjkajočih vrednosti spremenjen")
    if not X0.equals(reference["X"]):
        problems.append("original: X ni enak izvirnemu naboru")

    mono = by_arm["monotone"]["X"]
    noisy = by_arm["monotone_noise"]["X"]
    rank_changed_mono, f32_collapsed, shifts, moved = 0, 0, [], 0
    n_cells = 0
    for col in numeric:
        x0 = X0[col].to_numpy(dtype=float, na_value=np.nan)
        obs = ~np.isnan(x0)
        if obs.sum() < 2:
            continue
        r0 = rankdata(x0[obs])
        x1 = mono[col].to_numpy(dtype=float)[obs]
        if not np.array_equal(rankdata(x1), r0):
            rank_changed_mono += 1
        # CatBoost in TabPFN delata s float32: ali preslikava spremeni število različnih
        # vrednosti po zaokrožitvi (zlije ali razdruži vezi)? Informativno, ne napaka.
        if len(np.unique(x0[obs].astype(np.float32))) != len(np.unique(x1.astype(np.float32))):
            f32_collapsed += 1
        r2 = rankdata(noisy[col].to_numpy(dtype=float)[obs])
        u0, u2 = (r0 - 0.5) / obs.sum(), (r2 - 0.5) / obs.sum()
        shifts.append(np.abs(u2 - u0))
        moved += int((r2 != r0).sum())
        n_cells += int(obs.sum())
    if rank_changed_mono:
        problems.append(f"monotone: vrstni red spremenjen v {rank_changed_mono} atributih")
    shift = np.concatenate(shifts) if shifts else np.array([np.nan])
    info_mono = by_arm["monotone"]["perturbation"]
    return {
        "dataset_id": spec_id(plain if plain.get("source") == "csv" else int(plain["id"])),
        "dataset": reference["name"],
        "n_rows": len(X0),
        "n_numeric": len(numeric),
        "n_transformed": info_mono["n_transformed"],
        "n_unchanged": info_mono["n_unchanged"],
        "n_categorical": len(cat),
        "float32_distinct_changed": f32_collapsed,
        "noise_mean_abs_ecdf_shift": float(np.nanmean(shift)),
        "noise_median_abs_ecdf_shift": float(np.nanmedian(shift)),
        "noise_share_cells_rank_moved": moved / n_cells if n_cells else float("nan"),
        "problems": "; ".join(problems),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Naloži vse različice in preveri preslikave (le podatki, brez modelov)")
    parser.add_argument("--check-out", default=DEFAULT_CHECK, help="CSV s poročilom preverbe")
    args = parser.parse_args()

    config = load_config()
    leakage = config["leakage"]
    if leakage["arms"][0] != "original":
        raise SystemExit("leakage.arms se mora začeti z 'original' (izhodišče za padec)")

    candidates = dataset_specs(config, leakage["source_set"])[: int(leakage["n_nonsaturated"])]
    kept, dropped = [], []
    for ds in candidates:
        d = load(ds, config)
        n_numeric = len([c for c in d["X"].columns if c not in set(d["categorical_cols"])])
        (kept if n_numeric > 0 else dropped).append((ds, d["name"], n_numeric))

    cc18_specs = [s for ds, _, _ in kept for s in arm_specs(ds, leakage)]
    reference = dataset_specs(config, leakage["reference_set"])
    if len(reference) != 1 or not isinstance(reference[0], dict):
        raise SystemExit(f"{leakage['reference_set']}: pričakovan natanko en opis nabora")
    medic3_specs = arm_specs(reference[0], leakage)
    smoke_specs = arm_specs(int(leakage["smoke_dataset"]), leakage)
    for specs in (cc18_specs, medic3_specs, smoke_specs):
        ids = [spec_id(s) for s in specs]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"Podvojeni identifikatorji različic: {ids}")

    write_specs(OUT_CC18, cc18_specs)
    write_specs(OUT_MEDIC3, medic3_specs)
    write_specs(OUT_SMOKE, smoke_specs)
    n_arms = len(leakage["arms"])
    print(f"{len(kept)} naborov x {n_arms} različice = {len(cc18_specs)} opisov -> "
          f"{os.path.relpath(OUT_CC18, REPO_ROOT)} (SLURM --array=0-{len(kept) - 1})")
    for k, (ds, name, n_numeric) in enumerate(kept):
        print(f"  {k:2d}  {ds:6d}  {name}  ({n_numeric} številskih atributov)")
    print("Izpadli (brez številskega atributa, preslikava jih ne spremeni): "
          + (", ".join(f"{ds} {name}" for ds, name, _ in dropped) or "nobeden"))
    print(f"{len(medic3_specs)} opisi -> {os.path.relpath(OUT_MEDIC3, REPO_ROOT)}; "
          f"{len(smoke_specs)} opisi -> {os.path.relpath(OUT_SMOKE, REPO_ROOT)}")

    if not args.check:
        return
    groups = [cc18_specs[i:i + n_arms] for i in range(0, len(cc18_specs), n_arms)]
    medic3_path = os.path.join(REPO_ROOT, reference[0]["path"])
    if os.path.exists(medic3_path):
        groups.append(medic3_specs)
    else:
        print(f"OPOZORILO: {reference[0]['path']} ni na tem stroju - Medic3 ni preverjen.")
    rows = []
    for specs in groups:
        row = check_dataset(specs, config)
        rows.append(row)
        print(f"  {row['dataset_id']:>10}  {row['dataset'][:34]:34}  atributi {row['n_transformed']:4d}/"
              f"{row['n_numeric']:<4d}  f32 {row['float32_distinct_changed']:3d}  "
              f"premik ranga {row['noise_mean_abs_ecdf_shift']:.4f}  {row['problems'] or 'OK'}")
    report = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.check_out), exist_ok=True)
    report.to_csv(args.check_out, index=False)
    print(f"Poročilo: {os.path.relpath(args.check_out, REPO_ROOT)}")
    bad = report[report.problems != ""]
    if len(bad):
        raise SystemExit(f"NAPAKA: težave pri {len(bad)} naborih (stolpec problems)")
    print("Vse preverbe uspele.")


if __name__ == "__main__":
    main()
