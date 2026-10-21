"""Nalaganje podatkovnih množic in priprava skupnih razdelitev na folde.

Dva vira podatkov, en sam izhodni format:

  * OpenML  - opis je celo število (ID) ali {"source": "openml", "id": 31};
              nabor se prenese in predpomni v cache_dir.
  * CSV     - opis je {"source": "csv", "path": ..., "target": ..., ...};
              lokalna datoteka (tudi .zip s CSV-jem, pandas ga bere neposredno).
              Tako v benchmark pride zaupni nabor Medic3, ki ni na OpenML.

Ne glede na vir vrne load_dataset() isti slovar, zato ostale kode ne zanima,
od kod nabor prihaja.

Folde za dani nabor ustvari SAMO ENKRAT (StratifiedKFold oz. RepeatedStratifiedKFold
z istim semenom), nato jih uporabijo vsi algoritmi - rezultati so tako
primerljivi na identičnih razdelitvah na učne in testne primere.

Neobvezni ključi v opisu nabora (veljajo za oba vira):
  max_classes   - obdrži le N najpogostejših razredov (_limit_classes)
  perturbation  - različica nabora za test uhajanja (_perturb): original,
                  monotone ali monotone_noise; z njo še noise_sd in
                  perturbation_seed
  repeats       - uporabi le prvih `repeats` ponovitev prečnega preverjanja
                  (foldi so isti kot v diplomskih zagonih, le manj jih je)
"""

import os

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Pot do korena repozitorija (dve mapi navzgor od te datoteke) - da poti ne
# zapisujemo trdo in delujejo tudi, če repozitorij premaknemo.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _normalise_spec(spec):
    """Celo število pretvori v {"source": "openml", "id": n}; slovar preveri."""
    if isinstance(spec, int):
        return {"source": "openml", "id": spec}
    if isinstance(spec, dict):
        spec = dict(spec)
        spec.setdefault("source", "openml")
        if spec["source"] not in ("openml", "csv"):
            raise ValueError(f"Neznan vir nabora: {spec['source']!r}")
        return spec
    raise TypeError(f"Opis nabora mora biti int ali dict, ne {type(spec).__name__}")


def _load_openml(spec, cache_dir):
    """Prenese/predpomni nabor z OpenML in vrne (ime, X, y, kategorični stolpci)."""
    if cache_dir is None:
        cache_dir = os.path.join(REPO_ROOT, "data", "openml_cache")
    # Nastavitev je "root": openml pod njo sam doda org/openml/www, zato datoteke
    # končajo v <cache_dir>/org/openml/www. Pripis openml.config.cache_directory
    # je v openml >= 0.14 brez učinka (glej CLAUDE.md) - ne uporabljaj ga.
    openml.config.set_root_cache_directory(cache_dir)

    dataset = openml.datasets.get_dataset(int(spec["id"]))
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    categorical_cols = [
        col for col, is_cat in zip(attribute_names, categorical_indicator) if is_cat
    ]
    return dataset.name, X, y, categorical_cols


def _load_csv(spec):
    """Prebere lokalni CSV (ali .zip s CSV-jem) in vrne (ime, X, y, kategorični stolpci).

    Obvezni ključi: path, target. Neobvezni: name, drop_cols (stolpci, ki niso
    atributi, npr. zaporedna številka vrstice), categorical_cols (če manjka, so
    kategorični vsi stolpci z ne-številskimi vrednostmi), read_csv (dodatni
    argumenti za pandas.read_csv, npr. {"sep": ";"}). Ključ max_classes obdela
    load_dataset(), ker velja enako za OpenML in CSV vir.
    """
    path = spec["path"]
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(REPO_ROOT, path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nabor {spec.get('name', path)}: datoteka {path} ne obstaja")

    df = pd.read_csv(path, low_memory=False, **spec.get("read_csv", {}))
    target = spec["target"]
    drop_cols = [c for c in spec.get("drop_cols", []) if c in df.columns]
    y = df[target]
    X = df.drop(columns=[target] + drop_cols)

    if "categorical_cols" in spec:
        categorical_cols = [c for c in spec["categorical_cols"] if c in X.columns]
    else:
        categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    name = spec.get("name") or os.path.splitext(os.path.basename(path))[0]
    return name, X, y, categorical_cols


def _limit_classes(X, y, max_classes):
    """Obdrži samo vrstice max_classes najpogostejših razredov; vrne (X, y, opis).

    Zakaj: TabPFN v3 ima trdo mejo 160 razredov, Medic3 pa jih ima 220, zato na
    surovem naboru sploh ne steče. Da je primerjava vseh šestih algoritmov na
    istih podatkih mogoča, naredimo drugo različico nabora, omejeno na 160
    najpogostejših razredov. Vrstice ostalih razredov zavržemo (razredov NE
    zlivamo v skupni razred 'drugo', ker bi bilo teh razredov spet 161).

    Vrstni red: po pogostosti padajoče, ob enakem številu primerov po oznaki
    naraščajoče - da je izbor determinističen in ponovljiv.
    """
    counts = y.value_counts()
    order = sorted(counts.index, key=lambda c: (-counts[c], str(c)))
    keep = set(order[:max_classes])
    mask = y.isin(keep)
    info = {
        "max_classes": max_classes,
        "n_classes_before": int(len(counts)),
        "n_classes_after": int(len(keep)),
        "n_rows_before": int(len(y)),
        "n_rows_after": int(mask.sum()),
        "n_rows_dropped": int((~mask).sum()),
    }
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), info


# Različice nabora za test uhajanja podatkov (CLAUDE.md, What is next, korak 8;
# zasnova in viri v results/runs/cc18_leakage/DESIGN.md).
PERTURBATIONS = ("original", "monotone", "monotone_noise")


def _perturb(X, categorical_cols, arm, noise_sd=None, seed=None):
    """Spremeni številske atribute za test uhajanja; vrne (X, opis).

    original        - nič; izvirni podatki tečejo v istem zagonu kot ostali dve
                      različici, zato razlike med njimi niso razlike med stroji.
    monotone        - vsak številski atribut standardiziramo, z = (x - povprečje) / std,
                      in preslikamo s predznačenim kubičnim korenom cbrt(z), strogo
                      naraščajočo funkcijo (Biloš in sod. 2026, tabela 7). Vrstni red
                      in vezi ostanejo, spremenijo se razmiki med vrednostmi.
    monotone_noise  - kot monotone, le da pred korenom prištejemo šum N(0, noise_sd^2)
                      v enotah std atributa (en žreb na nabor, seme seed). Šum premeša
                      tudi vrstni red bližnjih vrednosti, torej rang, ki ga TabPFN-3.5
                      bere kot ločen kanal (poročilo TabPFN-3.5, razd. 3.1).

    Zakaj centriramo: cbrt ni neodvisen od premika - na atributu daleč od nič
    (npr. vrednosti 1000-1010) je skoraj linearen, in ker model vsak atribut sam
    standardizira, bi sprememba skoraj izginila. Deljenje s std obliko le
    poenoti (cbrt(c x) = cbrt(c) cbrt(x), model ta faktor sam izniči).

    Manjkajoče vrednosti ostanejo manjkajoče; kategorični, konstantni in povsem
    prazni atributi ostanejo nespremenjeni (prazen atribut ima pandas lahko kot
    tip object, npr. TBG v naboru sick). Povprečje in std sta izračunana na vseh
    vrsticah pred delitvijo na folde: preslikava ne uporabi oznak razreda in je
    za vse vrstice ista, torej je le drug zapis istega nabora (ne uhajanje v
    našem cevovodu).
    """
    if arm not in PERTURBATIONS:
        raise ValueError(f"Neznana različica {arm!r}; znane so {PERTURBATIONS}")
    categorical = set(categorical_cols)
    numeric = [c for c in X.columns if c not in categorical]
    info = {
        "arm": arm,
        "noise_sd": None,
        "seed": None,
        "n_numeric": len(numeric),
        "n_transformed": 0,
        "n_unchanged": 0,
        "n_noised_cells": 0,
    }
    if arm == "original":
        return X, info

    noisy = arm == "monotone_noise"
    if noisy:
        if noise_sd is None or not float(noise_sd) > 0 or seed is None:
            raise ValueError("Različica monotone_noise potrebuje noise_sd > 0 in perturbation_seed")
        info["noise_sd"], info["seed"] = float(noise_sd), int(seed)
        rng = np.random.default_rng(int(seed))

    new_cols = {}
    for col in numeric:
        if X[col].isna().all():
            info["n_unchanged"] += 1
            continue
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise TypeError(f"Atribut {col!r} ni kategoričen, a tudi ni številski ({X[col].dtype})")
        x = X[col].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(x).any():
            raise ValueError(f"Atribut {col!r} vsebuje neskončne vrednosti - preslikava zanje ni določena")
        observed = ~np.isnan(x)
        sd = x[observed].std()
        if not sd > 0:
            info["n_unchanged"] += 1
            continue
        z = (x - x[observed].mean()) / sd
        if noisy:
            z[observed] += float(noise_sd) * rng.standard_normal(int(observed.sum()))
            info["n_noised_cells"] += int(observed.sum())
        new_cols[col] = np.cbrt(z)  # realni kubični koren: liho in strogo naraščajoče, NaN ostane NaN
        info["n_transformed"] += 1

    if new_cols:
        X = X.copy()
        X[list(new_cols)] = pd.DataFrame(new_cols, index=X.index)
    return X, info


def load_dataset(spec, n_splits=5, random_state=42, cache_dir=None, n_repeats=1):
    """Naloži nabor (OpenML ali CSV) in pripravi razdelitve na folde.

    Vrne slovar: name, source ("openml"/"csv"), X (DataFrame), y (Series celih
    števil 0..K-1), classes (izvirne oznake razredov v vrstnem redu kod),
    categorical_cols (imena kategoričnih stolpcev), folds (seznam parov
    (train_idx, test_idx) pozicijskih indeksov; pri n_repeats > 1 jih je
    n_repeats * n_splits, fold k pripada ponovitvi k // n_splits), class_limit
    (None ali opis omejitve iz ključa max_classes) in perturbation (None ali
    opis različice za test uhajanja iz ključa perturbation, glej _perturb).
    """
    spec = _normalise_spec(spec)
    if spec["source"] == "openml":
        name, X, y, categorical_cols = _load_openml(spec, cache_dir)
    else:
        name, X, y, categorical_cols = _load_csv(spec)

    # Vrstice preštevilči na 0, 1, 2, ... - foldi vračajo pozicijske indekse.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Neobvezna omejitev na N najpogostejših razredov (glej _limit_classes).
    # Zgodi se PRED kodiranjem oznak, da so kode spet zvezne 0..N-1.
    class_limit = None
    if spec.get("max_classes"):
        X, y, class_limit = _limit_classes(X, y, int(spec["max_classes"]))

    # Neobvezna različica za test uhajanja (glej _perturb). Spremeni le X, zato so
    # stratificirani foldi spodaj za vse različice istega nabora enaki. Ime dobi
    # pripono, da se različice v results.csv nikoli ne zlijejo v en nabor.
    perturbation = None
    if spec.get("perturbation") is not None:
        X, perturbation = _perturb(
            X, categorical_cols, spec["perturbation"],
            noise_sd=spec.get("noise_sd"), seed=spec.get("perturbation_seed"),
        )
        name = f"{name}_{spec['perturbation']}"

    # Ciljne oznake (npr. "good"/"bad") pretvori v števila (0, 1, ...). To ni
    # uhajanje informacije: preslikava oznak ne uporabi ničesar iz atributov.
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y), name="target")

    # Razdelitve: "stratified" ohrani razmerje razredov v vsakem foldu.
    # list() jih shrani, da jih lahko vsi algoritmi berejo večkrat.
    if n_repeats > 1:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(X, y))

    # Neobvezno le prvih `repeats` ponovitev. Foldi so ustvarjeni z vsemi n_repeats
    # in nato odrezani, zato so foldi 0..repeats*n_splits-1 natanko foldi diplomskih
    # zagonov (isti učni in testni primeri, isto seme modela random_state + ponovitev).
    if spec.get("repeats") is not None:
        repeats = int(spec["repeats"])
        if not 1 <= repeats <= n_repeats:
            raise ValueError(f"repeats={repeats} mora biti med 1 in n_repeats={n_repeats}")
        folds = folds[: repeats * n_splits]

    return {
        "name": name,
        "source": spec["source"],
        "X": X,
        "y": y,
        "classes": [str(c) for c in encoder.classes_],
        "categorical_cols": categorical_cols,
        "folds": folds,
        "class_limit": class_limit,
        "perturbation": perturbation,
    }
