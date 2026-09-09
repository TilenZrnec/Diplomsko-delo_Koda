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
"""

import os

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
    argumenti za pandas.read_csv, npr. {"sep": ";"}).
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


def load_dataset(spec, n_splits=5, random_state=42, cache_dir=None, n_repeats=1):
    """Naloži nabor (OpenML ali CSV) in pripravi razdelitve na folde.

    Vrne slovar: name, source ("openml"/"csv"), X (DataFrame), y (Series celih
    števil 0..K-1), classes (izvirne oznake razredov v vrstnem redu kod),
    categorical_cols (imena kategoričnih stolpcev) in folds (seznam parov
    (train_idx, test_idx) pozicijskih indeksov; pri n_repeats > 1 jih je
    n_repeats * n_splits, fold k pripada ponovitvi k // n_splits).
    """
    spec = _normalise_spec(spec)
    if spec["source"] == "openml":
        name, X, y, categorical_cols = _load_openml(spec, cache_dir)
    else:
        name, X, y, categorical_cols = _load_csv(spec)

    # Vrstice preštevilči na 0, 1, 2, ... - foldi vračajo pozicijske indekse.
    X = X.reset_index(drop=True)

    # Ciljne oznake (npr. "good"/"bad") pretvori v števila (0, 1, ...). To ni
    # uhajanje informacije: preslikava oznak ne uporabi ničesar iz atributov.
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y.reset_index(drop=True)), name="target")

    # Razdelitve: "stratified" ohrani razmerje razredov v vsakem foldu.
    # list() jih shrani, da jih lahko vsi algoritmi berejo večkrat.
    if n_repeats > 1:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(X, y))

    return {
        "name": name,
        "source": spec["source"],
        "X": X,
        "y": y,
        "classes": [str(c) for c in encoder.classes_],
        "categorical_cols": categorical_cols,
        "folds": folds,
    }
