"""Nalaganje datasetov iz OpenML in priprava skupnih 5-fold CV razdelitev.

Za vsak dataset se folde ustvari samo enkrat (StratifiedKFold, isti random_state),
nato jih ponovno uporabijo vsi algoritmi - s tem so rezultati med algoritmi
primerljivi na identičnih train/test razdelitvah.
"""

import os

import openml
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Pot do korena repozitorija (dve mapi navzgor od te datoteke) - da poti ne
# zapisujemo trdo in delujejo tudi, če repozitorij premaknemo.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(openml_id, n_splits=5, random_state=42, cache_dir=None):
    """Prenese/predpomni dataset iz OpenML in pripravi CV folde.

    Vrne slovar: name, X (DataFrame), y (LabelEncoded Series), categorical_cols
    (imena kategoričnih stolpcev po OpenML metapodatkih) in folds (seznam
    (train_idx, test_idx) parov pozicijskih indeksov).
    """
    # Nastavi mapo, kamor OpenML shrani prenesene datasete (predpomnilnik).
    if cache_dir is None:
        cache_dir = os.path.join(REPO_ROOT, "data", "openml_cache")
    openml.config.cache_directory = cache_dir

    # Prenese dataset z OpenML (ali ga prebere iz predpomnilnika).
    dataset = openml.datasets.get_dataset(openml_id)

    # Razdeli podatke na X (vhodni stolpci) in y (ciljni stolpec, ki ga napovedujemo).
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )

    # Vrstice preštevilči na 0, 1, 2, ... - folde vračajo pozicijske indekse.
    X = X.reset_index(drop=True)

    # Ciljne oznake (npr. "good"/"bad") pretvori v števila (0, 1, ...).
    y = pd.Series(LabelEncoder().fit_transform(y), name="target")

    # Izbere imena stolpcev, ki so kategorični (po metapodatkih OpenML).
    categorical_cols = [
        col for col, is_cat in zip(attribute_names, categorical_indicator) if is_cat
    ]

    # Ustvari 5 train/test razdelitev; "stratified" ohrani razmerje razredov
    # v vsakem foldu. list() jih shrani, da jih lahko vsi algoritmi berejo večkrat.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))

    # Vse skupaj vrne v slovarju, ki ga uporabi run_benchmark.py.
    return {
        "name": dataset.name,
        "X": X,
        "y": y,
        "categorical_cols": categorical_cols,
        "folds": folds,
    }
