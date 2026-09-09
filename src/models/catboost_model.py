"""CatBoost (privzeti hiperparametri).

Nativno obravnava manjkajoče vrednosti (numerične) in kategorične
spremenljivke (podane preko cat_features). CatBoost ne dovoli float NaN v
kategoričnih stolpcih, zato jih pretvorimo v string ('nan' postane lastna
kategorija) - to ni imputacija, samo tipska pretvorba.

CatBoost privzeto piše dnevnik učenja v mapo catboost_info/ v trenutni
delovni mapi (verbose=False utiša samo stdout, ne pisanja datotek), zato
to izklopimo z allow_writing_files=False - drugače vsak zagon, tudi vsak
SLURM array task na Arnesu, pusti za sabo mapo s smetmi.
"""

import time

import pandas as pd
from catboost import CatBoostClassifier

from src.utils import compute_roc_auc, describe_device

USES_GPU = False

PREPROCESSING = (
    "nativna obravnava NaN (numerične); kategorične stolpce pretvorimo v "
    "string (NaN -> 'nan' kot lastna kategorija) in podamo kot cat_features"
)


def run(X_train, y_train, X_test, y_test, categorical_cols, random_state):
    result = {
        "model": "CatBoost",
        "roc_auc": None,
        "train_time_s": None,
        "inference_time_s": None,
        "error": None,
        "preprocessing": PREPROCESSING,
        "raw_error": None,
        "device": describe_device(USES_GPU),
        "proba": None,
    }
    try:
        X_train = X_train.copy()
        X_test = X_test.copy()

        # Vsako vrednost pretvorimo v niz, NaN v niz 'nan'. Od pandas 3.0 naprej
        # astype(str) manjkajočih vrednosti NE pretvori več v 'nan', ampak jih
        # pusti manjkajoče, CatBoost pa NaN v kategoričnem stolpcu zavrne - zato
        # eksplicitna preslikava po elementih (astype(object) najprej razveže
        # 'category' dtype, sicer bi map preskočil manjkajoče vrednosti).
        def _to_str(col):
            return col.astype(object).map(lambda v: "nan" if pd.isna(v) else str(v))

        for c in categorical_cols:
            X_train[c] = _to_str(X_train[c])
            X_test[c] = _to_str(X_test[c])

        clf = CatBoostClassifier(
            random_state=random_state, verbose=False, allow_writing_files=False
        )

        t0 = time.perf_counter()
        clf.fit(X_train, y_train, cat_features=categorical_cols)
        result["train_time_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = clf.predict_proba(X_test)
        result["inference_time_s"] = time.perf_counter() - t0

        result["roc_auc"] = compute_roc_auc(y_test, proba)
        result["proba"] = proba
    except Exception as e:
        result["error"] = str(e)
    return result
