"""XGBoost (privzeti hiperparametri).

Nativno obravnava manjkajoče vrednosti, zato numerični stolpci ostanejo
nespremenjeni. Kategorične spremenljivke se ordinalno kodirajo v številske
kode (NaN ostane NaN, da ga XGBoost obravnava nativno). Ordinalno kodiranje
nominalnim kategorijam vsili umeten vrstni red - to je del tega, kar merimo:
"najmanjša predobdelava, da algoritem vhod sploh sprejme", ne "najboljša".
"""

import time

import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from src.utils import compute_roc_auc, describe_device

USES_GPU = False

PREPROCESSING = (
    "nativna obravnava NaN (numerične ostanejo nespremenjene); kategorične "
    "stolpce ordinalno kodiramo v številske kode (NaN ohranjen)"
)


def run(X_train, y_train, X_test, y_test, categorical_cols, random_state):
    result = {
        "model": "XGBoost",
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

        if categorical_cols:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=np.nan,
            )
            X_train[categorical_cols] = enc.fit_transform(X_train[categorical_cols])
            X_test[categorical_cols] = enc.transform(X_test[categorical_cols])

        X_train = X_train.astype(float)
        X_test = X_test.astype(float)

        clf = XGBClassifier(random_state=random_state)

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        result["train_time_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = clf.predict_proba(X_test)
        result["inference_time_s"] = time.perf_counter() - t0

        result["roc_auc"] = compute_roc_auc(y_test, proba)
        result["proba"] = proba
    except Exception as e:
        result["error"] = str(e)
    return result
