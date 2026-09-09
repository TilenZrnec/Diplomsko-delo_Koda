"""LightGBM (privzeti hiperparametri).

Nativno obravnava manjkajoče vrednosti. Kategorične spremenljivke se
pretvorijo v pandas 'category' dtype, kar LightGBM prepozna kot nativne
kategorične stolpce (brez ročnega kodiranja).
"""

import time

from lightgbm import LGBMClassifier

from src.utils import compute_roc_auc, describe_device

USES_GPU = False

PREPROCESSING = (
    "nativna obravnava NaN; kategorične stolpce pretvorimo v pandas "
    "'category' dtype za nativno kategorično podporo LightGBM; preostale "
    "stolpce eksplicitno pretvorimo v float (nekateri numerični OpenML "
    "atributi se zaradi manjkajočih vrednosti naložijo kot object dtype)"
)


def run(X_train, y_train, X_test, y_test, categorical_cols, random_state):
    result = {
        "model": "LightGBM",
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

        numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
        # nekateri OpenML atributi so uradno numerični, a se zaradi manjkajočih
        # vrednosti naložijo kot object dtype (npr. stolpec, ki je 100 % NaN) -
        # LightGBM sprejme le int/float/bool/category, zato jih eksplicitno pretvorimo
        X_train[numeric_cols] = X_train[numeric_cols].astype(float)
        X_test[numeric_cols] = X_test[numeric_cols].astype(float)

        for c in categorical_cols:
            X_train[c] = X_train[c].astype("category")
            X_test[c] = X_test[c].astype("category").cat.set_categories(X_train[c].cat.categories)

        clf = LGBMClassifier(random_state=random_state, verbosity=-1)

        t0 = time.perf_counter()
        clf.fit(X_train, y_train, categorical_feature=categorical_cols or "auto")
        result["train_time_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = clf.predict_proba(X_test)
        result["inference_time_s"] = time.perf_counter() - t0

        result["roc_auc"] = compute_roc_auc(y_test, proba)
        result["proba"] = proba
    except Exception as e:
        result["error"] = str(e)
    return result
