"""XGBoost (privzeti hiperparametri, privzeta obravnava vhoda).

xgboost 3.x ima privzeto enable_categorical=True: stolpce pandas tipa
'category' obravnava sam (nativne kategorične delitve), manjkajoče vrednosti
prav tako. Zato kategoričnih stolpcev ne kodiramo, ampak jih podamo kot
'category' (OpenML jih tako že vrne). Predobdelava je le tista, brez katere
xgboost vhoda ne sprejme (glej prepare()).

Do 2026-10-14 (zagon cc18_v3) smo kategorične stolpce ordinalno kodirali v
številske kode. To je bil najmanjši korak v xgboost 2.1.1, kjer je bil
enable_categorical privzeto False, v xgboost 3.4.1 pa ni več - zato smo modul
popravili, da XGBoost teče "kot iz škatle" (CLAUDE.md, Preprocessing policy ->
Audit).
"""

import time

from xgboost import XGBClassifier

from src.utils import compute_roc_auc, describe_device

USES_GPU = False

PREPROCESSING = (
    "nativna obravnava NaN in kategoričnih stolpcev (xgboost 3: privzeto "
    "enable_categorical=True); kategorične stolpce podamo kot pandas 'category' "
    "(testni fold z istimi kategorijami kot učni), numerične pretvorimo v float"
)


def prepare(X_train, X_test, categorical_cols):
    """Najmanjša predobdelava, brez katere xgboost vhoda ne sprejme.

    Numerične stolpce pretvorimo v float, ker xgboost stolpcev tipa object ne
    sprejme (nekateri numerični OpenML atributi se zaradi manjkajočih vrednosti
    naložijo kot object); pri stolpcih, ki so že števila, se vrednosti ne
    spremenijo. Kategorični stolpci postanejo 'category'; testni fold dobi
    kategorije učnega folda, da kode v obeh pomenijo isto. Pri OpenML naborih so
    stolpci že 'category' z istim naborom kategorij, zato je to tam brez učinka.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
    X_train[numeric_cols] = X_train[numeric_cols].astype(float)
    X_test[numeric_cols] = X_test[numeric_cols].astype(float)
    for c in categorical_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category").cat.set_categories(X_train[c].cat.categories)
    return X_train, X_test


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
        "classes": None,
    }
    try:
        X_train, X_test = prepare(X_train, X_test, categorical_cols)

        clf = XGBClassifier(random_state=random_state)

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        result["train_time_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = clf.predict_proba(X_test)
        result["inference_time_s"] = time.perf_counter() - t0

        # classes_ so oznake razredov v vrstnem redu stolpcev proba. Pri naborih
        # z zelo redkimi razredi jih je lahko manj kot v celotnem naboru (razreda,
        # ki ga v učnem foldu ni, model ne pozna), zato jih metrika potrebuje.
        result["classes"] = clf.classes_
        result["roc_auc"] = compute_roc_auc(y_test, proba, clf.classes_)
        result["proba"] = proba
    except Exception as e:
        result["error"] = str(e)
    return result
