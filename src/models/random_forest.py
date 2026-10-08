"""RandomForest (sklearn, privzeti hiperparametri).

Nima nativne podpore za manjkajoče vrednosti ali kategorične spremenljivke,
zato: median imputacija (numerične), most-frequent imputacija + ordinalno
kodiranje (kategorične). Predobdelava se prilega samo na train fold.
"""

import time

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.utils import compute_roc_auc, describe_device

# Teče na CPU; runner ga ne ogreva (ni nalaganja utež ali CUDA inicializacije).
USES_GPU = False

PREPROCESSING = (
    "median imputacija (numerične) + most-frequent imputacija in ordinalno "
    "kodiranje (kategorične); RandomForest nima nativne podpore za NaN/kategorije"
)


def run(X_train, y_train, X_test, y_test, categorical_cols, random_state):
    result = {
        "model": "RandomForest",
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
        numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical_cols),
        ])
        clf = Pipeline([
            ("preprocess", preprocessor),
            # n_jobs=-1 ni hiperparameter modela, ampak nastavitev računanja:
            # drevesa so neodvisna, zato so naučena drevesa bitno identična
            # serijskim. Vzporedni predict_proba pa verjetnosti dreves sešteva v
            # vrstnem redu, ki se med izvedbami razlikuje, zato se ROC-AUC med
            # ponovljenimi izvedbami lahko razlikuje na peti decimalki
            # (preverjeno 2026-10-08 na cmc; cc18_v2 proti cc18_v3: 31/1080
            # učenj, največ 2.3e-5). Brez tega RF kot edini od štirih ansamblov
            # uporablja eno jedro od osmih.
            ("model", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        ])

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
