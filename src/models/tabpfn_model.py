"""TabPFN v3 (tabpfn paket >= 8, privzeti hiperparametri).

Eksperiment: podatke podamo v čim bolj surovi obliki (brez naše predobdelave)
in preverimo, kaj model obravnava nativno. Če surovi vhod sproži napako, to
zabeležimo (raw_error) in šele nato uporabimo minimalni popravek - ne
predobdelujemo tiho.

Ugotovljeno na dejanskih podatkih: TabPFN-3 surove podatke obravnava nativno
na vseh naborih CC18, vključno z 'object' stolpci z manjkajočimi vrednostmi
(npr. stolpec TBG v naboru 'sick'), kjer je TabPFN v2 (tabpfn 2.0.9) še
potreboval fallback. Fallback (ordinalno kodiranje kategoričnih stolpcev,
NaN ohranjen) je ohranjen kot varovalka za nove nabore.

Časi: klic fit() pri TabPFN ne uči ničesar (uteži so predhodno naučene), zato
je "čas učenja" majhen in ves strošek je v predict_proba. Prvi klic v procesu
poleg tega vsebuje nalaganje utež na GPU in inicializacijo CUDA; runner zato
pred prvim merjenim učenjem opravi en ogrevalni klic (USES_GPU = True).

Opomba: prva uporaba zahteva enkraten sprejem licence prek PriorLabs računa
(interaktivna prijava v brskalniku ali TABPFN_TOKEN okoljska spremenljivka).
"""

import time

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from src.utils import compute_roc_auc, describe_device

USES_GPU = True

RAW_PREPROCESSING = "raw (brez predobdelave)"
FALLBACK_PREPROCESSING = (
    "fallback: ordinalno kodiranje kategoričnih stolpcev v številske kode "
    "(NaN ohranjen); TabPFN-jeva interna predobdelava ne zna surovih "
    "string/object kategoričnih stolpcev pretvoriti v float"
)


def _fit_predict(clf, X_train, y_train, X_test):
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = clf.predict_proba(X_test)
    inf_time = time.perf_counter() - t0
    return proba, train_time, inf_time


def run(X_train, y_train, X_test, y_test, categorical_cols, random_state):
    from tabpfn import TabPFNClassifier

    result = {
        "model": "TabPFN",
        "roc_auc": None,
        "train_time_s": None,
        "inference_time_s": None,
        "error": None,
        "preprocessing": RAW_PREPROCESSING,
        "raw_error": None,
        "device": describe_device(USES_GPU),
        "proba": None,
    }

    try:
        clf = TabPFNClassifier(random_state=random_state)
        proba, train_time, inf_time = _fit_predict(clf, X_train, y_train, X_test)
    except Exception as e:
        result["raw_error"] = str(e)
        result["preprocessing"] = FALLBACK_PREPROCESSING
        try:
            X_train_fb = X_train.copy()
            X_test_fb = X_test.copy()
            if categorical_cols:
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=np.nan,
                )
                X_train_fb[categorical_cols] = enc.fit_transform(X_train_fb[categorical_cols])
                X_test_fb[categorical_cols] = enc.transform(X_test_fb[categorical_cols])
            X_train_fb = X_train_fb.astype(float)
            X_test_fb = X_test_fb.astype(float)

            clf = TabPFNClassifier(random_state=random_state)
            proba, train_time, inf_time = _fit_predict(clf, X_train_fb, y_train, X_test_fb)
        except Exception as e2:
            result["error"] = f"raw failed: {result['raw_error']}; fallback failed: {e2}"
            return result

    result["train_time_s"] = train_time
    result["inference_time_s"] = inf_time
    result["roc_auc"] = compute_roc_auc(y_test, proba)
    result["proba"] = proba
    return result
