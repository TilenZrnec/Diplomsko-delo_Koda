"""TabICL (tabicl paket, privzeti hiperparametri).

Eksperiment: podatke podamo v čim bolj surovi obliki. Če surovi vhod sproži
napako, to zabeležimo (raw_error) in šele nato uporabimo minimalni popravek.

Ugotovljeno na dejanskih podatkih: s torch 2.4.1 je privzeta nastavitev
device='auto' sprožila napako ('Expected a torch.device with a specified
index or an integer, but got: cuda') in je bil potreben fallback z
eksplicitnim device='cuda:0'. Od torch 2.13 naprej surovi zagon deluje brez
fallbacka; fallback je ohranjen kot varovalka.

Časi: enako kot pri TabPFN je učenje le shranjevanje konteksta, strošek je v
predict_proba; prvi klic v procesu vsebuje nalaganje utež, zato runner pred
prvim merjenim učenjem opravi ogrevalni klic (USES_GPU = True).
"""

import time

import torch

from src.utils import compute_roc_auc, describe_device

USES_GPU = True

RAW_PREPROCESSING = "raw (brez predobdelave)"


def _fit_predict(clf, X_train, y_train, X_test):
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = clf.predict_proba(X_test)
    inf_time = time.perf_counter() - t0
    return proba, train_time, inf_time


def run(X_train, y_train, X_test, y_test, categorical_cols, random_state):
    from tabicl import TabICLClassifier

    result = {
        "model": "TabICL",
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
        clf = TabICLClassifier(random_state=random_state)
        proba, train_time, inf_time = _fit_predict(clf, X_train, y_train, X_test)
    except Exception as e:
        result["raw_error"] = str(e)
        fallback_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        result["preprocessing"] = (
            f"fallback: eksplicitno device='{fallback_device}' (privzeti "
            "device='auto' ne razreši torch.device indeksa v tem okolju; "
            "podatki ostanejo nespremenjeni)"
        )
        try:
            clf = TabICLClassifier(random_state=random_state, device=fallback_device)
            proba, train_time, inf_time = _fit_predict(clf, X_train, y_train, X_test)
        except Exception as e2:
            result["error"] = f"raw failed: {result['raw_error']}; fallback failed: {e2}"
            return result

    result["train_time_s"] = train_time
    result["inference_time_s"] = inf_time
    result["roc_auc"] = compute_roc_auc(y_test, proba)
    result["proba"] = proba
    return result
