"""TabPFN-3.5 (paket tabpfn 9.1, privzeti hiperparametri).

Model: od zamrznitve 2026-10-05 je to TabPFN-3.5, ki je zamenjal TabPFN-3 iz
zagona cc18_v2 (tabpfn 8.5.0). Različico modela izberemo IZRECNO
(make_classifier, ModelVersion.V3_5), čeprav je v tabpfn 9 že privzeta: paket
privzeto različico bere iz nastavitev (tudi iz okoljske spremenljivke
TABPFN_MODEL_VERSION) in jo je med 8.5.0 in 9.0.0 že enkrat tiho zamenjal
(V3 -> V3_5). create_default_for_version nastavi le pot do uteži in
n_estimators="auto", kar je enako privzetemu konstruktorju; vse ostalo je
privzeto (tudi softmax_temperature="auto", torej temperatura, ki jo določa
kontrolna točka - v tabpfn 8.5.0 je bila privzeta fiksna 0,9).

Eksperiment: podatke podamo v čim bolj surovi obliki (brez naše predobdelave)
in preverimo, kaj model obravnava nativno. Če surovi vhod sproži napako, to
zabeležimo (raw_error) in šele nato uporabimo minimalni popravek - ne
predobdelujemo tiho. Pri TabPFN-3 je surovi vhod deloval na vseh naborih CC18;
fallback (ordinalno kodiranje kategoričnih stolpcev, NaN ohranjen) ostaja kot
varovalka za nove nabore.

Časi: klic fit() pri TabPFN ne uči ničesar (uteži so predhodno naučene), zato
je "čas učenja" majhen in ves strošek je v predict_proba. Prvi klic v procesu
poleg tega vsebuje nalaganje utež na GPU in inicializacijo CUDA; runner zato
pred prvim merjenim učenjem opravi en ogrevalni klic (USES_GPU = True).

Opomba: prenos utež zahteva enkraten sprejem licence prek PriorLabs računa
(interaktivna prijava v brskalniku ali TABPFN_TOKEN okoljska spremenljivka).
Uteži TabPFN-3.5 so pod nekomercialno licenco, ki dovoljuje "testing,
evaluation, and internal benchmarking". Različici Plus in Thinking sta na voljo
le prek API-ja Prior Labs in ju ne uporabljamo: računska vozlišča so brez
interneta, zaupni Medic3 pa ne sme zapustiti naših strojev.
"""

import os
import time

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from src.utils import compute_roc_auc, describe_device

USES_GPU = True

MODEL_LABEL = "TabPFN-3.5"

RAW_PREPROCESSING = "raw (brez predobdelave)"
FALLBACK_PREPROCESSING = (
    "fallback: ordinalno kodiranje kategoričnih stolpcev v številske kode "
    "(NaN ohranjen); TabPFN-jeva interna predobdelava ne zna surovih "
    "string/object kategoričnih stolpcev pretvoriti v float"
)


def make_classifier(random_state, **overrides):
    """TabPFNClassifier z izrecno izbranim modelom TabPFN-3.5 in privzetimi nastavitvami.

    Edino mesto, kjer nastane klasifikator: uporabljata ga run() spodaj in
    scripts/prestage.py, zato predpriprava prenese natanko uteži, ki jih nato
    uporabi benchmark.
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5, random_state=random_state, **overrides
    )


def checkpoint_name():
    """Ime datoteke z utežmi, ki jih uporabi make_classifier (uteži se ne naložijo).

    pip freeze v manifest.json pove le različico paketa tabpfn, ne pa, katere
    uteži je paket naložil; runner to ime zato zapiše v manifest posebej.
    """
    return os.path.basename(str(make_classifier(0).model_path))


def _fit_predict(clf, X_train, y_train, X_test):
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = clf.predict_proba(X_test)
    inf_time = time.perf_counter() - t0
    return proba, train_time, inf_time


def run(X_train, y_train, X_test, y_test, categorical_cols, random_state):
    result = {
        "model": MODEL_LABEL,
        "roc_auc": None,
        "train_time_s": None,
        "inference_time_s": None,
        "error": None,
        "preprocessing": RAW_PREPROCESSING,
        "raw_error": None,
        "device": describe_device(USES_GPU),
        "proba": None,
        "classes": None,
    }

    try:
        clf = make_classifier(random_state)
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

            clf = make_classifier(random_state)
            proba, train_time, inf_time = _fit_predict(clf, X_train_fb, y_train, X_test_fb)
        except Exception as e2:
            result["error"] = f"raw failed: {result['raw_error']}; fallback failed: {e2}"
            return result

    result["train_time_s"] = train_time
    result["inference_time_s"] = inf_time
    # V try, ker mora tudi napaka pri izračunu metrike pristati v stolpcu error
    # in ne podreti celotnega opravila (pravilo "vsak model odpove mehko").
    try:
        result["classes"] = clf.classes_
        result["roc_auc"] = compute_roc_auc(y_test, proba, clf.classes_)
        result["proba"] = proba
    except Exception as e:
        result["error"] = f"izračun ROC-AUC ni uspel: {e}"
    return result
