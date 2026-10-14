"""RandomForest (sklearn, privzeti hiperparametri, privzeta obravnava vhoda).

scikit-learn od različice 1.4 manjkajoče vrednosti v RandomForest obravnava sam
(pri vsaki delitvi se nauči, v katero vejo gredo vrstice z NaN), zato jih ne
nadomeščamo. Nizov (kategorij) pa ne sprejme in nima parametra za kategorične
stolpce, zato kategorične stolpce ordinalno kodiramo; NaN ostane NaN. Kodirnik
se prilega samo na učni fold.

Do 2026-10-14 (zagon cc18_v3) smo manjkajoče vrednosti nadomeščali z mediano
oz. najpogostejšo vrednostjo, ker smo napačno domnevali, da RandomForest NaN ne
podpira. To ni bila najmanjša predobdelava in je na Medic3 (82 % manjkajočih)
modelu odvzelo informacijo o tem, kaj manjka (CLAUDE.md, Preprocessing policy ->
Audit).
"""

import time

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.utils import compute_roc_auc, describe_device

# Teče na CPU; runner ga ne ogreva (ni nalaganja utež ali CUDA inicializacije).
USES_GPU = False

PREPROCESSING = (
    "nativna obravnava NaN (scikit-learn >= 1.4); kategorične stolpce ordinalno "
    "kodiramo, ker RandomForest nizov ne sprejme (NaN ostane NaN, neznana "
    "kategorija -> -1); numerične pretvorimo v float"
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
        # V float le zaradi numeričnih OpenML stolpcev, ki se naložijo kot object;
        # pri stolpcih, ki so že števila, se vrednosti ne spremenijo.
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[numeric_cols] = X_train[numeric_cols].astype(float)
        X_test[numeric_cols] = X_test[numeric_cols].astype(float)

        # Vrstni red stolpcev (najprej numerični, nato kategorični) je enak kot v
        # cc18_v3, zato so rezultati na naborih brez manjkajočih vrednosti enaki
        # prejšnjim (RandomForest izbira atribute naključno po njihovem mestu).
        preprocessor = ColumnTransformer([
            ("num", "passthrough", numeric_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             categorical_cols),
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
