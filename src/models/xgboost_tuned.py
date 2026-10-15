"""XGBoost z uglaševanjem hiperparametrov: iskanje po mreži z gnezdenim prečnim preverjanjem.

Poskus 4 (mentorjev načrt): koliko XGBoostu pomaga uglaševanje in ali s tem dohiti
temeljna modela. Vhod je natanko isti kot pri privzetem XGBoost
(xgboost_model.prepare, "kot iz škatle"); razlikujejo se le hiperparametri.

Postopek v enem zunanjem foldu (run() dobi le učni in testni del tega folda):
  1. Učni del razdeli na config["tuning"]["inner_splits"] notranjih foldov
     (StratifiedKFold, seme = random_state). Testni del pri izbiri NE sodeluje.
  2. Za vsako kombinacijo mreže brez n_estimators (24) in vsak notranji fold en
     model do največjega n_estimators; ROC-AUC na notranjem testnem delu se
     prebere pri vsakem n_estimators (iteration_range - model s k drevesi je
     natanko prvih k dreves večjega modela, preverjeno 2026-10-13). Skupaj
     32 x 4 = 128 nastavitev (config.yaml).
  3. Zmaga nastavitev z največjim povprečjem po notranjih foldih. Ob natančnem
     izenačenju zmaga prejšnja v vrstnem redu mreže: privzete vrednosti so v
     config.yaml prve, manj dreves pred več - privzeta nastavitev torej zmaga
     vsako izenačenje, v katerem je.
  4. Zmagovalec se nauči na CELEM učnem delu (natanko izbrano število dreves) in
     se enkrat oceni na testnem delu - to je rezultat.
  5. Za preverbo in primerjavo časov isti proces nauči še privzeti XGBoost
     (xgboost_model.run, isti klic kot v cc18_v4): njegov ROC-AUC mora biti enak
     vrstici v cc18_v4, njegov čas je izhodišče za ceno uglaševanja na istem stroju.

train_time_s = iskanje + ponovno učenje (cena uglaševanja), inference_time_s =
napoved zmagovalca. Podrobnosti gredo v stolpec hyperparameters (JSON) in v
tuning/<nabor>/xgboost_tuned_r<ponovitev>_f<fold>.csv (vse nastavitve mreže).
Viri: Probst, Boulesteix in Bischl 2019 (mreža); Cawley in Talbot 2010, Varma in
Simon 2006, Bischl in sod. 2023 (gnezdeno prečno preverjanje).
"""

import itertools
import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import load_config
from src.models import xgboost_model  # privzeti XGBoost: prepare() in preverba
from src.utils import compute_roc_auc, describe_device

USES_GPU = False

PREPROCESSING = (
    xgboost_model.PREPROCESSING
    + "; uglaševanje: iskanje po mreži (config.yaml: tuning), gnezdeno stratificirano "
    "prečno preverjanje na učnem delu, kodiranje vhoda v vsakem notranjem foldu posebej"
)


def grid_from_config(config=None):
    """(seznam kombinacij brez n_estimators, naraščajoči n_estimators, inner_splits)."""
    tuning = (config or load_config())["tuning"]
    grid = tuning["grid"]
    names = list(grid)
    models = [dict(zip(names, values)) for values in itertools.product(*(grid[n] for n in names))]
    return models, sorted(tuning["n_estimators"]), int(tuning["inner_splits"])


def search(X_train, y_train, categorical_cols, random_state, config=None):
    """Notranje prečno preverjanje vseh nastavitev; vrne tabelo, urejeno po vrstnem redu mreže."""
    models, rounds, inner_splits = grid_from_config(config)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    scores = {}  # (indeks kombinacije, n_estimators) -> seznam ROC-AUC po notranjih foldih
    for a, b in inner.split(X_train, y_train):
        X_a, X_b = xgboost_model.prepare(X_train.iloc[a], X_train.iloc[b], categorical_cols)
        y_a, y_b = y_train.iloc[a], y_train.iloc[b]
        for m, params in enumerate(models):
            clf = XGBClassifier(random_state=random_state, n_estimators=rounds[-1], **params)
            clf.fit(X_a, y_a)
            for k in rounds:
                proba = clf.predict_proba(X_b, iteration_range=(0, k))
                scores.setdefault((m, k), []).append(compute_roc_auc(y_b, proba, clf.classes_))
    rows = []
    for order, (m, k) in enumerate((m, k) for m in range(len(models)) for k in rounds):
        folds = scores[(m, k)]
        rows.append({**models[m], "n_estimators": k, "order": order,
                     **{f"inner_{j + 1}": v for j, v in enumerate(folds)},
                     "inner_mean": sum(folds) / len(folds)})
    table = pd.DataFrame(rows)
    # Rang: največje povprečje prej; ob izenačenju prejšnji v vrstnem redu mreže.
    ranked = table.sort_values(["inner_mean", "order"], ascending=[False, True], kind="stable")
    table.loc[ranked.index, "rank"] = range(1, len(table) + 1)
    table["rank"] = table["rank"].astype(int)
    return table, list(models[0]) + ["n_estimators"]


def run(X_train, y_train, X_test, y_test, categorical_cols, random_state):
    result = {
        "model": "XGBoost (uglašen)",
        "roc_auc": None,
        "train_time_s": None,
        "inference_time_s": None,
        "error": None,
        "preprocessing": PREPROCESSING,
        "raw_error": None,
        "device": describe_device(USES_GPU),
        "proba": None,
        "classes": None,
        "tuning": None,
        "tuning_table": None,
    }
    try:
        t0 = time.perf_counter()
        table, param_names = search(X_train, y_train, categorical_cols, random_state)
        search_s = time.perf_counter() - t0

        best = table.loc[table["rank"] == 1].iloc[0]
        chosen = {n: (int(best[n]) if n == "n_estimators" else float(best[n])) for n in param_names}
        default = table.loc[table["order"] == 0].iloc[0]  # prva točka mreže = privzete vrednosti

        X_tr, X_te = xgboost_model.prepare(X_train, X_test, categorical_cols)
        clf = XGBClassifier(random_state=random_state,
                            **{n: chosen[n] for n in param_names})
        t0 = time.perf_counter()
        clf.fit(X_tr, y_train)
        refit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = clf.predict_proba(X_te)
        result["inference_time_s"] = time.perf_counter() - t0
        result["train_time_s"] = search_s + refit_s
        result["classes"] = clf.classes_
        result["roc_auc"] = compute_roc_auc(y_test, proba, clf.classes_)
        result["proba"] = proba

        # Privzeti XGBoost na istem foldu: preverba proti cc18_v4 in izhodišče za čase.
        base = xgboost_model.run(X_train, y_train, X_test, y_test, categorical_cols, random_state)

        inner_cols = [c for c in table.columns if c.startswith("inner_") and c != "inner_mean"]
        result["tuning"] = {
            "chosen": chosen,
            "chosen_inner_mean": float(best["inner_mean"]),
            "chosen_inner_folds": [float(best[c]) for c in inner_cols],
            "default_inner_mean": float(default["inner_mean"]),
            "default_rank": int(default["rank"]),
            "default_test_roc_auc": base["roc_auc"],
            "default_train_time_s": base["train_time_s"],
            "default_error": base["error"],
            "n_configs": int(len(table)),
            "inner_splits": len(inner_cols),
            "search_time_s": search_s,
            "refit_time_s": refit_s,
        }
        result["tuning_table"] = table
    except Exception as e:
        result["error"] = str(e)
    return result
