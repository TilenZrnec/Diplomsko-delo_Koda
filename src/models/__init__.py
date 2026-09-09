"""Register modelov: ime iz config.yaml -> modul z run() funkcijo.

Vsak modul izpostavi:
  * run(X_train, y_train, X_test, y_test, categorical_cols, random_state) -> dict
    s ključi model, roc_auc, train_time_s, inference_time_s, error,
    preprocessing, raw_error, device, proba;
  * USES_GPU (bool) - ali teče na GPU; runner take modele pred prvim merjenim
    učenjem ogreje, da fold 0 ne vsebuje nalaganja utež in inicializacije CUDA.
"""

from . import (
    catboost_model,
    lightgbm_model,
    random_forest,
    tabicl_model,
    tabpfn_model,
    xgboost_model,
)

REGISTRY = {
    "random_forest": random_forest,
    "xgboost": xgboost_model,
    "lightgbm": lightgbm_model,
    "catboost": catboost_model,
    "tabpfn": tabpfn_model,
    "tabicl": tabicl_model,
}
