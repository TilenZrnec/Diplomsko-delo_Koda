"""Revizija: ali vsak algoritem teče "kot iz škatle" (privzeti hiperparametri,
le predobdelava, brez katere knjižnica vhoda ne sprejme).

Dva dela:
  1. Nastavitve in vhod. Za vsak algoritem iz REGISTRY požene run() modula na
     foldu 0 izbranega nabora (privzeto OpenML 38 sick, ki ima kategorične
     stolpce IN manjkajoče vrednosti) in pri tem prestreže fit() knjižničnega
     razreda: zapiše parametre, s katerimi je bil model ustvarjen, argumente
     klica fit in kakšen vhod je model dobil (tipi stolpcev, število NaN).
     Parametre primerja s privzetimi (razred, ustvarjen brez argumentov).
     Dovoljena so le odstopanja v ALLOWED (seme, računanje, izpis). Nato isto
     ponovi z neurejenimi kategoričnimi stolpci: openml-python jih označi kot
     ordered=True, napovedi pa od te oznake ne smejo biti odvisne. Pri LightGBM
     še preveri, da je naš izrecni categorical_feature enak privzetemu klicu
     fit(X, y) na neurejenih kategoričnih stolpcih.
  2. Nujnost predobdelave. Na majhnih sintetičnih podatkih preveri, da knjižnica
     brez našega koraka vhoda res ne sprejme, in da tistega, kar zna sama (NaN,
     kategorije), sprejme. Če se nameščena različica knjižnice obnaša drugače od
     pričakovanega, je treba modul popraviti (tako smo 2026-10-14 našli
     XGBoost in RandomForest).

Zagon (iz korena repozitorija; TabPFN/TabICL na CPU delata, a počasi):
    python scripts/audit_defaults.py
    python scripts/audit_defaults.py --dataset 31 --algorithms xgboost random_forest
Izhodna koda 1, če kak parameter odstopa od privzetega izven ALLOWED, če kakšno
učenje odpove ali če se kaka knjižnica obnaša drugače od pričakovanega.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.models import REGISTRY  # noqa: E402

# Parametri, ki smejo odstopati od privzetih, ker ne spremenijo modela.
ALLOWED = {
    "random_state": "seme (ponovljivost), ne hiperparameter",
    "n_jobs": "število jeder (RandomForest); drevesa so enaka kot serijsko",
    "verbosity": "izpis (LightGBM)",
    "verbose": "izpis (CatBoost)",
    "allow_writing_files": "pisanje pomožnih datotek na disk (CatBoost)",
}


def library_classes():
    """Ime algoritma -> knjižnični razred, katerega fit prestrežemo."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from tabicl import TabICLClassifier
    from tabpfn import TabPFNClassifier
    from xgboost import XGBClassifier
    return {"random_forest": RandomForestClassifier, "xgboost": XGBClassifier,
            "lightgbm": LGBMClassifier, "catboost": CatBoostClassifier,
            "tabpfn": TabPFNClassifier, "tabicl": TabICLClassifier}


def describe(X):
    """Kratek opis vhoda, ki ga je dobil model."""
    if isinstance(X, pd.DataFrame):
        kinds = X.dtypes.astype(str).value_counts().to_dict()
        return {"type": "DataFrame", "shape": X.shape, "dtypes": kinds,
                "n_nan": int(X.isna().sum().sum())}
    a = np.asarray(X)
    n_nan = int(np.isnan(a).sum()) if a.dtype.kind == "f" else None
    return {"type": type(X).__name__, "shape": a.shape, "dtypes": {str(a.dtype): a.shape[-1]},
            "n_nan": n_nan}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True  # NaN ni enak NaN, a privzeto missing=nan pomeni isto
    try:
        r = a == b
        return bool(r) if isinstance(r, (bool, np.bool_)) else bool(np.all(r))
    except Exception:
        return repr(a) == repr(b)


def tabpfn_auto_model_path():
    """Uteži, ki bi jih TabPFNClassifier() naložil s privzetim model_path='auto'."""
    from tabpfn.model_loading import resolve_model_path, resolve_model_version
    version = resolve_model_version(None)
    return str(resolve_model_path(None, which="classifier", version=version.value)[0][0])


def intercept(cls, store):
    """Ovije cls.fit: ob prvem klicu zapiše parametre, argumente in vhod."""
    had_own = "fit" in cls.__dict__
    original = cls.fit

    def fit(self, X, y=None, *args, **kwargs):
        if "params" not in store:
            store["params"] = self.get_params(deep=False)
            store["fit_kwargs"] = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                                   for k, v in kwargs.items()}
            store["X"] = describe(X)
        return original(self, X, y, *args, **kwargs)

    cls.fit = fit

    def restore():
        if had_own:
            cls.fit = original
        else:
            del cls.fit
    return restore


def audit_settings(dataset_id, algorithms):
    config = load_config()
    ds = load_dataset(dataset_id, n_splits=config["n_splits"], random_state=config["random_state"],
                      cache_dir=config["cache_dir"], n_repeats=config["n_repeats"])
    tr, te = ds["folds"][0]
    X, y, cats = ds["X"], ds["y"], ds["categorical_cols"]
    print(f"1. NASTAVITVE IN VHOD - {ds['name']} (OpenML {dataset_id}), fold 0: {len(tr)} učnih "
          f"vrstic, {X.shape[1]} atributov ({len(cats)} kategoričnih), "
          f"{int(X.isna().sum().sum())} manjkajočih vrednosti v naboru\n")
    classes = library_classes()
    failures = []
    for name in algorithms:
        cls = classes[name]
        store = {}
        restore = intercept(cls, store)
        try:
            res = REGISTRY[name].run(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], cats,
                                     config["random_state"])
        finally:
            restore()
        default = cls().get_params(deep=False)
        params = store.get("params", {})
        deviations = {k: (v, default.get(k, "<ni nastavljen>")) for k, v in params.items()
                      if k not in default or not same(v, default[k])}
        explained = dict(ALLOWED)
        if name == "tabpfn" and "model_path" in deviations:
            # Različico modela izberemo izrecno (glej tabpfn_model); dovoljeno le,
            # če je to natanko datoteka, ki bi jo izbral privzeti 'auto'.
            auto = tabpfn_auto_model_path()
            if str(deviations["model_path"][0]) == auto:
                explained["model_path"] = ("izrecno izbrane privzete uteži: 'auto' v tem okolju "
                                           f"izbere isto datoteko ({os.path.basename(auto)})")
        bad = {k: v for k, v in deviations.items() if k not in explained}
        print(f"[{name}] {cls.__module__}.{cls.__name__}   ROC-AUC {res['roc_auc']}"
              + (f"   NAPAKA: {res['error']}" if res["error"] else ""))
        print(f"    predobdelava: {res['preprocessing']}")
        for k, (ours, dflt) in sorted(deviations.items()):
            why = explained.get(k, "NI DOVOLJENO")
            print(f"    odstopanje {k} = {ours!r} (privzeto {dflt!r}) -> {why}")
        if not deviations:
            print("    vsi parametri privzeti")
        if store.get("fit_kwargs"):
            print(f"    argumenti fit: {store['fit_kwargs']}")
        print(f"    vhod modela: {store.get('X')}")

        # openml-python vse kategorične stolpce označi kot urejene (ordered=True),
        # čeprav nominalni atributi vrstnega reda nimajo. Napovedi od te oznake
        # ne smejo biti odvisne.
        X_unordered = X.copy()
        for c in cats:
            X_unordered[c] = X_unordered[c].cat.as_unordered()
        res_u = REGISTRY[name].run(X_unordered.iloc[tr], y.iloc[tr], X_unordered.iloc[te],
                                   y.iloc[te], cats, config["random_state"])
        d_ordered = max_abs_diff(res["proba"], res_u["proba"])
        print(f"    oznaka ordered iz OpenML spremeni napovedi: "
              f"{'ne' if d_ordered == 0 else f'DA (do {d_ordered:.1e})'}")
        if d_ordered != 0:
            failures.append(f"{name}: ordered")

        if name == "lightgbm":
            # Privzeti categorical_feature='auto' upošteva le NEurejene kategorične
            # stolpce. Naš izrecni seznam mora dati isto kot privzeti klic fit(X, y)
            # na pravilno označenih (neurejenih) nominalnih stolpcih.
            from lightgbm import LGBMClassifier
            X_tr, X_te = X_unordered.iloc[tr].copy(), X_unordered.iloc[te].copy()
            obj = [c for c in X.columns if X[c].dtype == object]
            X_tr[obj] = X_tr[obj].astype(float)
            X_te[obj] = X_te[obj].astype(float)
            plain = LGBMClassifier(random_state=config["random_state"], verbosity=-1)
            d_plain = max_abs_diff(res["proba"], plain.fit(X_tr, y.iloc[tr]).predict_proba(X_te))
            print(f"    enako privzetemu klicu fit(X, y) na neurejenih kategoričnih stolpcih: "
                  f"{'da' if d_plain == 0 else f'NE (razlika do {d_plain:.1e})'}")
            if d_plain != 0:
                failures.append("lightgbm: plain")
        print()
        if bad or res["error"] or "params" not in store:
            failures.append(name)
    return failures


def max_abs_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).max())


def audit_necessity():
    """Ali knjižnica brez našega koraka vhod zavrne (oz. sprejme, kar zna sama)."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    n = 200
    num = rng.normal(size=n)
    num[rng.random(n) < 0.3] = np.nan
    cat = pd.Series(pd.Categorical(rng.choice(["a", "b", "c"], n)))
    cat[:10] = np.nan
    y = rng.integers(0, 2, n)
    X_cat = pd.DataFrame({"num": num, "cat": cat})
    X_nan = pd.DataFrame({"num": num, "num2": rng.normal(size=n)})
    X_obj = X_nan.astype({"num2": object})

    def cb():
        return CatBoostClassifier(iterations=5, verbose=False, allow_writing_files=False)

    # (opis, pričakovano: True = sprejme, False = zavrne, poskus)
    checks = [
        ("RandomForest sprejme NaN -> ne nadomeščamo", True,
         lambda: RandomForestClassifier(n_estimators=5).fit(X_nan, y)),
        ("RandomForest zavrne kategorije (nize) -> ordinalno kodiramo", False,
         lambda: RandomForestClassifier(n_estimators=5).fit(X_cat, y)),
        ("XGBoost sprejme 'category' z NaN -> ne kodiramo", True,
         lambda: XGBClassifier(n_estimators=5).fit(X_cat, y)),
        ("XGBoost zavrne numerični stolpec tipa object -> float", False,
         lambda: XGBClassifier(n_estimators=5).fit(X_obj, y)),
        ("LightGBM sprejme 'category' z NaN -> ne kodiramo", True,
         lambda: LGBMClassifier(n_estimators=5, verbosity=-1).fit(X_cat, y)),
        ("LightGBM zavrne numerični stolpec tipa object -> float", False,
         lambda: LGBMClassifier(n_estimators=5, verbosity=-1).fit(X_obj, y)),
        ("CatBoost zavrne 'category' brez cat_features -> podamo cat_features", False,
         lambda: cb().fit(X_cat, y)),
        ("CatBoost zavrne NaN v kategoričnem stolpcu -> NaN v niz 'nan'", False,
         lambda: cb().fit(X_cat, y, cat_features=["cat"])),
        ("CatBoost sprejme NaN v numeričnem stolpcu -> ne nadomeščamo", True,
         lambda: cb().fit(X_nan, y)),
    ]
    print("2. NUJNOST PREDOBDELAVE (sintetični podatki)\n")
    failures = []
    for text, expect_ok, attempt in checks:
        try:
            attempt()
            ok, detail = True, ""
        except Exception as e:
            ok, detail = False, f" ({type(e).__name__}: {str(e).splitlines()[0][:90]})"
        verdict = "ustreza" if ok == expect_ok else "NE USTREZA - preveri modul"
        print(f"    {'sprejme' if ok else 'zavrne '}  {text}: {verdict}{detail}")
        if ok != expect_ok:
            failures.append(text)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=int, default=38, help="OpenML ID (privzeto 38 sick)")
    parser.add_argument("--algorithms", nargs="+", default=None,
                        help="privzeto vsi iz config.yaml")
    args = parser.parse_args()
    algorithms = args.algorithms or load_config()["algorithms"]
    failures = audit_settings(args.dataset, algorithms)
    failures += audit_necessity()
    print("\nREZULTAT:", "vse ustreza" if not failures else f"NE USTREZA: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
