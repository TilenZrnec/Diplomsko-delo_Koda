"""Diagnostika: zakaj ima LightGBM s privzetimi hiperparametri na Medic3 ROC-AUC ~0,5.

V zagonih medic3_raw in medic3_160 (2026-10-10/11) je imel LightGBM povprečni
ROC-AUC 0,503 oz. 0,506, ostalih pet algoritmov na istih foldih 0,81-0,96.
Skript pokaže, ali je kriva naša koda ali privzete nastavitve LightGBM, in
zapiše številke, ki jih navaja diploma. Vse na foldu 0 (ponovitev 0, seme 42):

  benchmark    - naša pot: src.data.load_dataset + src.models.lightgbm_model.run.
                 ROC-AUC primerja z vrstico fold 0 v results.csv zagona; učenje
                 ponovi z istimi klici in eval_set (učna izguba po iteracijah) ter
                 prešteje, koliko začetnih iteracij je enakih neodvisni poti.
  independent  - BREZ naše kode: CSV prebere pandas, oznake np.unique, fold 0 da
                 sklearnov RepeatedStratifiedKFold z istimi parametri iz
                 config.yaml, model nativni lgb.train (nastavljeni le objective,
                 num_class in seed), metriko sklearn.roc_auc_score. Trije poskusi
                 na istem foldu: privzete nastavitve, reg_lambda=1 in
                 min_child_weight=1 (privzeti vrednosti XGBoost). Za vsako
                 iteracijo zapiše učno in testno multi_logloss (izračuna ju
                 LightGBM sam) in največjo absolutno vrednost lista med drevesi
                 te iteracije; prešteje opozorila, ki jih LightGBM izpiše.

Zagon (iz korena repozitorija; samo CPU, nič ne potrebuje GPU):
    python scripts/diagnose_lightgbm_medic3.py                        # oba dela, obe različici
    python scripts/diagnose_lightgbm_medic3.py --dataset-set medic3_160 --part independent
    python scripts/diagnose_lightgbm_medic3.py --part independent --settings default
        # zadnje lahko teče tudi v starejšem okolju (npr. LightGBM 4.6.0 v tabularOriginal)

Izhod v results/runs/<zagon>/diagnostics/ (samo metrike, nobenih podatkov),
v imenu datoteke je različica LightGBM, da se zagoni v različnih okoljih ne
prepišejo: lightgbm_<v>_independent.csv, lightgbm_<v>_curve.csv,
lightgbm_<v>_benchmark.csv, lightgbm_<v>_benchmark_curve.csv, lightgbm_<v>_info.json.
Zahteva data/medic3/Medic3.csv. Del benchmark naj teče za delom independent (ali
kot --part all), sicer primerjave krivulj ni.
"""

import os

# Število niti kot na Arnesu (run_medic3.sh: OMP_NUM_THREADS=8), nastavljeno pred
# uvozom knjižnic, da ga OpenMP upošteva.
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse  # noqa: E402
import collections  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {"medic3": "medic3_raw", "medic3_160": "medic3_160"}
SETTINGS = {
    "default": {},
    "reg_lambda=1": {"reg_lambda": 1.0},
    "min_child_weight=1": {"min_child_weight": 1.0},
}
# Privzeti n_estimators v LGBMClassifier in num_boost_round v lgb.train.
N_ROUNDS = 100


class _Tally:
    """Logger za lgb.register_logger: namesto izpisa prešteje opozorila LightGBM."""

    def __init__(self):
        self.warnings = collections.Counter()

    def info(self, msg):
        # Sporočila iz C++ pridejo vsa skozi info(); opozorila imajo oznako [Warning].
        if "[Warning]" in msg:
            self.warnings[msg.split("[Warning]", 1)[1].strip()[:100]] += 1

    def warning(self, msg):
        self.warnings[str(msg).strip()[:100]] += 1


def read_config():
    with open(os.path.join(REPO_ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


def load_independent(set_name, cfg):
    """Medic3 in fold 0 brez naše kode (pandas, numpy, sklearn), po opisu nabora."""
    with open(os.path.join(REPO_ROOT, cfg["dataset_sets"][set_name])) as f:
        spec = json.load(f)[0]
    df = pd.read_csv(os.path.join(REPO_ROOT, spec["path"]), low_memory=False)
    y = df[spec["target"]]
    X = df.drop(columns=[spec["target"]] + list(spec.get("drop_cols", []))).astype(float)
    if spec.get("max_classes"):
        # isto pravilo kot opis nabora: najpogostejši razredi, ob izenačenju po oznaki
        counts = y.value_counts()
        keep = sorted(counts.index, key=lambda c: (-counts[c], str(c)))[: int(spec["max_classes"])]
        mask = y.isin(keep).to_numpy()
        X, y = X[mask], y[mask]
    classes, codes = np.unique(y.to_numpy(), return_inverse=True)
    splitter = RepeatedStratifiedKFold(
        n_splits=cfg["n_splits"], n_repeats=cfg["n_repeats"], random_state=cfg["random_state"]
    )
    train_idx, test_idx = next(splitter.split(np.zeros(len(codes)), codes))
    return X.to_numpy(), codes, len(classes), train_idx, test_idx


def max_abs_leaf(booster):
    """Največja |vrednost lista| med drevesi vsake iteracije (seznam po iteracijah)."""
    out = []
    for i in range(booster.current_iteration()):
        model = booster.dump_model(num_iteration=1, start_iteration=i)
        largest = 0.0
        for tree in model["tree_info"]:
            stack = [tree["tree_structure"]]
            while stack:
                node = stack.pop()
                if "leaf_value" in node:
                    largest = max(largest, abs(node["leaf_value"]))
                else:
                    stack += [node["left_child"], node["right_child"]]
        out.append(largest)
    return out


def run_independent(set_name, cfg, settings):
    X, y, n_classes, tr, te = load_independent(set_name, cfg)
    print(f"[{set_name}] neodvisna pot: {len(tr)} učnih, {len(te)} testnih vrstic, "
          f"{X.shape[1]} atributov, {n_classes} razredov", flush=True)
    rows, curve, warnings = [], [], {}
    for name in settings:
        tally = _Tally()
        lgb.register_logger(tally)
        params = {"objective": "multiclass", "num_class": n_classes,
                  "seed": cfg["random_state"], **SETTINGS[name]}
        dtrain = lgb.Dataset(X[tr], y[tr])
        dtest = lgb.Dataset(X[te], y[te], reference=dtrain)
        ev = {}
        t0 = time.perf_counter()
        booster = lgb.train(params, dtrain, num_boost_round=N_ROUNDS,
                            valid_sets=[dtrain, dtest], valid_names=["train", "test"],
                            callbacks=[lgb.record_evaluation(ev)])
        seconds = time.perf_counter() - t0
        proba = booster.predict(X[te])
        leaves = max_abs_leaf(booster)
        train_ll, test_ll = ev["train"]["multi_logloss"], ev["test"]["multi_logloss"]
        for i in range(len(train_ll)):
            curve.append({"setting": name, "iteration": i + 1, "train_logloss": train_ll[i],
                          "test_logloss": test_ll[i],
                          "max_abs_leaf": leaves[i] if i < len(leaves) else np.nan})
        row = {
            "setting": name,
            "roc_auc": roc_auc_score(y[te], proba, multi_class="ovr", average="macro",
                                     labels=np.arange(n_classes)),
            "accuracy": float((proba.argmax(axis=1) == y[te]).mean()),
            "majority_share_test": float(np.bincount(y[te]).max() / len(te)),
            "mean_max_proba": float(proba.max(axis=1).mean()),
            "train_logloss_first": train_ll[0],
            "train_logloss_min": min(train_ll),
            "train_logloss_last": train_ll[-1],
            "max_abs_leaf_first": leaves[0],
            "max_abs_leaf_max": max(leaves),
            "rounds_evaluated": len(train_ll),
            "iterations_in_model": booster.current_iteration(),
            "seconds": seconds,
        }
        rows.append(row)
        warnings[name] = dict(tally.warnings.most_common())
        print(f"  {name:20s} ROC-AUC {row['roc_auc']:.4f}  točnost {row['accuracy']:.4f}  "
              f"učna izguba {row['train_logloss_first']:.3f} -> {row['train_logloss_last']:.3f}  "
              f"največji |list| {row['max_abs_leaf_max']:.3g}  ({seconds:.0f} s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(curve), warnings


def run_benchmark(set_name, run_dir):
    """Fold 0 po naši poti in primerjava z results.csv zagona ter z neodvisno potjo.

    1. src.models.lightgbm_model.run - natanko modul, ki je tekel v zagonu.
    2. Isti klici kot v modulu (pri Medic3 brez kategoričnih stolpcev je njegova
       predobdelava le astype(float)), le z eval_set na učnem foldu, ki modela ne
       spremeni - da dobimo učno izgubo po iteracijah. Enak ROC-AUC kot v 1.
       pomeni, da je učenje na tem stroju ponovljivo iz zagona v zagon.
    3. Krivuljo iz 2 primerja s privzeto krivuljo neodvisne poti iste različice
       LightGBM (če je že narejena): koliko začetnih iteracij je enakih.
    """
    sys.path.insert(0, REPO_ROOT)
    from src.config import dataset_specs, load_config
    from src.data import load_dataset
    from src.models import lightgbm_model
    from src.utils import compute_roc_auc

    config = load_config()
    ds = load_dataset(dataset_specs(config, set_name)[0], n_splits=config["n_splits"],
                      random_state=config["random_state"], cache_dir=config["cache_dir"],
                      n_repeats=config["n_repeats"])
    assert not ds["categorical_cols"], "Medic3 nima kategoričnih stolpcev"
    tr, te = ds["folds"][0]
    X, y = ds["X"], ds["y"]
    t0 = time.perf_counter()
    res = lightgbm_model.run(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te],
                             ds["categorical_cols"], config["random_state"])
    seconds = time.perf_counter() - t0

    X_tr, X_te = X.iloc[tr].astype(float), X.iloc[te].astype(float)
    clf = lgb.LGBMClassifier(random_state=config["random_state"], verbosity=-1)
    clf.fit(X_tr, y.iloc[tr], categorical_feature="auto",
            eval_set=[(X_tr, y.iloc[tr])], eval_names=["train"])
    replay_auc = compute_roc_auc(y.iloc[te], clf.predict_proba(X_te), clf.classes_)
    ours = np.asarray(clf.evals_result_["train"]["multi_logloss"])

    curve = pd.DataFrame({"iteration": np.arange(1, len(ours) + 1), "train_logloss_ours": ours})
    identical = diff_first = np.nan
    indep_path = os.path.join(run_dir, "diagnostics", f"lightgbm_{lgb.__version__}_curve.csv")
    if os.path.exists(indep_path):
        ind = pd.read_csv(indep_path).query("setting == 'default'").train_logloss.to_numpy()
        n = min(len(ind), len(ours))
        curve["train_logloss_independent"] = pd.Series(ind[:n])
        diff = np.abs(ours[:n] - ind[:n])
        bad = np.nonzero(diff > 1e-12)[0]
        identical = int(bad[0]) if len(bad) else n  # število enakih začetnih iteracij
        diff_first = float(diff[0])

    stored = np.nan
    results_csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        match = df[(df.algorithm == "lightgbm") & (df.fold == 0)]
        if len(match):
            stored = float(match.roc_auc.iloc[0])
    row = {"fold": 0, "roc_auc_module": res["roc_auc"], "roc_auc_replay": replay_auc,
           "roc_auc_results_csv": stored, "error": res["error"],
           "identical_iterations_vs_independent": identical,
           "abs_diff_first_iteration": diff_first, "seconds_module": seconds}
    print(f"[{set_name}] naša pot, fold 0: ROC-AUC modul {res['roc_auc']:.6f}, ponovitev "
          f"{replay_auc:.6f}, results.csv {stored:.6f}; enakih začetnih iteracij z neodvisno "
          f"potjo: {identical}", flush=True)
    return pd.DataFrame([row]), curve


def library_defaults():
    """Privzete vrednosti regularizacije listov, prebrane iz nameščenih knjižnic."""
    out = {}
    params = lgb.LGBMClassifier().get_params()
    out["lightgbm"] = {"version": lgb.__version__, "reg_lambda": params["reg_lambda"],
                       "min_child_weight": params["min_child_weight"]}
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 4)), rng.integers(0, 3, 200)
    try:
        import xgboost
        cfg = json.loads(xgboost.XGBClassifier().fit(X, y).get_booster().save_config())
        tp = cfg["learner"]["gradient_booster"]["tree_train_param"]
        out["xgboost"] = {"version": xgboost.__version__, "lambda": float(tp["lambda"]),
                          "min_child_weight": float(tp["min_child_weight"])}
    except Exception as e:  # knjižnice v okolju morda ni
        out["xgboost"] = {"error": str(e)}
    try:
        import catboost
        m = catboost.CatBoostClassifier(iterations=5, verbose=False, allow_writing_files=False)
        out["catboost"] = {"version": catboost.__version__,
                           "l2_leaf_reg": m.fit(X, y).get_all_params()["l2_leaf_reg"]}
    except Exception as e:
        out["catboost"] = {"error": str(e)}
    return out


def provenance():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True).stdout.strip()
        except OSError:
            return None
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    # "dirty" gleda le kodo in konfiguracijo; izhodi diagnostike v results/ (tudi
    # tisti, ki jih je pravkar zapisal zagon v drugem okolju) niso del tega.
    return {"git_commit": git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", "src", "scripts", "config.yaml")),
            "script_sha256": script_sha, "hostname": platform.node(),
            "python": platform.python_version(), "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-set", choices=sorted(RUNS), action="append",
                        help="privzeto obe različici")
    parser.add_argument("--part", choices=["all", "independent", "benchmark"], default="all")
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    args = parser.parse_args()

    cfg = read_config()
    info = {"provenance": provenance(), "defaults": library_defaults(),
            "settings": {k: SETTINGS[k] for k in args.settings}, "n_rounds": N_ROUNDS}
    prefix = f"lightgbm_{lgb.__version__}"
    for set_name in args.dataset_set or list(RUNS):
        run_dir = os.path.join(REPO_ROOT, "results", "runs", RUNS[set_name])
        out_dir = os.path.join(run_dir, "diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        if args.part in ("all", "independent"):
            summary, curve, warnings = run_independent(set_name, cfg, args.settings)
            summary.to_csv(os.path.join(out_dir, f"{prefix}_independent.csv"), index=False)
            curve.to_csv(os.path.join(out_dir, f"{prefix}_curve.csv"), index=False)
            with open(os.path.join(out_dir, f"{prefix}_info.json"), "w") as f:
                json.dump({**info, "dataset_set": set_name, "warnings": warnings}, f, indent=2)
        if args.part in ("all", "benchmark"):
            row, curve = run_benchmark(set_name, run_dir)
            row.to_csv(os.path.join(out_dir, f"{prefix}_benchmark.csv"), index=False)
            curve.to_csv(os.path.join(out_dir, f"{prefix}_benchmark_curve.csv"), index=False)
        print(f"[{set_name}] zapisano v {os.path.relpath(out_dir, REPO_ROOT)}/", flush=True)


if __name__ == "__main__":
    main()
