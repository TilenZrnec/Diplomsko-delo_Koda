"""Pomožne funkcije, skupne vsem modelom."""

import os

import numpy as np
from sklearn.metrics import roc_auc_score


def scored_class_indices(y_true, classes):
    """Indeksi stolpcev proba, po katerih se sme povprečiti več-razredni ROC-AUC.

    Stolpec v predict_proba obstaja samo za razrede, ki jih je model videl v
    UČNI množici (`classes` = model.classes_, v istem vrstnem redu kot stolpci).
    ROC-AUC posameznega razreda pa je definiran samo, če je razred prisoten v
    TESTNI množici (`y_true`) - sicer v one-vs-rest podnalogi ni nobenega
    pozitivnega primera.

    Pri naborih, kjer ima vsak razred vsaj n_splits primerov (vsi CC18 nabori in
    Medic3, katerega najmanjši razred ima 50 vrstic), sta seznama enaka in ta
    funkcija vrne vse stolpce. Razlikujeta se le pri naboru z razredom, manjšim
    od n_splits: stratificirana delitev ga ne more razporediti v vsak fold.
    Funkcija je zato varovalka za prihodnje nabore, ne popravek za Medic3.
    """
    present = set(np.unique(np.asarray(y_true)).tolist())
    return [i for i, c in enumerate(np.asarray(classes).tolist()) if c in present]


def compute_roc_auc(y_true, proba, classes=None):
    """Izračuna ROC-AUC iz predict_proba izhoda; podpira binarno in več-razredno.

    Binarno: površina pod ROC krivuljo za verjetnost razreda 1.
    Več-razredno: one-vs-rest za vsak razred posebej, nato neuteženo povprečje
    (macro), da ima vsak razred enako težo ne glede na pogostost.

    `classes` so oznake razredov, ki pripadajo stolpcem `proba` (model.classes_).
    Povpreči se SAMO po razredih, za katere je metrika definirana (glej
    scored_class_indices); koliko jih je bilo, zapiše runner v stolpec
    n_classes_scored, da je v rezultatih vidno, če se folda razlikujeta.
    Kadar so prisotni vsi razredi, je izid bitno identičen klicu
    roc_auc_score(..., multi_class="ovr", average="macro") - preverjeno.

    Vrne None, če ostaneta manj kot dva razreda in povprečje ne pove ničesar.
    """
    y_true = np.asarray(y_true)
    classes = np.arange(proba.shape[1]) if classes is None else np.asarray(classes)

    if proba.shape[1] == 2:
        return roc_auc_score(y_true, proba[:, 1])

    indices = scored_class_indices(y_true, classes)
    if len(indices) < 2:
        return None
    scores = [
        roc_auc_score((y_true == classes[i]).astype(int), proba[:, i]) for i in indices
    ]
    return float(np.mean(scores))


def cpu_threads():
    """Koliko niti smejo uporabiti CPU algoritmi na tem stroju/opravilu.

    Na Arnesu batch skripta nastavi OMP_NUM_THREADS na število dodeljenih jeder;
    lokalno velja število jeder stroja. Vrednost gre v stolpec 'device', da je
    ob primerjavi časov jasno, s koliko jedri je algoritem tekel.
    """
    for var in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(var)
        if value and value.isdigit():
            return int(value)
    return os.cpu_count() or 1


def describe_device(uses_gpu):
    """Kratek opis naprave, na kateri je algoritem tekel, za stolpec 'device'.

    Ansambli tečejo na CPU ('cpu x8' = 8 niti), temeljna modela na GPU, če je
    na voljo ('cuda: NVIDIA H100 ...'), sicer na CPU. Časi med napravama niso
    primerljivi, zato je naprava zapisana ob vsakem učenju.
    """
    if uses_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda: {torch.cuda.get_device_name(0)}"
        except ImportError:
            pass
    return f"cpu x{cpu_threads()}"
