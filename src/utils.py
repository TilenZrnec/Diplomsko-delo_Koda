"""Pomožne funkcije, skupne vsem modelom."""

import os

from sklearn.metrics import roc_auc_score


def compute_roc_auc(y_true, proba):
    """Izračuna ROC-AUC iz predict_proba izhoda; podpira binarno in več-razredno.

    Binarno: površina pod ROC krivuljo za verjetnost razreda 1.
    Več-razredno: one-vs-rest za vsak razred posebej, nato neuteženo povprečje
    (macro), da ima vsak razred enako težo ne glede na pogostost.
    """
    n_classes = proba.shape[1]
    if n_classes == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")


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
