"""Metrike uspešnosti, ki jih računamo iz shranjenih napovedi (ne med učenjem).

Glavna metrika benchmarka je ROC-AUC (src/utils.compute_roc_auc), ki jo vsak
modul izračuna ob učenju in je v results.csv. Točnost in logaritemsko izgubo
izračuna šele scripts/compute_metrics.py iz datotek predictions/*.npz, zato za
njiju ni treba ponoviti nobenega učenja. Ta modul namenoma ni del učne poti
(modeli in runner ga ne uvozijo), da se koda, ki uči, z dodajanjem metrik ne
spremeni.

Vse tri metrike sprejmejo isti vhod kot compute_roc_auc: y_true (kode razredov
testnih primerov), proba (predict_proba) in classes (koda razreda vsakega
stolpca proba, tj. model.classes_; v .npz shranjeno kot proba_classes).
"""

import numpy as np
from sklearn.metrics import log_loss

# Metrika -> ali je višja vrednost boljša. Na tem slovarju temeljijo rangi v
# src/summary.py in src/stats.py (logaritemska izguba je edina, pri kateri je
# boljša nižja vrednost).
HIGHER_IS_BETTER = {"roc_auc": True, "accuracy": True, "log_loss": False}


def compute_accuracy(y_true, proba, classes=None):
    """Točnost: delež testnih primerov, pri katerih ima pravi razred največjo verjetnost.

    Napoved je razred z največjo verjetnostjo v vrstici (pri izenačenju prvi
    stolpec), kar je tudi pravilo, po katerem predict() vrne razred pri vseh
    šestih algoritmih.
    """
    classes = np.arange(proba.shape[1]) if classes is None else np.asarray(classes)
    predicted = classes[np.argmax(proba, axis=1)]
    return float(np.mean(predicted == np.asarray(y_true)))


def full_proba(proba, classes, n_classes):
    """Proba z enim stolpcem za vsak razred nabora (0..n_classes-1), v float64.

    Razred, ki ga v učnem foldu ni bilo, v proba nima stolpca (glej
    src/utils.scored_class_indices); tu dobi verjetnost 0. Vsaka vrstica je
    normirana na vsoto 1: shranjene napovedi so float32 in se zaradi zaokrožanja
    seštejejo v 1 le na ~1e-7, kar bi sklearn.metrics.log_loss opozoril.
    Normiranje odpravi le to napako (koliko je bila, vrne row_sum_error).
    """
    full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    full[:, np.asarray(classes).astype(int)] = proba
    sums = full.sum(axis=1, keepdims=True)
    return full / sums, float(np.max(np.abs(sums - 1.0)))


def compute_log_loss(y_true, proba, classes=None, n_classes=None):
    """Logaritemska izguba: povprečje -log verjetnosti, ki jo model da pravemu razredu.

    Računa jo sklearn.metrics.log_loss, ki verjetnosti omeji na [eps, 1 - eps],
    kjer je eps strojna natančnost tipa; pri float64 je eps = 2.2e-16, zato en
    primer prispeva največ -log(eps) = 36.04. Vrne (izguba, row_sum_error).
    `n_classes` je število vseh razredov nabora; privzeto največja oznaka + 1.
    """
    y_true = np.asarray(y_true)
    classes = np.arange(proba.shape[1]) if classes is None else np.asarray(classes)
    if n_classes is None:
        n_classes = int(max(classes.max(), y_true.max())) + 1
    full, row_sum_error = full_proba(proba, classes, n_classes)
    return float(log_loss(y_true, full, labels=np.arange(n_classes))), row_sum_error
