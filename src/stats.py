"""Statistično preverjanje razlik med algoritmi čez več naborov (Demšar 2006).

Zagon: python -m src.stats <run_id | mapa zagona | results.csv> [--no-saturated]

Vprašanje, na katero odgovarja: ali so razlike v rangih med šestimi algoritmi
čez N naborov večje, kot bi jih pričakovali po naključju? Postopek:

1. Friedmanov test (neparametrični ANOVA nad rangi): ničelna domneva je, da so
   vsi algoritmi enakovredni. Če ga zavrnemo, sledi post-hoc.
2. Nemenyijev post-hoc: dva algoritma se značilno razlikujeta, če se njuna
   povprečna ranga razlikujeta za vsaj kritično razdaljo
   CD = q_alpha * sqrt(k(k+1) / (6N)). Izriše se diagram kritične razdalje.
3. Parni Wilcoxonov test predznačenih rangov na povprečnem ROC-AUC po naborih,
   s Holmovim popravkom za večkratno testiranje - občutljivejši od Nemenyija
   (Benavoli, Corani in Mangili 2016 priporočajo prav to kombinacijo).

Vse teče na ravni NABORA (en podatek na nabor = povprečje čez folde), ne folda.
Neuspeh dobi najslabši rang (isto pravilo kot v src/summary.py); Wilcoxon
primerja le nabore, kjer sta oba algoritma uspela.
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import load_config  # noqa: E402
from src.summary import (  # noqa: E402
    dataset_pivot, load_results, rank_matrix, resolve_results, saturated_mask, to_latex,
)

# Demšar (2006), tabela 5: kritične vrednosti q_0.05 za Nemenyijev test, k = 2..10.
# Uporabijo se le, če scipy studentized_range ni na voljo.
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def nemenyi_q(k, alpha):
    """Kritična vrednost q_alpha = studentizirani razpon / sqrt(2)."""
    try:
        return stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    except Exception:  # noqa: BLE001 - stara scipy brez studentized_range
        if alpha == 0.05 and k in _Q05:
            return _Q05[k]
        raise


def critical_difference(k, n, alpha):
    return nemenyi_q(k, alpha) * np.sqrt(k * (k + 1) / (6.0 * n))


def friedman(ranks):
    """Friedmanov test nad matriko rangov (nabor x algoritem)."""
    stat, p = stats.friedmanchisquare(*[ranks[c].values for c in ranks.columns])
    return float(stat), float(p)


def wilcoxon_holm(pivot, alpha):
    """Parni Wilcoxonovi testi na povprečnem ROC-AUC po naborih s Holmovim popravkom.

    Vrne DataFrame s stolpci: a, b, n (skupni nabori), mean_diff (a - b),
    wins_a, wins_b, p, p_holm, significant.
    """
    algos = list(pivot.columns)
    rows = []
    for a, b in itertools.combinations(algos, 2):
        both = pivot[[a, b]].dropna()
        diff = both[a] - both[b]
        if len(both) < 5 or np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(both[a], both[b], zero_method="zsplit").pvalue)
        rows.append({
            "a": a, "b": b, "n": int(len(both)),
            "mean_diff": float(diff.mean()),
            "wins_a": int((diff > 0).sum()), "wins_b": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()), "p": p,
        })
    out = pd.DataFrame(rows)
    # Holm: p-vrednosti uredimo naraščajoče, i-to pomnožimo z (m - i + 1) in
    # vzamemo kumulativni maksimum, da popravljene ostanejo monotone.
    m = len(out)
    order = np.argsort(out["p"].values)
    adjusted = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, out["p"].values[idx] * (m - i))
        adjusted[idx] = min(1.0, running)
    out["p_holm"] = adjusted
    out["significant"] = out["p_holm"] < alpha
    return out


def cd_diagram(mean_ranks, cd, path, title=None):
    """Diagram kritične razdalje (Demšar 2006): os rangov, algoritmi, črte za klike."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = mean_ranks.sort_values()
    names = list(ranks.index)
    k = len(names)
    lo, hi = 1.0, float(k)

    half = (k + 1) // 2
    fig_h = 1.4 + 0.45 * half
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    ax.set_xlim(lo - 0.2, hi + 0.2)
    ax.set_ylim(-0.6 * half - 0.5, 1.5)
    ax.axis("off")

    # Os rangov.
    ax.plot([lo, hi], [0, 0], color="black", lw=1.2)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.15], color="black", lw=1)
        ax.text(r, 0.3, str(r), ha="center", va="bottom", fontsize=9)

    # Kritična razdalja nad osjo.
    ax.plot([lo, lo + cd], [1.0, 1.0], color="black", lw=2)
    ax.plot([lo, lo], [0.9, 1.1], color="black", lw=1)
    ax.plot([lo + cd, lo + cd], [0.9, 1.1], color="black", lw=1)
    ax.text(lo + cd / 2, 1.15, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)

    # Algoritmi: levi (boljši) na levo, desni na desno, izmenično po višini.
    for i, name in enumerate(names):
        r = ranks[name]
        if i < half:
            y, x_txt, ha = -0.6 * (i + 1), lo - 0.15, "right"
        else:
            y, x_txt, ha = -0.6 * (k - i), hi + 0.15, "left"
        ax.plot([r, r], [0, y], color="black", lw=0.8)
        ax.plot([r, x_txt], [y, y], color="black", lw=0.8)
        ax.text(x_txt, y, f"{name} ({r:.2f})", ha=ha, va="center", fontsize=9)

    # Klike: največje skupine zaporednih algoritmov, katerih razpon rangov < CD.
    vals = ranks.values
    cliques = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(c[0] <= i and j <= c[1] for c in cliques):
            cliques.append((i, j))
        i += 1
    for level, (a, b) in enumerate(cliques):
        y = -0.25 - 0.18 * level
        ax.plot([vals[a] - 0.05, vals[b] + 0.05], [y, y], color="black", lw=3, solid_capstyle="butt")

    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return cliques


def analyse(path_or_id, config=None, drop_saturated=False, write=True):
    config = config or load_config()
    alpha = config["alpha"]
    results_csv, base_dir = resolve_results(path_or_id, config)
    print(f"Vir: {os.path.relpath(results_csv, REPO_ROOT)}\n")

    df = load_results(results_csv)
    pivot = dataset_pivot(df)
    suffix = ""
    if drop_saturated:
        sat = saturated_mask(pivot, config["saturation_threshold"])
        pivot = pivot[~sat]
        suffix = "_nonsaturated"
        print(f"Brez nasičenih naborov: {int(sat.sum())} izločenih, {len(pivot)} ostane.\n")

    ranks = rank_matrix(pivot)
    k, n = ranks.shape[1], ranks.shape[0]
    mean_ranks = ranks.mean().sort_values()

    stat, p = friedman(ranks)
    cd = critical_difference(k, n, alpha)
    print(f"Friedmanov test: k = {k} algoritmov, N = {n} naborov, chi2_F = {stat:.3f}, p = {p:.2e}")
    print(f"{'Zavrnemo' if p < alpha else 'NE zavrnemo'} ničelno domnevo enakovrednosti (alpha = {alpha}).")
    print(f"Nemenyi: kritična razdalja CD = {cd:.3f}\n")
    print("Povprečni rangi:")
    for name, r in mean_ranks.items():
        print(f"  {name:15s} {r:.3f}")

    nem = []
    for a, b in itertools.combinations(mean_ranks.index, 2):
        d = abs(mean_ranks[a] - mean_ranks[b])
        nem.append({"a": a, "b": b, "rank_diff": d, "significant": d >= cd})
    nemenyi = pd.DataFrame(nem)

    wil = wilcoxon_holm(pivot, alpha)
    print("\nWilcoxon (povprečni ROC-AUC po naborih) + Holm:")
    for _, r in wil.sort_values("p_holm").iterrows():
        flag = "*" if r["significant"] else " "
        print(f" {flag} {r['a']:14s} vs {r['b']:14s} n={r['n']:3d} "
              f"zmage {r['wins_a']:2d}:{r['wins_b']:2d} diff={r['mean_diff']:+.4f} "
              f"p={r['p']:.2e} p_holm={r['p_holm']:.2e}")

    if write:
        out_dir = os.path.join(base_dir, "summary")
        os.makedirs(out_dir, exist_ok=True)
        cliques = cd_diagram(
            mean_ranks, cd, os.path.join(out_dir, f"cd_diagram{suffix}.png"),
            title=f"Nemenyi, N = {n} naborov, alpha = {alpha}",
        )
        friedman_df = pd.DataFrame([{
            "k": k, "N": n, "chi2_F": stat, "p": p, "alpha": alpha, "CD": cd,
            "reject_H0": p < alpha,
        }])
        friedman_df.to_csv(os.path.join(out_dir, f"friedman{suffix}.csv"), index=False)
        nemenyi.to_csv(os.path.join(out_dir, f"nemenyi{suffix}.csv"), index=False)
        wil.to_csv(os.path.join(out_dir, f"wilcoxon_holm{suffix}.csv"), index=False)
        with open(os.path.join(out_dir, f"wilcoxon_holm{suffix}.tex"), "w", encoding="utf-8") as f:
            f.write(to_latex(
                wil[["a", "b", "n", "mean_diff", "wins_a", "wins_b", "p_holm", "significant"]],
                "Parni Wilcoxonovi testi s Holmovim popravkom.", f"tab:wilcoxon{suffix.replace('_', '-')}",
                ".4f",
            ))
        print(f"\nZapisano v {os.path.relpath(out_dir, REPO_ROOT)}/: cd_diagram{suffix}.png/.pdf, "
              f"friedman{suffix}.csv, nemenyi{suffix}.csv, wilcoxon_holm{suffix}.csv/.tex")
        if cliques:
            groups = [", ".join(mean_ranks.index[a:b + 1]) for a, b in cliques]
            print("Skupine brez značilne razlike (Nemenyi): " + " | ".join(groups))

    return {"friedman": (stat, p), "cd": cd, "mean_ranks": mean_ranks, "nemenyi": nemenyi, "wilcoxon": wil}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistični testi razlik med algoritmi.")
    parser.add_argument("results", help="run_id, mapa zagona ali pot do results.csv")
    parser.add_argument("--no-saturated", action="store_true", help="Izloči nasičene nabore")
    parser.add_argument("--no-write", action="store_true", help="Samo izpis, brez zapisa summary/")
    args = parser.parse_args()
    analyse(args.results, drop_saturated=args.no_saturated, write=not args.no_write)
