"""Iz manifest.json enega zagona naredi LaTeX tabelo različic knjižnic za diplomo.

Zagon: python scripts/gen_version_table.py <run_id | mapa zagona> [--output POT]

Privzeto zapiše <mapa zagona>/summary/razlicice.tex. Tabela ima iste stolpce
kot tabela \\ref{tab:razlicice} v diplomi (algoritem, knjižnica, razred,
različica) in dodatno vrstice za Python, torch in CUDA - da so različice v
besedilu vedno prepisane iz zagona, ki je dal rezultate, ne na roko.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.merge_results import resolve_run_dir  # noqa: E402

# (algoritem, paket na PyPI, razred) v vrstnem redu, kot nastopajo v diplomi.
ROWS = [
    ("Naključni gozd", "scikit-learn", "RandomForestClassifier"),
    ("XGBoost", "xgboost", "XGBClassifier"),
    ("LightGBM", "lightgbm", "LGBMClassifier"),
    ("CatBoost", "catboost", "CatBoostClassifier"),
    ("TabPFN", "tabpfn", "TabPFNClassifier"),
    ("TabICL", "tabicl", "TabICLClassifier"),
]


def versions_from_manifest(manifest):
    """Slovar paket -> različica iz seznama pip_freeze (imena brez občutljivosti na velikost)."""
    out = {}
    for line in manifest.get("pip_freeze", []):
        if "==" in line:
            name, version = line.split("==", 1)
            out[name.lower()] = version
    return out


def build_table(manifest):
    versions = versions_from_manifest(manifest)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Uporabljeni algoritmi, implementacije in pripete različice knjižnic.}",
        "\\label{tab:razlicice}",
        "\\begin{tabular}{llll}",
        "\\hline",
        "Algoritem & Knjižnica & Razred & Različica \\\\",
        "\\hline",
    ]
    for algo, package, cls in ROWS:
        version = versions.get(package.lower(), "?")
        lines.append(f"{algo} & {package} & \\texttt{{{cls.replace('_', '_')}}} & {version} \\\\")
    lines += ["\\hline"]
    lines.append(f"Python & -- & -- & {manifest.get('python', '?')} \\\\")
    torch_v = manifest.get("torch") or versions.get("torch", "?")
    lines.append(f"PyTorch & torch & -- & {torch_v} \\\\")
    if manifest.get("cuda_version"):
        lines.append(f"CUDA & -- & -- & {manifest['cuda_version']} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    header = (
        f"% Samodejno iz {manifest.get('run_id', '?')}/manifest.json "
        f"(git {manifest.get('git_commit', '?')}, {manifest.get('created', '?')}). Ne urejaj na roko.\n"
    )
    return header + "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="LaTeX tabela različic iz manifest.json.")
    parser.add_argument("run", help="run_id ali mapa zagona")
    parser.add_argument("--output", default=None, help="Izhodna .tex (privzeto <mapa>/summary/razlicice.tex)")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    output = args.output or os.path.join(run_dir, "summary", "razlicice.tex")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    table = build_table(manifest)
    with open(output, "w", encoding="utf-8") as f:
        f.write(table)
    print(table)
    print(f"Zapisano v {os.path.relpath(output, REPO_ROOT)}")


if __name__ == "__main__":
    main()
