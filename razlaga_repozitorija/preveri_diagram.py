"""Preveri, da je razlaga_repozitorija/zaporedje.puml usklajen z izvorno kodo.

Diagram zaporedja hitro zastara: preimenuješ datoteko, dodaš algoritem ali
zbrišeš skripto - diagram pa še vedno kaže staro stanje in te zavaja. Ta
skripta to ujame in je namenjena zagonu iz kljuke (hook) pred 'git push'.

Preverja v obe smeri:
  1. MANJKA V DIAGRAMU - vsaka izvorna datoteka (src/**/*.py, scripts/**/*.py,
     scripts/**/*.sh, config.yaml) mora biti v .puml omenjena s POLNO potjo.
     Ujame novo datoteko, ki je nihče ni vrisal.
  2. NE OBSTAJA VEČ - vsaka pot oblike src/... ali scripts/... z ".py"/".sh",
     ki jo diagram omenja, mora na disku še obstajati.
     Ujame preimenovanje in brisanje.

Namenoma uporablja samo standardno knjižnico, da deluje s katerimkoli
sistemskim pythonom - kljuka ne sme biti odvisna od okolja 'tabular'.

Zagon:

    python3 razlaga_repozitorija/preveri_diagram.py
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUML_PATH = os.path.join(REPO_ROOT, "razlaga_repozitorija", "zaporedje.puml")

# Mape z izvorno kodo, ki jih diagram mora pokrivati, in pripone v njih.
SOURCE_DIRS = {"src": (".py",), "scripts": (".py", ".sh")}
# Posamezne datoteke iz korena, ki so prav tako del poteka.
EXTRA_FILES = ("config.yaml",)

# Poti, kot jih diagram zapiše: src/... ali scripts/... s pripono .py/.sh.
# Oglata oklepaja v "src/models/<algoritem>.py" (nadomestna oznaka za šest
# modelov) namenoma NISTA v razredu znakov, zato se ta zapis ne ujame.
PATH_RE = re.compile(r"(?:src|scripts)/[A-Za-z0-9_./-]+\.(?:py|sh)")


def source_files():
    """Vrne urejen seznam poti (relativno na koren repozitorija), ki jih diagram mora omenjati."""
    found = []
    for directory, suffixes in SOURCE_DIRS.items():
        for dirpath, dirnames, filenames in os.walk(os.path.join(REPO_ROOT, directory)):
            # Prevedeni predpomnilnik pythona ni izvorna koda.
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for filename in filenames:
                if filename.endswith(suffixes):
                    abs_path = os.path.join(dirpath, filename)
                    found.append(os.path.relpath(abs_path, REPO_ROOT))
    for filename in EXTRA_FILES:
        if os.path.exists(os.path.join(REPO_ROOT, filename)):
            found.append(filename)
    return sorted(found)


def datotek(n):
    """Slovenska dvojina/množina: 1 datoteka, 2 datoteki, 3-4 datoteke, 5+ datotek."""
    ostanek = n % 100
    if ostanek == 1:
        return "1 datoteka"
    if ostanek == 2:
        return "2 datoteki"
    if ostanek in (3, 4):
        return f"{n} datoteke"
    return f"{n} datotek"


def main():
    if not os.path.exists(PUML_PATH):
        print(f"NAPAKA: diagram ne obstaja: {os.path.relpath(PUML_PATH, REPO_ROOT)}", file=sys.stderr)
        return 1

    with open(PUML_PATH, encoding="utf-8") as f:
        diagram = f.read()

    # PlantUML bere "__besedilo__" kot podčrtano, zato bi se "__init__.py"
    # izrisal kot "init.py". V diagramu je zato zapisan kot "~__init~__.py"
    # (~ je znak za ubežanje). Za primerjavo tilde odstranimo, da se pot spet
    # ujema z resnično potjo na disku.
    diagram = diagram.replace("~", "")

    on_disk = source_files()

    # 1. Katere izvorne datoteke diagram zamolči?
    missing = [path for path in on_disk if path not in diagram]

    # 2. Katere poti diagram omenja, a jih na disku ni več?
    mentioned = sorted(set(PATH_RE.findall(diagram)))
    stale = [path for path in mentioned if not os.path.exists(os.path.join(REPO_ROOT, path))]

    if not missing and not stale:
        print(f"Diagram zaporedja je usklajen ({datotek(len(on_disk))} izvorne kode).")
        return 0

    rel_puml = os.path.relpath(PUML_PATH, REPO_ROOT)
    print(f"NAPAKA: {rel_puml} ni usklajen z izvorno kodo.\n", file=sys.stderr)
    if missing:
        print(f"Manjka v diagramu ({datotek(len(missing))}) - obstaja v kodi, a ni vrisano:",
              file=sys.stderr)
        for path in missing:
            print(f"  + {path}", file=sys.stderr)
        print(file=sys.stderr)
    if stale:
        print(f"Ni več v kodi ({datotek(len(stale))}) - diagram omenja pot, ki je "
              "preimenovana ali zbrisana:", file=sys.stderr)
        for path in stale:
            print(f"  - {path}", file=sys.stderr)
        print(file=sys.stderr)
    print(f"Popravi {rel_puml}, dodaj spremembo v commit in poskusi znova.", file=sys.stderr)
    print("Za enkraten obvod (npr. delna veja): git push --no-verify", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
