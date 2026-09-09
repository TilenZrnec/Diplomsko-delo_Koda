"""Branje config.yaml - edinega vira parametrov eksperimenta.

Vsak skript, ki potrebuje seme, število foldov, seznam algoritmov ali poti,
jih dobi od tu. Nič od tega ni zapisano trdo nikjer drugje v kodi; s tem je
zagotovljeno, da lokalni pilot in SLURM polje na Arnesu tečeta z istimi
parametri, ker bereta isto datoteko.

Uporaba:

    from src.config import load_config, dataset_specs
    config = load_config()                      # slovar iz config.yaml
    specs = dataset_specs(config, "cc18")       # seznam naborov iz dataset_sets

Relativne poti v config.yaml (cache_dir, results_dir, dataset_sets) so mišljene
relativno na koren repozitorija; load_config() jih pretvori v absolutne, zato
skripte delujejo ne glede na to, iz katere mape so pognane.
"""

import json
import os

import yaml

# Koren repozitorija = mapa nad src/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")

# Ključi, ki morajo biti v config.yaml, in njihova privzeta vrednost, če manjkajo.
# Privzete vrednosti so tu samo zato, da starejši config.yaml ne podre zagona;
# v repozitoriju so vsi ključi zapisani eksplicitno.
DEFAULTS = {
    "random_state": 42,
    "n_splits": 5,
    "n_repeats": 1,
    "cache_dir": "data/openml_cache",
    "results_dir": "results/runs",
    "save_predictions": True,
    "saturation_threshold": 0.995,
    "alpha": 0.05,
}


def _absolute(path):
    """Relativno pot razreši glede na koren repozitorija; absolutno pusti pri miru."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def load_config(path=None):
    """Prebere config.yaml in vrne slovar z absolutnimi potmi.

    Zahteva ključa 'algorithms' in 'dataset_sets'; za ostale uporabi DEFAULTS,
    če manjkajo. Tako je napaka v konfiguraciji vidna takoj, ne šele sredi
    dolgega zagona na gruči.
    """
    path = path or CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for key, value in DEFAULTS.items():
        config.setdefault(key, value)

    for key in ("algorithms", "dataset_sets"):
        if key not in config or not config[key]:
            raise KeyError(f"config.yaml: manjka obvezen ključ '{key}'")

    config["cache_dir"] = _absolute(config["cache_dir"])
    config["results_dir"] = _absolute(config["results_dir"])
    config["dataset_sets"] = {name: _absolute(p) for name, p in config["dataset_sets"].items()}
    config["config_path"] = os.path.abspath(path)
    return config


def dataset_specs(config, set_name):
    """Vrne seznam opisov naborov za poimenovani nabor iz dataset_sets.

    Vsak element je bodisi celo število (OpenML ID) bodisi slovar z opisom
    lokalne datoteke - oboje zna sprejeti src.data.load_dataset().
    """
    try:
        ids_file = config["dataset_sets"][set_name]
    except KeyError:
        known = ", ".join(sorted(config["dataset_sets"]))
        raise KeyError(f"Neznan nabor '{set_name}'; v config.yaml so: {known}") from None
    with open(ids_file, encoding="utf-8") as f:
        specs = json.load(f)
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"{ids_file}: pričakovan neprazen JSON seznam")
    return specs


def spec_id(spec):
    """Kratek, za ime datoteke varen identifikator nabora ('31' ali 'medic3')."""
    if isinstance(spec, int):
        return str(spec)
    if isinstance(spec, dict):
        if spec.get("source", "openml") == "openml":
            return str(int(spec["id"]))
        name = spec.get("name") or os.path.splitext(os.path.basename(spec["path"]))[0]
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    raise TypeError(f"Nepodprt opis nabora: {spec!r}")
