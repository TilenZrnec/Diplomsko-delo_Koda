# Rezultati

Vsak zagon benchmarka ima svojo mapo, ločeno po tem, kje je tekel. Vse CSV
datoteke imajo iste stolpce: `dataset, algorithm, fold, roc_auc, train_time_s,
inference_time_s`. Nič v tej mapi se ne ureja na roko — vse nastane iz zagona.

| Mapa | Zagon | Vsebina |
|---|---|---|
| `local/` | lokalni pilot, NVIDIA RTX 3060 (WSL2), 3 nabori | `results_local_subset.csv` (90 vrstic) in `preprocessing_log.md` |
| `arnes/subset/` | validacija prenosa na gručo Arnes (H100), isti 3 nabori | `results_arnes_subset.csv`, `per_dataset/` vhodi, oba `pip freeze`, `PROVENANCE.md` |
| `arnes/cc18/` | **rezultat diplome** — poln OpenML-CC18, 72 naborov | `results_arnes_cc18.csv` (2160 vrstic) in `PROVENANCE.md` |
| `per_dataset/` | scratch, ni v gitu | surovi izhod SLURM polja, vhod za `merge_results.py` |

Nekaj pravil, ki se jih splača poznati:

- **Tabela za diplomo je `arnes/cc18/results_arnes_cc18.csv`.** `src/summary.py`
  brez argumenta povzame lokalni pilot, kar je napačna tabela — pot je treba
  podati eksplicitno:
  `python -m src.summary results/arnes/cc18/results_arnes_cc18.csv`.
- **`local/results_local_subset.csv` je nedotakljiv.** To je izhodiščna
  meritev, s katero je bil primerjan prenos na gručo; združevanja z gruče gredo
  vedno v novo datoteko, nikoli sem.
- **`local/preprocessing_log.md` opisuje zadnji lokalni zagon.**
  `run_benchmark.py` ga ob vsakem zagonu prepiše na novo. Za CC18 tega dnevnika
  ni, ker ga arneška pot (`run_one_dataset.py`) ne piše.
- **`per_dataset/` je scratch in ne sodi med rezultate.** `run_one_dataset.py`
  obravnava obstoječ `<id>.csv` kot "already done" in nabor preskoči, zato je
  pred vsakim resnim zagonom treba `rm -rf results/per_dataset/*`. Kurirani
  rezultati se iz njega prekopirajo v svojo mapo s `PROVENANCE.md`.
