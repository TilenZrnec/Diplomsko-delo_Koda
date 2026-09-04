# razlaga_repozitorija

Pomožne datoteke za razumevanje kode. **Niso del eksperimenta** — nič tukaj ne
vpliva na rezultate v `results/`.

| Datoteka | Kaj je |
|---|---|
| `zaporedje.puml` | Diagram zaporedja (PlantUML): katera datoteka kliče katero, v štirih fazah |
| `preveri_diagram.py` | Preveri, da je diagram še usklajen z izvorno kodo |
| `hooks/pre-push` | Kljuka, ki to preverjanje požene ob vsakem `git push` |

## Diagram zaporedja

`zaporedje.puml` vsebuje **štiri ločene diagrame** v eni datoteki:

1. `01_priprava` — zamrznitev seznama naborov (`gen_cc18_ids.py`) in profiliranje
   njihovih velikosti (`profile_datasets.py`)
2. `02_lokalni_pilot` — `python -m src.run_benchmark` na 3 naborih
3. `03_arnes` — predpriprava, SLURM polje, združevanje rezultatov
4. `04_analiza` — `src/summary.py`, `compare_results.py`, `profile_medic3.py`

### Izris

Najlažje z razširitvijo **PlantUML** v VS Code (`Alt+D` za predogled). Iz ukazne
vrstice, če je nameščen `plantuml`:

```bash
plantuml razlaga_repozitorija/zaporedje.puml         # -> štirje PNG-ji
plantuml -tsvg razlaga_repozitorija/zaporedje.puml   # -> štirje SVG-ji
```

Če `plantuml` ni nameščen, zadošča `plantuml.jar` in java:

```bash
java -jar plantuml.jar -tpng razlaga_repozitorija/zaporedje.puml
```

Nastanejo `01_priprava.png`, `02_lokalni_pilot.png`, `03_arnes.png` in
`04_analiza.png` (imena določajo oznake za `@startuml`).

Izrisane slike so **gitignorirane** (`.gitignore` v tej mapi). Namenoma:
commitana slika bi lahko tiho kazala staro stanje, medtem ko je `.puml` že
posodobljen — točno tisto, kar `preveri_diagram.py` preprečuje. Vir resnice je
`.puml`, slika je vedno le izpeljanka.

### Ubežni znak `~`

V besedilu diagrama boš videl zapise `~__init~__.py`, `~--ids-file` in
`~#SBATCH`. PlantUML namreč `__tako__` izrisuje podčrtano, `--tako--`
prečrtano, `#` na začetku vrstice pa kot oštevilčen seznam — brez `~` bi se
`__init__.py` izrisal kot `init.py`, `--mem=64G` pa kot prečrtan `mem=64G`.
Znak `~` se **ne izriše**; `preveri_diagram.py` ga pred primerjavo odstrani,
zato ne moti preverjanja poti.

## Preverjanje usklajenosti

```bash
python3 razlaga_repozitorija/preveri_diagram.py
```

Preverja v obe smeri:

- **manjka v diagramu** — nova datoteka v `src/` ali `scripts/`, ki je nihče
  ni vrisal;
- **ne obstaja več** — pot, ki jo diagram omenja, a je bila preimenovana ali
  zbrisana.

Pokrite so `src/**/*.py`, `scripts/**/*.py`, `scripts/**/*.sh` in `config.yaml`.
Skripta uporablja samo standardno knjižnico, zato deluje s katerimkoli
sistemskim pythonom — okolje `tabular` ni potrebno.

## Namestitev kljuke pred potiskom

Kljuke (hooks) živijo v `.git/hooks/`, ki se prek GitHuba **ne prenaša**, zato
jih je treba na vsakem računalniku (PC in prenosnik) vklopiti posebej — z enim
ukazom iz korena repozitorija:

```bash
git config core.hooksPath razlaga_repozitorija/hooks
```

Preveri z `git config --get core.hooksPath`, izklopi z
`git config --unset core.hooksPath`.

Odslej vsak `git push` najprej požene preverjanje; če diagram ni usklajen,
potisk **ne gre skozi** in izpiše se, katere poti so odveč ali manjkajo.
Obvod za en sam potisk: `git push --no-verify`.
