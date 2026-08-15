"""Profilira nabor Medic3 - "podatkovna kartica" pred kakršnimkoli modeliranjem.

Medic3 ni z OpenML, ampak lokalna CSV datoteka zaupnih (anonimiziranih in
zamegljenih) medicinskih podatkov. Preden ga sploh damo v benchmark, moramo
vedeti, kaj v njem je: koliko razredov, koliko manjka, ali sta stolpca Index
in Field vira "bližnjice" (leakage) in ali manjkajočost sama nosi informacijo
o ciljni spremenljivki.

Skripta namenoma uporablja samo standardno knjižnico in bere datoteko v
pretoku (vrstico po vrstico), zato deluje tudi v okolju brez pandas in ne
naloži 142 MB v pomnilnik naenkrat.

Zagon iz korena repozitorija:

    python scripts/profile_medic3.py [--path data/medic3/Medic3.csv]
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

# Stolpca, ki po navodilu mentorja nista atributa: Index je zaporedna številka
# vrstice, Field je (predzadnji stolpec) oznaka področja/oddelka. Class je cilj.
ID_COL = "Index"
FIELD_COL = "Field"
TARGET_COL = "Class"


def profile(path):
    csv.field_size_limit(10**9)

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_i = header.index(ID_COL)
        fld_i = header.index(FIELD_COL)
        cls_i = header.index(TARGET_COL)
        # Indeksi pravih atributov (vse razen Index/Field/Class).
        feat = [i for i in range(len(header)) if i not in (idx_i, fld_i, cls_i)]

        n = 0
        class_n = Counter()               # razred -> št. vrstic
        field_n = Counter()               # Field  -> št. vrstic
        field_class = defaultdict(Counter)  # Field -> razdelitev razredov
        col_obs = Counter()               # stolpec -> v koliko vrsticah ima vrednost
        obs_by_class = defaultdict(Counter)  # stolpec -> (razred -> opažen)
        obs_per_row = Counter()           # št. opaženih atributov -> št. vrstic
        nonnumeric = defaultdict(Counter)  # stolpec -> ne-številske vrednosti
        distinct = {i: set() for i in feat}  # omejeno na 51 vrednosti (kardinalnost)

        for row in reader:
            n += 1
            if len(row) != len(header):
                print(f"OPOZORILO: vrstica {n} ima {len(row)} polj namesto {len(header)}")
                continue

            cls = row[cls_i]
            class_n[cls] += 1
            field_n[row[fld_i]] += 1
            field_class[row[fld_i]][cls] += 1

            n_obs = 0
            for i in feat:
                v = row[i]
                if v == "":
                    continue
                n_obs += 1
                col_obs[i] += 1
                obs_by_class[i][cls] += 1
                if len(distinct[i]) <= 50:
                    distinct[i].add(v)
                try:
                    float(v)
                except ValueError:
                    nonnumeric[i][v] += 1
            obs_per_row[n_obs] += 1

    return {
        "header": header, "n": n, "feat": feat, "class_n": class_n,
        "field_n": field_n, "field_class": field_class, "col_obs": col_obs,
        "obs_by_class": obs_by_class, "obs_per_row": obs_per_row,
        "nonnumeric": nonnumeric, "distinct": distinct,
    }


def report(p):
    header, n, feat = p["header"], p["n"], p["feat"]
    class_n, col_obs = p["class_n"], p["col_obs"]

    print(f"=== Oblika ===\nvrstic: {n}\nstolpcev: {len(header)} "
          f"(od tega {len(feat)} atributov + {ID_COL}/{FIELD_COL}/{TARGET_COL})")

    # --- Ciljna spremenljivka -------------------------------------------------
    counts = class_n.most_common()
    print(f"\n=== Ciljna spremenljivka {TARGET_COL} ===")
    print(f"različnih razredov: {len(class_n)}  <- to NI binarni problem")
    print(f"najpogostejši: {counts[:5]}")
    print(f"najredkejši:   {counts[-5:]}")
    sizes = sorted(class_n.values())
    print(f"vrstic na razred: min={sizes[0]} mediana={sizes[len(sizes)//2]} max={sizes[-1]}")
    print(f"5-kratni stratificirani CV zahteva >=5 vrstic na razred: "
          f"{'OK' if sizes[0] >= 5 else 'NE GRE'}")
    print(f"delež večinskega razreda: {100*sizes[-1]/n:.1f}%  "
          f"(naključno ugibanje med {len(class_n)} razredi: {100/len(class_n):.2f}%)")

    # --- Manjkajoče vrednosti -------------------------------------------------
    total = n * len(feat)
    miss = total - sum(col_obs[i] for i in feat)
    print(f"\n=== Manjkajoče vrednosti (samo atributi) ===")
    print(f"skupaj: {miss}/{total} = {100*miss/total:.1f}%")
    buckets = Counter()
    for i in feat:
        frac = col_obs[i] / n
        buckets[">50%" if frac > .5 else "10-50%" if frac > .1
                else "2-10%" if frac > .02 else "<2%"] += 1
    print("stolpci po deležu vrstic, kjer imajo vrednost:")
    for k in (">50%", "10-50%", "2-10%", "<2%"):
        print(f"   {k:>7}: {buckets[k]:3d} stolpcev")
    obs_rows = sorted(p["obs_per_row"].items())
    cum, med = 0, None
    for k, c in obs_rows:
        cum += c
        if med is None and cum >= n / 2:
            med = k
    print(f"opaženih atributov na vrstico: min={obs_rows[0][0]} "
          f"mediana={med} max={obs_rows[-1][0]} (od {len(feat)})")

    # --- Tipi -----------------------------------------------------------------
    nn = [header[i] for i in p["nonnumeric"]]
    lowcard = sorted((len(p["distinct"][i]), header[i]) for i in feat
                     if len(p["distinct"][i]) <= 50)
    print(f"\n=== Tipi ===")
    print(f"stolpcev z ne-številskimi vrednostmi: {len(nn)} {nn[:5]}")
    print(f"stolpcev z <=50 različnimi vrednostmi (kandidati za kategorične): "
          f"{len(lowcard)} od {len(feat)}")
    print(f"   najnižja kardinalnost: {lowcard[:5]}")

    # --- Bližnjice (leakage) --------------------------------------------------
    print(f"\n=== Bližnjice ===")
    # Field: če vedno napovemo najpogostejši razred znotraj Fielda, kako dobro gre?
    hit = sum(max(v.values()) for v in p["field_class"].values())
    print(f"{FIELD_COL}: {len(p['field_n'])} različnih vrednosti; napoved samo iz "
          f"{FIELD_COL} da {100*hit/n:.1f}% točnost "
          f"(večinski razred: {100*max(class_n.values())/n:.1f}%) -> zato ga odstranimo")

    # Manjkajočost kot napovednik: kako zelo se delež opaženosti stolpca
    # razlikuje med razredi? Velik razpon = "kateri izvid je naročen" izda diagnozo.
    big = [c for c in class_n if class_n[c] >= 200]
    spread = []
    for i in feat:
        if col_obs[i] < 0.02 * n:
            continue
        rates = sorted((p["obs_by_class"][i][c] / class_n[c], c) for c in big)
        spread.append((rates[-1][0] - rates[0][0], header[i], rates[-1], rates[0]))
    spread.sort(reverse=True)
    print(f"\nmanjkajočost sama kot napovednik {TARGET_COL} "
          f"(razredi z >=200 vrsticami, 5 najmočnejših stolpcev):")
    for d, name, hi, lo in spread[:5]:
        print(f"   {name}: opažen pri {hi[1]} v {100*hi[0]:.1f}% vrstic, "
              f"pri {lo[1]} v {100*lo[0]:.1f}% -> razpon {100*d:.1f} odstotnih točk")
    print("   -> vzorec manjkajočosti je močan signal; imputacija ga uniči, "
          "zato ga je treba ohraniti (npr. z indikatorji manjkajočosti)")


def main():
    parser = argparse.ArgumentParser(description="Profiliranje nabora Medic3.")
    parser.add_argument("--path", default=os.path.join("data", "medic3", "Medic3.csv"),
                        help="pot do Medic3.csv")
    args = parser.parse_args()
    report(profile(args.path))


if __name__ == "__main__":
    main()
