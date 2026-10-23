#!/usr/bin/env python3
"""Transforme une population exacte en agents identifiables et interrogeables.

Le problème : avec les 45 tables de `constraint_cells.csv`, seules ~302 personas
sont distinctes sur 500 individus. Une identité unique ne peut donc pas être
construite avec les seuls attributs contraints A0..A11.

Solution : on ajoute deux attributs HORS TABLES (`prenom`, `nom`), tirés sans
remise. Ils ne changent aucune marge (les tables ne les couvrent pas) et
garantissent que les 500 vecteurs d'attributs sont distincts. L'identité
publique est un hash déterministe de TOUS les attributs (contraints + identité),
donc elle est bien dérivée des attributs, pas d'un compteur arbitraire.

Sorties (dans --out, défaut `agents/`) :
  agents.csv    : id, identity_key, prenom, nom, A0..A11
  agents.jsonl  : un agent par ligne, avec son prompt système prêt pour un LLM
  labels.json   : libellés lisibles, consommés par la GUI
  manifest.json : contrôles (marges, unicité, seed, labels)

Les libellés de `tools/agent_labels.fr.json` sont appliqués par défaut. Ils
rendent le prompt parlant (`"classe d'âge : 40 à 59 ans"`) et les prénoms
cohérents avec le sexe. `--no-labels` revient aux `A0=v1` bruts.

Usage :
  uv run python tools/build_agents.py constraint_cells.csv population.csv
  uv run python tools/build_agents.py constraint_cells.csv population.csv \
      --labels mon_fichier.json --seed 7 --out agents
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PRENOMS_F = [
    "Léa", "Inès", "Chloé", "Manon", "Sarah", "Alice", "Louise", "Jade",
    "Emma", "Lina", "Mila", "Zoé", "Nina", "Juliette", "Clara", "Margot",
    "Élise", "Anna", "Sofia", "Awa", "Yasmine", "Camille", "Dominique", "Claude",
]

PRENOMS_M = [
    "Hugo", "Nathan", "Louis", "Gabriel", "Jules", "Adam", "Raphaël", "Arthur",
    "Paul", "Victor", "Simon", "Antoine", "Baptiste", "Étienne", "Théo", "Nicolas",
    "Yanis", "Karim", "Mehdi", "Lucas", "Maxime", "Alex", "Sam", "Charlie",
]

PRENOMS_N = ["Camille", "Sam", "Charlie", "Dominique", "Claude", "Alex"]

NOMS = [
    "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand",
    "Leroy", "Moreau", "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David",
    "Bertrand", "Roux", "Vincent", "Fournier", "Morel", "Girard", "André", "Lefèvre",
    "Mercier", "Dupont", "Lambert", "Bonnet", "François", "Martinez", "Legrand", "Garnier",
    "Faure", "Rousseau", "Blanc", "Guérin", "Muller", "Henry", "Roussel", "Nicolas",
    "Perrin", "Morin", "Mathieu", "Clément", "Gauthier", "Dumont", "Lopez", "Fontaine",
]


def natural_key(name: str):
    """A2 < A10, v2 < v10."""
    m = re.match(r"^(.*?)(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def load_tables(path: Path):
    tables: dict[str, list[tuple[tuple[tuple[str, str], ...], int]]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cell = tuple(sorted(tuple(kv.split("=", 1)) for kv in row["cell"].split(";")))
            tables.setdefault(row["table_id"], []).append((cell, int(row["target"])))
    return tables


def margin_error(population: list[dict[str, str]], tables) -> tuple[int, int]:
    """Erreur L1 et nombre de cellules violées, recalculés depuis zéro."""
    error = violated = 0
    for cells in tables.values():
        scope = [a for a, _ in cells[0][0]]
        counts = Counter(tuple((a, row[a]) for a in scope) for row in population)
        for cell, target in cells:
            d = abs(counts.get(cell, 0) - target)
            error += d
            violated += d != 0
    return error, violated


def identity_key(values: dict[str, str]) -> str:
    """Identité déterministe, dérivée de tous les attributs (ordre stable)."""
    payload = "|".join(f"{k}={v}" for k, v in sorted(values.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def persona_prompt(agent: dict, labels: dict[str, str]) -> str:
    """Prompt système : une personne, pas un modèle. Les labels sont optionnels."""
    lines = [
        f"Tu es {agent['prenom']} {agent['nom']}.",
        "Tu participes à une discussion en tant que personne réelle. Réponds à la "
        "première personne, avec un ton naturel, et reste cohérent avec ton profil. "
        "Ne dis jamais que tu es une IA, un modèle de langage ou un agent synthétique. "
        "Si une question sort de ton profil, improvise une réponse plausible sans "
        "sortir du personnage.",
        "",
        "Ton profil :",
    ]
    for attr, value in agent["attributes"].items():
        if attr in ("prenom", "nom"):
            continue  # déjà dans « Tu es ... »
        label = labels.get(attr, attr)
        value = labels.get(f"{attr}={value}", value)
        lines.append(f"- {label} : {value}")
    return "\n".join(lines)


def name_pool(row: dict[str, str], labels: dict[str, str]) -> str:
    """Catégorie de prénoms cohérente avec le sexe, si les libellés le disent."""
    sex = labels.get("A1=" + row.get("A1", ""), "").lower()
    if "femme" in sex:
        return "f"
    if "homme" in sex:
        return "m"
    return "n" if sex else "all"


def assign_names(rows: list[dict[str, str]], labels: dict[str, str],
                 rng: random.Random) -> list[tuple[str, str]]:
    pools = {"f": PRENOMS_F, "m": PRENOMS_M, "n": PRENOMS_N,
             "all": PRENOMS_F + PRENOMS_M + PRENOMS_N}
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(name_pool(row, labels), []).append(i)

    out: list[tuple[str, str] | None] = [None] * len(rows)
    used: set[tuple[str, str]] = set()
    for key, idxs in groups.items():
        pool = pools[key]
        combos = [(p, nom) for p in pool for nom in NOMS]
        if len(combos) < len(idxs):
            raise SystemExit(
                f"Pas assez de combinaisons prénom/nom pour {len(idxs)} agents "
                f"({len(combos)} disponibles). Ajoute des noms dans NOMS.")
        rng.shuffle(combos)
        picked: list[tuple[str, str]] = []
        for combo in combos:
            if combo in used:
                continue
            used.add(combo)
            picked.append(combo)
            if len(picked) == len(idxs):
                break
        if len(picked) < len(idxs):
            raise SystemExit(
                f"Pas assez de combinaisons prénom/nom uniques pour {len(idxs)} agents. "
                "Ajoute des noms dans NOMS.")
        for i, name in zip(idxs, picked):
            out[i] = name
    return out  # type: ignore[return-value]


def build(constraints: Path, population_csv: Path, out: Path, seed: int,
          labels_path: Path | None) -> dict:
    tables = load_tables(constraints)
    with population_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Population vide.")
    attrs = sorted((k for k in rows[0] if k != "id"), key=natural_key)

    before_error, before_violated = margin_error(rows, tables)
    if before_error:
        raise SystemExit(
            f"Population d'entrée inexacte : erreur L1={before_error} "
            f"sur {before_violated} cellules. Corrige-la avant d'ajouter des agents."
        )

    rng = random.Random(seed)
    labels = {}
    labels_ref = None
    if labels_path:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        try:
            labels_ref = str(labels_path.resolve().relative_to(
                Path(__file__).resolve().parent.parent))
        except ValueError:
            labels_ref = str(labels_path)
    names = assign_names(rows, labels, rng)

    agents = []
    for i, (row, (prenom, nom)) in enumerate(zip(rows, names), start=1):
        values = {a: row[a] for a in attrs}
        values["prenom"] = prenom
        values["nom"] = nom
        agent = {
            "id": i,
            "prenom": prenom,
            "nom": nom,
            "identity_key": identity_key(values),
            "attributes": values,
        }
        agent["persona"] = persona_prompt(agent, labels)
        agents.append(agent)

    # Unicité : aucune collision de hash, aucun vecteur d'attributs en double.
    keys = {a["identity_key"] for a in agents}
    vectors = {tuple(sorted(a["attributes"].items())) for a in agents}
    if len(keys) != len(agents) or len(vectors) != len(agents):
        raise SystemExit(
            f"Identités non uniques : {len(keys)} hash et {len(vectors)} vecteurs "
            f"pour {len(agents)} agents. Change la seed ou ajoute des noms."
        )

    out.mkdir(parents=True, exist_ok=True)
    columns = ["id", "identity_key", "prenom", "nom", *attrs]
    with (out / "agents.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for a in agents:
            w.writerow({**{c: a[c] for c in ("id", "identity_key", "prenom", "nom")},
                        **{x: a["attributes"][x] for x in attrs}})

    with (out / "agents.jsonl").open("w", encoding="utf-8") as f:
        for a in agents:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    after_error, after_violated = margin_error(
        [a["attributes"] for a in agents], tables)
    manifest = {
        "n_agents": len(agents),
        "distinct_base_personas": len({tuple(r[a] for a in attrs) for r in rows}),
        "distinct_agents": len(vectors),
        "identity_collisions": len(agents) - len(keys),
        "margin_error": after_error,
        "margin_violated_cells": after_violated,
        "seed": seed,
        "identity_attributes": ["prenom", "nom"],
        "constrained_attributes": attrs,
        "labels_file": labels_ref,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Libellés lisibles pour la GUI (le prompt, lui, est déjà écrit dans agents.jsonl).
    labels_out = out / "labels.json"
    if labels:
        labels_out.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    elif labels_out.exists():
        labels_out.unlink()
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("constraints", type=Path)
    p.add_argument("population", type=Path)
    p.add_argument("--out", type=Path, default=Path("agents"))
    p.add_argument("--seed", type=int, default=0)
    default_labels = Path(__file__).resolve().parent / "agent_labels.fr.json"
    p.add_argument("--labels", type=Path, default=default_labels,
                   help="JSON {attribut: label, 'A0=v1': label_valeur} "
                        f"(défaut : {default_labels.name})")
    p.add_argument("--no-labels", action="store_true", help="attributs affichés bruts (A0=v1)")
    args = p.parse_args()

    labels = None if args.no_labels else args.labels
    if labels is not None and not labels.exists():
        print(f"Fichier de libellés introuvable : {labels}", file=sys.stderr)
        return 1
    m = build(args.constraints, args.population, args.out, args.seed, labels)
    print(f"{m['n_agents']} agents -> {args.out}/agents.csv et agents.jsonl")
    print(f"  personas contraintes distinctes : {m['distinct_base_personas']}")
    print(f"  identités distinctes            : {m['distinct_agents']}")
    print(f"  marges                          : erreur L1={m['margin_error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
