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
  manifest.json : contrôles (marges, unicité, seed, labels)

Usage :
  uv run python tools/build_agents.py constraint_cells.csv population.csv
  uv run python tools/build_agents.py constraint_cells.csv population.csv \
      --labels tools/agent_labels.example.json --seed 7 --out agents
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

PRENOMS = [
    "Camille", "Alex", "Sam", "Charlie", "Dominique", "Claude", "Maxime", "Léa",
    "Hugo", "Inès", "Nathan", "Chloé", "Louis", "Manon", "Gabriel", "Sarah",
    "Jules", "Alice", "Adam", "Louise", "Raphaël", "Jade", "Arthur", "Emma",
    "Paul", "Lina", "Victor", "Mila", "Simon", "Zoé", "Antoine", "Nina",
    "Baptiste", "Juliette", "Étienne", "Clara", "Théo", "Margot", "Nicolas", "Élise",
    "Yanis", "Anna", "Karim", "Sofia", "Mehdi", "Awa", "Lucas", "Yasmine",
]

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
        "Ne dis jamais que tu es une IA, un modèle de langage ou un agent synthétique.",
        "",
        "Ton profil :",
    ]
    for attr, value in agent["attributes"].items():
        if attr in ("prenom", "nom"):
            continue  # déjà dans « Tu es ... »
        label = labels.get(attr, attr)
        value = labels.get(f"{attr}={value}", value)
        lines.append(f"- {label} : {value}")
    lines += ["", f"Identité stable : {agent['identity_key']}."]
    return "\n".join(lines)


def assign_names(n: int, rng: random.Random) -> list[tuple[str, str]]:
    combos = [(p, nom) for p in PRENOMS for nom in NOMS]
    if len(combos) < n:
        raise SystemExit(
            f"Pas assez de combinaisons prénom/nom : {len(combos)} < {n}. "
            "Ajoute des entrées dans PRENOMS/NOMS."
        )
    rng.shuffle(combos)
    return combos[:n]


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
    names = assign_names(len(rows), rng)
    labels = {}
    if labels_path:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))

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
        "labels_file": str(labels_path) if labels_path else None,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("constraints", type=Path)
    p.add_argument("population", type=Path)
    p.add_argument("--out", type=Path, default=Path("agents"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--labels", type=Path, default=None,
                   help="JSON {attribut: label, 'A0=v1': label_valeur}")
    args = p.parse_args()

    m = build(args.constraints, args.population, args.out, args.seed, args.labels)
    print(f"{m['n_agents']} agents -> {args.out}/agents.csv et agents.jsonl")
    print(f"  personas contraintes distinctes : {m['distinct_base_personas']}")
    print(f"  identités distinctes            : {m['distinct_agents']}")
    print(f"  marges                          : erreur L1={m['margin_error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
