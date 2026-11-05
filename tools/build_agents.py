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
  agents.csv    : id, identity_key, cluster_id, copy_index, copies, prenom, nom, A0..A11
  agents.jsonl  : un agent par ligne, avec son prompt système prêt pour un LLM
  labels.json   : libellés lisibles, consommés par la GUI
  manifest.json : contrôles (marges, unicité, seed, labels)

Les copies exactes sont CONSERVÉES : deux agents qui partagent les mêmes 12
attributs contraints restent deux lignes, avec la même `cluster_id`. Les champs
`copy_index` (1..N) et `copies` (N) rendent la duplication visible au lieu de la
masquer derrière des prénoms différents.

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
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path

TOOL = "build_agents"
TOOL_VERSION = "2"
ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "synthetic-population/agents")

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


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def persona_prompt(agent: dict, labels: dict[str, str]) -> str:
    """Prompt système : une personne, pas un modèle. Les labels sont optionnels."""
    lines = [
        f"Tu es {agent['prenom']} {agent['nom']}, une personne réelle. Tu réponds "
        "toujours en français, à la première personne, avec un ton naturel.",
        "Tu dois rester strictement cohérent avec le profil ci-dessous. Si une "
        "question porte sur une information présente dans ton profil (âge, sexe, "
        "logement, revenu, diplôme, nationalité, enfants, santé, équipement…), tu "
        "réponds avec EXACTEMENT la valeur de ton profil, sans l'inventer ni la "
        "contredire. Tu ne dis jamais que tu es une IA, un modèle de langage ou un "
        "agent synthétique. Pour une question hors profil, tu improvises une réponse "
        "plausible qui ne contredit pas ton profil.",
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

    # Profil = les 12 attributs contraints. Plusieurs agents peuvent partager
    # exactement le même profil : on le rend explicite via cluster_id/copy_index.
    bases: dict[tuple[str, ...], list[int]] = {}
    for i, row in enumerate(rows):
        bases.setdefault(tuple(row[a] for a in attrs), []).append(i)
    cluster_ids = {
        base: "p" + hashlib.sha256(
            "|".join(f"{a}={v}" for a, v in zip(attrs, base)).encode("utf-8")
        ).hexdigest()[:7]
        for base in bases
    }
    copy_index = {i: k for base, idxs in bases.items() for k, i in enumerate(idxs, start=1)}

    agents = []
    for i, (row, (prenom, nom)) in enumerate(zip(rows, names), start=1):
        values = {a: row[a] for a in attrs}
        values["prenom"] = prenom
        values["nom"] = nom
        base = tuple(row[a] for a in attrs)
        agent = {
            "id": i,
            "stable_id": str(uuid.uuid5(ID_NAMESPACE, identity_key(values))),
            "prenom": prenom,
            "nom": nom,
            "identity_key": identity_key(values),
            "cluster_id": cluster_ids[base],
            "copy_index": copy_index[i - 1],
            "copies": len(bases[base]),
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
    columns = ["id", "stable_id", "identity_key", "cluster_id", "copy_index", "copies",
               "prenom", "nom", *attrs]
    meta_cols = ("id", "stable_id", "identity_key", "cluster_id", "copy_index", "copies",
                 "prenom", "nom")
    with (out / "agents.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for a in agents:
            w.writerow({**{c: a[c] for c in meta_cols},
                        **{x: a["attributes"][x] for x in attrs}})

    with (out / "agents.jsonl").open("w", encoding="utf-8") as f:
        for a in agents:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    after_error, after_violated = margin_error(
        [a["attributes"] for a in agents], tables)
    manifest = {
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "n_agents": len(agents),
        "distinct_base_personas": len(bases),
        "distinct_profiles": len(bases),
        "distinct_agents": len(vectors),
        "duplicated_agents": sum(len(v) for v in bases.values() if len(v) > 1),
        "exact_copies": len(agents) - len(bases),
        "max_copies": max(len(v) for v in bases.values()),
        "identity_collisions": len(agents) - len(keys),
        "margin_error": after_error,
        "margin_violated_cells": after_violated,
        "seed": seed,
        "identity_attributes": ["prenom", "nom"],
        "profile_attributes": attrs,
        "cluster_field": "cluster_id",
        "constrained_attributes": attrs,
        "labels_file": labels_ref,
        "inputs": {
            "constraints_sha256": file_sha256(constraints),
            "population_sha256": file_sha256(population_csv),
            "labels_sha256": file_sha256(labels_path),
        },
        "git_commit": git_commit(),
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


def check(constraints: Path, population_csv: Path, out: Path, seed: int,
          labels_path: Path | None) -> int:
    """Vérifie que `out` correspond exactement aux entrées actuelles.

    Reconstruit dans un dossier temporaire et compare octet à octet ; n'écrit rien.
    """
    if not out.exists():
        print(f"Dossier absent : {out}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        manifest = build(constraints, population_csv, fresh, seed, labels_path)
        ok = True
        for name in ("agents.csv", "agents.jsonl", "labels.json"):
            a, b = fresh / name, out / name
            if not b.exists() and not a.exists():
                continue
            if not (a.exists() and b.exists()):
                print(f"  {name}: présence différente", file=sys.stderr)
                ok = False
            elif a.read_bytes() != b.read_bytes():
                print(f"  {name}: contenu différent", file=sys.stderr)
                ok = False
        committed = out / "manifest.json"
        if committed.exists():
            want = json.loads(committed.read_text(encoding="utf-8"))
            # git_commit est informatif (le commit courant change après coup) :
            # on l'ignore dans la comparaison.
            want.pop("git_commit", None)
            fresh_manifest = dict(manifest)
            fresh_manifest.pop("git_commit", None)
            if want != fresh_manifest:
                print("  manifest.json: différent", file=sys.stderr)
                ok = False
        else:
            print("  manifest.json: absent", file=sys.stderr)
            ok = False
    if ok:
        print(f"{out} est à jour (entrées inchangées, seed {seed}).")
        return 0
    print("Vérification échouée : relance build_agents.py.", file=sys.stderr)
    return 1


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
    p.add_argument("--check", action="store_true",
                   help="vérifie que --out correspond aux entrées actuelles (n'écrit rien)")
    args = p.parse_args()

    labels = None if args.no_labels else args.labels
    if labels is not None and not labels.exists():
        print(f"Fichier de libellés introuvable : {labels}", file=sys.stderr)
        return 1
    if args.check:
        return check(args.constraints, args.population, args.out, args.seed, labels)
    m = build(args.constraints, args.population, args.out, args.seed, labels)
    print(f"{m['n_agents']} agents -> {args.out}/agents.csv et agents.jsonl")
    print(f"  identités uniques                : {m['distinct_agents']}")
    print(f"  profils distincts (12 attributs) : {m['distinct_profiles']}")
    print(f"  copies en trop conservées       : {m['exact_copies']} "
          f"({m['duplicated_agents']} agents concernés, max ×{m['max_copies']}, champ cluster_id)")
    print(f"  marges                           : erreur L1={m['margin_error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
