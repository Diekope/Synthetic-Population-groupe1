#!/usr/bin/env python3
"""Sonde une population d'agents : pose les mêmes questions à tout ou partie des
agents, puis agrège les réponses.

Chaque agent est une persona (voir `tools/build_agents.py`). Le questionnaire est
un fichier JSON :

  {
    "title": "Opinions 2026",
    "questions": [
      {"id": "q1", "text": "Êtes-vous favorable au télétravail ?",
       "choices": ["oui", "non", "ne se prononce pas"]},
      {"id": "q2", "text": "Quel est votre principal souci ?"}
    ]
  }

Une question SANS "choices" est ouverte ; une question AVEC "choices" est fermée.

Sélection de la population (par défaut : les 500 agents) :
  --where A1=v2        ne garde que les agents vérifiant attribut=valeur (répétable)
  --agents 1,5,23      identifiants explicites
  --one-per-cluster    un seul agent par profil contraint (302 au lieu de 500)
  --limit 50 --seed 0  échantillon aléatoire reproductible

Sortie :
  --out reponses.jsonl   une ligne par (agent, question) ; **reprise** automatique :
                         les couples déjà présents sont sautés, on peut relancer.
  --report rapport.json  agrégats (choix, répartition par attribut, erreurs).

Fournisseurs : mock (hors ligne, déterministe), openai, ollama — comme chat_agent.
Concurrence bornée (1–4 conseillé pour Ollama).

Exemples :
  uv run python tools/survey.py surveys/demo.fr.json --provider mock --limit 8
  uv run python tools/survey.py surveys/demo.fr.json --one-per-cluster \
      --provider ollama --model llama3.2:1b --concurrency 2
  uv run python tools/survey.py surveys/demo.fr.json --where A1=v2 --crosstab A1,A9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_agent  # noqa: E402


# ---------------------------------------------------------------- questionnaire

def load_survey(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("questions") or []
    if not raw:
        raise SystemExit(f"Aucune question dans {path}")
    questions = []
    seen = set()
    for i, q in enumerate(raw, start=1):
        if isinstance(q, str):
            q = {"text": q}
        if not q.get("text"):
            raise SystemExit(f"Question {i} sans texte dans {path}")
        qid = str(q.get("id") or f"q{i}")
        if qid in seen:
            raise SystemExit(f"Identifiant de question en double : {qid}")
        seen.add(qid)
        choices = q.get("choices")
        if choices is not None and (not isinstance(choices, list) or not choices):
            raise SystemExit(f"Choix invalides pour la question {qid}")
        questions.append({"id": qid, "text": q["text"], "choices": choices})
    return {"title": data.get("title") or path.stem, "questions": questions}


# ---------------------------------------------------------------------- choix

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().strip().strip(".!,;:").split())


def match_choice(answer: str, choices: list[str]) -> str | None:
    """Rapproche une réponse libre d'un des choix autorisés (sinon None)."""
    a = _norm(answer)
    if not a:
        return None
    for c in sorted(choices, key=len, reverse=True):  # le plus spécifique d'abord
        if _norm(c) == a:
            return c
    for c in sorted(choices, key=len, reverse=True):
        if _norm(c) in a:
            return c
    return None


# ------------------------------------------------------------------ sélection

def select_agents(agents: list[dict], *, where: list[str] | None = None,
                  ids: list[int] | None = None, one_per_cluster: bool = False,
                  limit: int | None = None, seed: int = 0) -> list[dict]:
    sel = list(agents)
    for clause in (where or []):
        if "=" not in clause:
            raise SystemExit(f"--where attendu sous la forme attribut=valeur, reçu {clause!r}")
        attr, value = clause.split("=", 1)
        sel = [a for a in sel if a["attributes"].get(attr) == value]
    if ids:
        wanted = set(ids)
        sel = [a for a in sel if a["id"] in wanted]
    if one_per_cluster:
        seen, kept = set(), []
        for a in sel:
            cid = a.get("cluster_id", a["identity_key"])
            if cid not in seen:
                seen.add(cid)
                kept.append(a)
        sel = kept
    if limit is not None and limit < len(sel):
        sel = random.Random(seed).sample(sel, limit)
        sel.sort(key=lambda a: a["id"])
    return sel


def parse_where(values: list[str] | None) -> list[str]:
    return [v for raw in (values or []) for v in raw.split(",") if v.strip()]


# ------------------------------------------------------------------- prompts

def survey_prompt(question: dict) -> str:
    lines = [f"Question : {question['text']}"]
    if question.get("choices"):
        lines.append("Choisis une seule réponse parmi :")
        lines += [f"- {c}" for c in question["choices"]]
        lines.append("Réponds uniquement par le libellé exact du choix, sans rien ajouter.")
    else:
        lines.append("Réponds en une ou deux phrases, à la première personne.")
    return "\n".join(lines)


def mock_answer(agent: dict, question: dict) -> str:
    """Réponse simulée mais déterministe : valide le pipeline sans réseau."""
    seed = int(hashlib.sha256(
        f"{agent['identity_key']}|{question['id']}".encode()).hexdigest(), 16)
    rng = random.Random(seed)
    if question.get("choices"):
        return rng.choice(question["choices"])
    return (f"({agent['prenom']} {agent['nom']}) réponse simulée à "
            f"« {question['text']} » — branche un vrai fournisseur pour une vraie réponse.")


def ask(agent: dict, question: dict, provider: str, model: str,
        temperature: float) -> str:
    if provider == "mock":
        return mock_answer(agent, question)
    history = [{"role": "user", "content": survey_prompt(question)}]
    return chat_agent.respond(agent, history, provider, model, temperature)


# --------------------------------------------------------------------- exécution

def _ordered_questions(survey: dict, agent: dict, shuffle: bool) -> list[dict]:
    qs = survey["questions"]
    if not shuffle:
        return qs
    seed = int(hashlib.sha256(agent["identity_key"].encode()).hexdigest(), 16)
    qs = list(qs)
    random.Random(seed).shuffle(qs)
    return qs


def run_survey(agents: list[dict], survey: dict, *, provider: str, model: str,
               out: Path, concurrency: int = 2, temperature: float = 0.8,
               shuffle: bool = False, retries: int = 1,
               progress: bool = True, on_progress=None, force: bool = False) -> list[dict]:
    """Exécute la sonde. Écrit `out` au fil de l'eau, reprend si le fichier existe
    (sauf `force=True`, qui repart de zéro)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    answers: list[dict] = []
    if out.exists() and not force:
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done.add((rec["identity_key"], rec["question_id"]))
            answers.append(rec)

    tasks = []
    for agent in agents:
        for q in _ordered_questions(survey, agent, shuffle):
            if (agent["identity_key"], q["id"]) not in done:
                tasks.append((agent, q))

    total = len(tasks)
    if progress:
        print(f"{len(agents)} agents · {len(survey['questions'])} questions · "
              f"{total} requête(s) à faire "
              f"({len(answers)} déjà présentes, reprise)")
    if on_progress:
        on_progress(0, total)
    if not total:
        return answers

    lock = threading.Lock()
    fh = out.open("w" if force else "a", encoding="utf-8")
    counter = {"n": 0}

    def work(agent: dict, question: dict) -> dict:
        t0 = time.perf_counter()
        error = None
        text = ""
        for attempt in range(retries + 1):
            try:
                text = ask(agent, question, provider, model, temperature)
                error = None
                break
            except chat_agent.LLMError as e:
                error = str(e)
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
        choice = match_choice(text, question["choices"]) if question.get("choices") else None
        rec = {
            "survey": survey["title"],
            "agent_id": agent["id"],
            "identity_key": agent["identity_key"],
            "cluster_id": agent.get("cluster_id"),
            "name": f"{agent['prenom']} {agent['nom']}",
            "question_id": question["id"],
            "question": question["text"],
            "choices": question.get("choices"),
            "answer": text,
            "choice": choice,
            "provider": provider,
            "model": model,
            "ms": round((time.perf_counter() - t0) * 1000),
        }
        if error:
            rec["error"] = error
        return rec

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures = {ex.submit(work, a, q): (a, q) for a, q in tasks}
            for fut in as_completed(futures):
                rec = fut.result()
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    answers.append(rec)
                    counter["n"] += 1
                    if on_progress:
                        on_progress(counter["n"], total)
                    if progress and (counter["n"] % 10 == 0 or counter["n"] == total):
                        print(f"  {counter['n']}/{total}")
    finally:
        fh.close()
    return answers


# ------------------------------------------------------------------ agrégation

def aggregate(survey: dict, answers: list[dict], agents: list[dict],
              crosstab: list[str] | None = None) -> dict:
    by_key = {a["identity_key"]: a for a in agents}
    report = {"title": survey["title"], "n_agents": len(agents),
              "n_answers": len(answers), "questions": []}
    for q in survey["questions"]:
        rows = [a for a in answers if a["question_id"] == q["id"]]
        errors = sum(1 for r in rows if r.get("error"))
        entry = {"id": q["id"], "text": q["text"], "choices": q.get("choices"),
                 "n": len(rows), "errors": errors}
        if q.get("choices"):
            counts = Counter((r.get("choice") or "hors-choix")
                             for r in rows if not r.get("error"))
            total = sum(counts.values()) or 1
            entry["counts"] = dict(counts.most_common())
            entry["proportions"] = {k: round(v / total, 4)
                                    for k, v in counts.most_common()}
            if crosstab:
                entry["crosstab"] = {}
                for attr in crosstab:
                    tab = defaultdict(Counter)
                    for r in rows:
                        if r.get("error"):
                            continue
                        a = by_key.get(r["identity_key"])
                        if a:
                            tab[a["attributes"].get(attr, "?")][r.get("choice") or "hors-choix"] += 1
                    entry["crosstab"][attr] = {k: dict(v) for k, v in tab.items()}
        else:
            entry["samples"] = [r["answer"] for r in rows if not r.get("error")][:10]
        report["questions"].append(entry)
    return report


def print_report(report: dict) -> None:
    print(f"\n=== {report['title']} — {report['n_agents']} agents, "
          f"{report['n_answers']} réponses ===")
    for q in report["questions"]:
        print(f"\n[{q['id']}] {q['text']}")
        if q["errors"]:
            print(f"  ({q['errors']} erreur(s))")
        if q["choices"]:
            for choice, n in q["counts"].items():
                p = q["proportions"][choice]
                bar = "█" * round(p * 30)
                print(f"  {choice:<22} {n:>5}  {p*100:5.1f} %  {bar}")
            for attr, tab in (q.get("crosstab") or {}).items():
                print(f"  par {attr} :")
                for value, counts in tab.items():
                    parts = ", ".join(f"{k}={v}" for k, v in counts.items())
                    print(f"    {value:<20} {parts}")
        else:
            for s in q["samples"]:
                print(f"  · {s}")


# ------------------------------------------------------------------------- CLI

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("survey", type=Path, help="questionnaire JSON")
    p.add_argument("--agents", type=Path, default=Path("agents/agents.jsonl"))
    p.add_argument("--out", type=Path, default=None,
                   help="réponses JSONL (défaut : surveys/<nom>.answers.jsonl)")
    p.add_argument("--report", type=Path, default=None,
                   help="rapport JSON (défaut : surveys/<nom>.report.json)")
    p.add_argument("--provider", choices=["mock", "openai", "ollama"], default="mock")
    p.add_argument("--model", default=None)
    p.add_argument("--concurrency", type=int, default=2,
                   help="requêtes en parallèle (1–4 conseillé pour Ollama)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--where", action="append",
                   help="filtre attribut=valeur (répétable, ou séparé par des virgules)")
    p.add_argument("--ids", default=None, help="identifiants séparés par des virgules")
    p.add_argument("--one-per-cluster", action="store_true",
                   help="un seul agent par profil contraint")
    p.add_argument("--limit", type=int, default=None, help="échantillon aléatoire")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle", action="store_true",
                   help="ordre des questions aléatoire par agent")
    p.add_argument("--crosstab", default=None,
                   help="attributs à croiser avec les choix, ex. A1,A9")
    p.add_argument("--force", action="store_true",
                   help="ignore un fichier de réponses existant (repart de zéro)")
    p.add_argument("--no-report", action="store_true")
    args = p.parse_args()

    agents = chat_agent.load_agents(args.agents)
    if not args.survey.exists():
        p.error(f"questionnaire introuvable : {args.survey}")
    survey = load_survey(args.survey)
    stem = args.survey.stem
    out = args.out or Path("surveys") / f"{stem}.answers.jsonl"
    report_path = args.report or Path("surveys") / f"{stem}.report.json"

    ids = [int(x) for x in args.ids.split(",")] if args.ids else None
    selected = select_agents(agents, where=parse_where(args.where), ids=ids,
                             one_per_cluster=args.one_per_cluster,
                             limit=args.limit, seed=args.seed)
    if not selected:
        p.error("aucun agent sélectionné (vérifie --where/--ids/--limit)")

    model = args.model or chat_agent_default_model(args.provider)
    concurrency = max(1, args.concurrency)
    if args.provider == "ollama" and concurrency > 4:
        print("note : Ollama met les requêtes en file ; 1–4 suffit. "
              f"Je borne à 4 (demandé {concurrency}).", file=sys.stderr)
        concurrency = 4

    answers = run_survey(selected, survey, provider=args.provider, model=model,
                         out=out, concurrency=concurrency,
                         temperature=args.temperature, shuffle=args.shuffle,
                         retries=args.retries, force=args.force)
    crosstab = [x for x in (args.crosstab or "").split(",") if x]
    report = aggregate(survey, answers, selected, crosstab)
    print_report(report)

    if not args.no_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"\nréponses : {out}\nrapport  : {report_path}")
    return 0


def chat_agent_default_model(provider: str) -> str:
    import os
    env = os.environ.get("OPENAI_MODEL") or os.environ.get("OLLAMA_MODEL")
    if env:
        return env
    return {"openai": "gpt-4o-mini", "ollama": "llama3.2:1b", "mock": "mock"}[provider]


if __name__ == "__main__":
    sys.exit(main())
