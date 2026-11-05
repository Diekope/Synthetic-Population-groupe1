#!/usr/bin/env python3
"""Mesure la fidélité d'un modèle aux profils : pose à chaque agent une question
factuelle par attribut et vérifie que la réponse correspond bien à son profil.

C'est l'outil qui manquait pour choisir un modèle : le benchmark ne mesurait que
la vitesse, pas le respect des attributs.

Les sondes sont dans `tools/fidelity_probes.fr.json` (une question par attribut,
et les motifs acceptés par valeur). Les libellés de `agent_labels.fr.json` sont
ajoutés automatiquement aux motifs.

Fournisseurs : mock (hors ligne ; répond parfaitement, pour valider le harnais),
openai, ollama.

Exemples :
  # valider le harnais hors ligne (fidélité attendue = 100 %)
  uv run python tools/eval_fidelity.py --provider mock --limit 10

  # comparer plusieurs modèles sur les 302 profils distincts
  uv run python tools/eval_fidelity.py --one-per-cluster \
      --models llama3.2:1b,llama3.2:3b,qwen3:0.6b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_agent  # noqa: E402
import survey as survey_mod  # noqa: E402

DEFAULT_PROBES = Path(__file__).resolve().parent / "fidelity_probes.fr.json"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)          # « 1 500 » -> « 1500 »
    s = re.sub(r"[^a-z0-9]+", " ", s)               # ponctuation -> espace
    return " ".join(s.split())


def token_in(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])",
                     haystack) is not None


def matches(answer: str, spec: dict) -> bool:
    a = norm(answer)
    for pat in spec.get("accept", []):
        if token_in(a, norm(pat)):
            return True
    rng = spec.get("range")
    if rng:
        for num in re.findall(r"\d+", a):
            if rng[0] <= int(num) <= rng[1]:
                return True
    return False


def load_probes(path: Path, labels: dict[str, str]) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    probes = data.get("attributes") or {}
    if not probes:
        raise SystemExit(f"Aucune sonde dans {path}")
    for attr, probe in probes.items():
        for value, spec in probe.get("values", {}).items():
            label = labels.get(f"{attr}={value}")
            if label:
                spec.setdefault("accept", [])
                if label not in spec["accept"]:
                    spec["accept"].append(label)   # le libellé exact est accepté
    return probes


# ------------------------------------------------------------------- exécution

def _ask(agent: dict, attr: str, probe: dict, provider: str, model: str,
         temperature: float, labels: dict[str, str]) -> str:
    if provider == "mock":
        value = agent["attributes"][attr]
        return f"Pour moi, c'est clair : {labels.get(f'{attr}={value}', value)}."
    history = [{"role": "user",
                "content": probe["question"] + " Réponds en une phrase courte."}]
    return chat_agent.respond(agent, history, provider, model, temperature)


def evaluate(agents: list[dict], probes: dict[str, dict], *, provider: str,
             model: str, temperature: float = 0.0, labels: dict[str, str] | None = None,
             concurrency: int = 2, progress: bool = True,
             on_progress=None) -> list[dict]:
    labels = labels or {}
    tasks = [(a, attr) for a in agents for attr in probes]
    total = len(tasks)
    results: list[dict] = []
    lock = threading.Lock()
    counter = {"n": 0}
    if progress:
        print(f"{len(agents)} agents × {len(probes)} attributs = {total} questions "
              f"({provider}/{model})")

    def work(agent: dict, attr: str) -> dict:
        probe = probes[attr]
        expected = agent["attributes"][attr]
        spec = probe["values"][expected]
        error = None
        try:
            answer = _ask(agent, attr, probe, provider, model, temperature, labels)
        except chat_agent.LLMError as e:
            answer, error = "", str(e)
        except Exception as e:  # noqa: BLE001
            answer, error = "", f"{type(e).__name__}: {e}"
        ok = (not error) and matches(answer, spec)
        return {
            "agent_id": agent["id"], "identity_key": agent["identity_key"],
            "cluster_id": agent.get("cluster_id"),
            "attribute": attr, "question": probe["question"],
            "expected": expected, "expected_label": labels.get(f"{attr}={expected}", expected),
            "answer": answer, "correct": ok, "error": error,
        }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futures = [ex.submit(work, a, attr) for a, attr in tasks]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                results.append(rec)
                counter["n"] += 1
                if on_progress:
                    on_progress(counter["n"], total)
                if progress and (counter["n"] % 25 == 0 or counter["n"] == total):
                    print(f"  {counter['n']}/{total}")
    return results


def aggregate(results: list[dict], agents: list[dict]) -> dict:
    per_attr: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_agent: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    errors = 0
    mismatches = []
    for r in results:
        if r["error"]:
            errors += 1
        per_attr[r["attribute"]][1] += 1
        per_agent[r["agent_id"]][1] += 1
        if r["correct"]:
            per_attr[r["attribute"]][0] += 1
            per_agent[r["agent_id"]][0] += 1
        elif not r["error"] and len(mismatches) < 40:
            mismatches.append({
                "agent_id": r["agent_id"], "attribute": r["attribute"],
                "expected": r["expected_label"], "answer": r["answer"][:200]})
    correct = sum(c for c, _ in per_attr.values())
    total = sum(t for _, t in per_attr.values()) or 1
    return {
        "n_agents": len(agents), "n_questions": len(results), "errors": errors,
        "overall": round(correct / total, 4),
        "per_attribute": {a: round(c / t, 4) for a, (c, t) in sorted(per_attr.items())},
        "per_agent": {str(i): round(c / t, 4) for i, (c, t) in sorted(per_agent.items())},
        "mismatches": mismatches,
    }


def print_report(model: str, report: dict) -> None:
    print(f"\n=== fidélité {model} — {report['overall']*100:.1f} % "
          f"({report['n_questions']} questions, {report['errors']} erreurs) ===")
    for attr, acc in report["per_attribute"].items():
        bar = "█" * round(acc * 24)
        print(f"  {attr:<5} {acc*100:5.1f} %  {bar}")


# ------------------------------------------------------------------------- CLI

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=Path, default=Path("agents/agents.jsonl"))
    p.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    p.add_argument("--labels", type=Path, default=None,
                   help="défaut : labels.json à côté des agents")
    p.add_argument("--provider", choices=["mock", "openai", "ollama"], default="mock")
    p.add_argument("--model", default=None)
    p.add_argument("--models", default=None,
                   help="liste (provider ollama) à comparer, ex. a,b,c")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--where", action="append")
    p.add_argument("--ids", default=None)
    p.add_argument("--one-per-cluster", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="réponses brutes JSONL")
    p.add_argument("--report", type=Path, default=None, help="rapport JSON")
    args = p.parse_args()

    agents = chat_agent.load_agents(args.agents)
    labels_path = args.labels or (args.agents.parent / "labels.json")
    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    probes = load_probes(args.probes, labels)

    ids = [int(x) for x in args.ids.split(",")] if args.ids else None
    selected = survey_mod.select_agents(
        agents, where=survey_mod.parse_where(args.where), ids=ids,
        one_per_cluster=args.one_per_cluster, limit=args.limit, seed=args.seed)
    if not selected:
        p.error("aucun agent sélectionné")

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    if models:
        provider = "ollama"
        run_models = models
    else:
        provider = args.provider
        run_models = [args.model or {"openai": "gpt-4o-mini", "ollama": "llama3.2:1b",
                                     "mock": "mock"}[provider]]

    summaries = {}
    for model in run_models:
        results = evaluate(selected, probes, provider=provider, model=model,
                           temperature=args.temperature, labels=labels,
                           concurrency=max(1, min(4, args.concurrency)))
        report = aggregate(results, selected)
        report.update({"provider": provider, "model": model})
        summaries[model] = report
        print_report(model, report)
        if args.out:
            out = args.out if len(run_models) == 1 else args.out.with_name(
                f"{args.out.stem}.{model.replace('/', '_').replace(':', '_')}{args.out.suffix}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps({**r, "model": model}, ensure_ascii=False) + "\n")

    if len(run_models) > 1:
        print("\n=== comparaison ===")
        for model in run_models:
            print(f"  {model:<28} {summaries[model]['overall']*100:5.1f} %")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"\nrapport : {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
