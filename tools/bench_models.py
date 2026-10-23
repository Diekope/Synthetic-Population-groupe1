#!/usr/bin/env python3
"""Compare des modèles Ollama sur une vraie persona : latence, tokens/s, RAM.

Un seul modèle sert les 500 agents (la persona est dans le prompt) : le coût est
la mémoire du modèle + le contexte, pas « un modèle par agent ».

  uv run python tools/bench_models.py                       # modèles installés
  uv run python tools/bench_models.py --models qwen2.5:0.5b,llama3.2:1b,llama3.2:3b
  uv run python tools/bench_models.py --models qwen2.5:0.5b --pull

Prérequis : `ollama serve` lancé. `--pull` télécharge les modèles manquants.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "http://localhost:11434"


def api(path: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def installed() -> dict[str, float]:
    try:
        return {m["name"]: m.get("size", 0) / 1e9 for m in api("/api/tags", timeout=3)["models"]}
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise SystemExit("Ollama injoignable. Lance `ollama serve` puis réessaie.")


def pull(model: str) -> None:
    print(f"  téléchargement de {model}…", end="", flush=True)
    req = urllib.request.Request(HOST + "/api/pull", data=json.dumps({"model": model}).encode(),
                                 headers={"Content-Type": "application/json"})
    last = -1
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            total, done = msg.get("total"), msg.get("completed")
            if total and done:
                pct = int(100 * done / total)
                if pct >= last + 10:
                    print(f" {pct}%", end="", flush=True)
                    last = pct
            if msg.get("error"):
                raise SystemExit(f"\n  échec : {msg['error']}")
    print(" ok")


def persona_prompt(agents: Path) -> str:
    if agents.exists():
        try:
            first = json.loads(agents.read_text(encoding="utf-8").splitlines()[0])
            return first["persona"]
        except (ValueError, IndexError, KeyError):
            pass
    return ("Tu es Camille Martin. Réponds à la première personne, de façon naturelle, "
            "sans jamais dire que tu es une IA.")


def bench(model: str, system: str, question: str, runs: int, num_predict: int) -> dict:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"num_predict": num_predict, "temperature": 0.8}}
    if any(tag in model.lower() for tag in ("qwen3", "deepseek-r1", "qwq", "reason")):
        payload["think"] = False

    # chargement + warmup (non mesuré)
    api("/api/chat", payload, timeout=900)

    walls, tps, samples = [], [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = api("/api/chat", payload, timeout=900)
        walls.append(time.perf_counter() - t0)
        ev = r.get("eval_count") or 0
        dur = r.get("eval_duration") or 0
        if ev and dur:
            tps.append(ev / (dur / 1e9))
        samples.append((r.get("message") or {}).get("content", ""))

    try:
        ps = {m["name"]: m for m in api("/api/ps", timeout=3)["models"]}
    except (urllib.error.URLError, TimeoutError, ValueError):
        ps = {}
    vram = ps.get(model, {}).get("size_vram", 0) / 1e9

    return {
        "model": model,
        "wall_ms": statistics.median(walls) * 1000,
        "tok_s": statistics.median(tps) if tps else 0,
        "vram_gb": vram,
        "sample": samples[0].replace("\n", " ")[:90],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="", help="liste séparée par des virgules (défaut : installés)")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--num-predict", type=int, default=120)
    p.add_argument("--question", default="Présente-toi en deux phrases, puis dis ce que tu penses du travail.")
    p.add_argument("--agents", type=Path, default=Path("agents/agents.jsonl"))
    p.add_argument("--pull", action="store_true", help="télécharge les modèles manquants")
    args = p.parse_args()

    have = installed()
    models = [m.strip() for m in args.models.split(",") if m.strip()] or list(have)
    if not models:
        print("Aucun modèle installé. Exemple : uv run python tools/bench_models.py "
              "--models qwen2.5:0.5b --pull", file=sys.stderr)
        return 1

    system = persona_prompt(args.agents)
    print(f"Machine cible : 8 Go de RAM (M2). Un modèle partagé par les 500 agents.\n")
    print(f"{'modèle':<22}{'taille':>8}{'latence':>10}{'tok/s':>8}{'VRAM':>8}  extrait")
    print("-" * 100)
    rows = []
    for m in models:
        if m not in have:
            if not args.pull:
                print(f"{m:<22}{'absent':>8}   (ajoute --pull)")
                continue
            pull(m)
            have = installed()
        try:
            r = bench(m, system, args.question, args.runs, args.num_predict)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"{m:<22}  erreur : {e}")
            continue
        rows.append(r)
        print(f"{r['model']:<22}{have.get(m, 0):>7.1f}G{r['wall_ms']:>9.0f}ms"
              f"{r['tok_s']:>8.1f}{r['vram_gb']:>7.1f}G  {r['sample']}")

    print("\nLecture : latence = réponse médiane complète ; tok/s = génération seule.")
    if rows:
        fast = max(rows, key=lambda r: r["tok_s"])
        print(f"Le plus rapide ici : {fast['model']} ({fast['tok_s']:.0f} tok/s). "
              "Sur 8 Go, viser un modèle ≤ 2,5 Go (Q4) laisse de la place au navigateur.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
