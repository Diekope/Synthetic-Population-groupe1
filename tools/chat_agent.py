#!/usr/bin/env python3
"""Interroge un agent (une persona) en conversation.

Lit `agents/agents.jsonl` produit par `tools/build_agents.py` et envoie le prompt
système de l'agent au fournisseur choisi.

Fournisseurs :
  mock    : hors ligne, réponses déterministes — sert aux tests du pipeline
  openai  : API compatible OpenAI (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL)
  ollama  : serveur local (OLLAMA_HOST, OLLAMA_MODEL)

Exemples :
  uv run python tools/chat_agent.py --list
  uv run python tools/chat_agent.py --agent 42 --message "Tu travailles dans quoi ?"
  uv run python tools/chat_agent.py --agent 42                  # REPL
  OPENAI_API_KEY=... uv run python tools/chat_agent.py --agent 42 --provider openai
  uv run python tools/chat_agent.py --agent 42 --provider ollama --model llama3.1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

QUIT = {"/quit", "/exit", "/q"}


class LLMError(RuntimeError):
    """Erreur de fournisseur, remontée proprement à la CLI ou à la GUI."""

HELP = """Commandes :
  /who              affiche l'agent courant
  /attrs            affiche ses attributs
  /agent <id|clé|nom> change d'agent
  /reset            efface l'historique
  /quit             quitte
"""


def load_agents(path: Path) -> list[dict]:
    agents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not agents:
        raise SystemExit(f"Aucun agent dans {path}")
    return agents


def select_agent(agents: list[dict], query: str) -> dict:
    q = query.strip().lower()
    if q == "random":
        return random.choice(agents)
    for a in agents:
        if str(a["id"]) == q or a["identity_key"].lower() == q:
            return a
    matches = [a for a in agents if f"{a['prenom']} {a['nom']}".lower() == q]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"Agent introuvable : {query!r} (essaie --list)")


def describe(agent: dict, with_attrs: bool = True) -> str:
    head = f"#{agent['id']:>3}  {agent['prenom']} {agent['nom']}  [{agent['identity_key']}]"
    if not with_attrs:
        return head
    attrs = "  ".join(f"{k}={v}" for k, v in agent["attributes"].items())
    return f"{head}\n      {attrs}"


def build_messages(agent: dict, history: list[dict]) -> list[dict]:
    return [{"role": "system", "content": agent["persona"]}, *history]


def call_openai(messages: list[dict], model: str) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY manquante.")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.8}).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise LLMError(f"API OpenAI : HTTP {e.code} — {e.read().decode(errors='replace')[:500]}")
    except urllib.error.URLError as e:
        raise LLMError(f"API OpenAI injoignable : {e.reason}")
    return data["choices"][0]["message"]["content"].strip()


def call_ollama(messages: list[dict], model: str) -> str:
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    payload = {"model": model, "messages": messages, "stream": False}
    # Les modèles « reasoning » (qwen3, deepseek-r1…) brûlent le budget en réflexion
    # avant de répondre. On la coupe par défaut ; OLLAMA_THINK=1 pour la réactiver.
    think = os.environ.get("OLLAMA_THINK", "").lower()
    if think:
        payload["think"] = think not in ("0", "false", "off", "no")
    elif any(tag in model.lower() for tag in ("qwen3", "deepseek-r1", "qwq", "reason")):
        payload["think"] = False
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
    except urllib.error.URLError as e:
        raise LLMError(f"Ollama injoignable ({base}) : {e.reason}. Lance `ollama serve`.")
    return data["message"]["content"].strip()


def call_mock(agent: dict, messages: list[dict]) -> str:
    """Réponse factice mais déterministe : vérifie le câblage sans réseau."""
    question = messages[-1]["content"]
    return (f"[mock] {agent['prenom']} {agent['nom']} "
            f"(identité {agent['identity_key']}) : je t'entends bien — « {question[:120]} ». "
            "Branche un vrai fournisseur (--provider openai|ollama) pour une vraie réponse.")


def respond(agent: dict, history: list[dict], provider: str, model: str) -> str:
    messages = build_messages(agent, history)
    if provider == "mock":
        return call_mock(agent, messages)
    if provider == "openai":
        return call_openai(messages, model)
    if provider == "ollama":
        return call_ollama(messages, model)
    raise LLMError(f"Fournisseur inconnu : {provider}")


def repl(agent: dict, agents: list[dict], provider: str, model: str, max_turns: int) -> None:
    history: list[dict] = []
    print(describe(agent))
    print("Discussion ouverte. /help pour les commandes.\n")
    while True:
        try:
            line = input("vous> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in QUIT:
            return
        if line == "/help":
            print(HELP)
            continue
        if line == "/who":
            print(describe(agent))
            continue
        if line == "/attrs":
            print(describe(agent, with_attrs=True))
            continue
        if line == "/reset":
            history.clear()
            print("historique effacé.")
            continue
        if line.startswith("/agent "):
            agent = select_agent(agents, line.split(maxsplit=1)[1])
            history.clear()
            print(describe(agent))
            continue
        history.append({"role": "user", "content": line})
        try:
            answer = respond(agent, history, provider, model)
        except LLMError as e:
            history.pop()
            print(f"erreur : {e}\n")
            continue
        history.append({"role": "assistant", "content": answer})
        if max_turns and len(history) > 2 * max_turns:
            del history[: len(history) - 2 * max_turns]
        print(f"\n{agent['prenom']}> {answer}\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=Path, default=Path("agents/agents.jsonl"))
    p.add_argument("--agent", default=None, help="id, clé d'identité, 'Prénom Nom' ou 'random'")
    p.add_argument("--list", action="store_true", help="liste les agents et quitte")
    p.add_argument("--message", "-m", default=None, help="message unique (non interactif)")
    p.add_argument("--provider", choices=["mock", "openai", "ollama"], default="mock")
    p.add_argument("--model", default=None,
                   help="défaut : OPENAI_MODEL / OLLAMA_MODEL / gpt-4o-mini / llama3.1")
    p.add_argument("--max-turns", type=int, default=12,
                   help="messages échangés conservés (0 = illimité)")
    args = p.parse_args()

    agents = load_agents(args.agents)
    if args.list:
        for a in agents:
            print(describe(a, with_attrs=False))
        return 0
    if args.agent is None:
        p.error("--agent est requis (ou --list). Exemple : --agent 1")

    agent = select_agent(agents, args.agent)
    model = args.model or os.environ.get("OPENAI_MODEL") or os.environ.get("OLLAMA_MODEL")
    if not model:
        model = {"openai": "gpt-4o-mini", "ollama": "llama3.1"}.get(args.provider, "mock")

    if args.message:
        history = [{"role": "user", "content": args.message}]
        try:
            print(respond(agent, history, args.provider, model))
        except LLMError as e:
            print(f"erreur : {e}", file=sys.stderr)
            return 2
        return 0

    repl(agent, agents, args.provider, model, args.max_turns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
