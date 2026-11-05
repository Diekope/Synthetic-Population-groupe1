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
  uv run python tools/chat_agent.py --agent 42                  # REPL (flux en direct)
  uv run python tools/chat_agent.py --agent 42 --seed 0 --temperature 0.7 --max-tokens 200
  OPENAI_API_KEY=... uv run python tools/chat_agent.py --agent 42 --provider openai
  uv run python tools/chat_agent.py --agent 42 --provider ollama --model llama3.2:1b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

QUIT = {"/quit", "/exit", "/q"}
RETRY_CODES = {429, 500, 502, 503, 504}


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


def select_agent(agents: list[dict], query: str, seed: int | None = None) -> dict:
    q = query.strip().lower()
    if q == "random":
        return random.Random(seed).choice(agents)
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


# --------------------------------------------------------- budget de contexte

def estimate_tokens(text: str) -> int:
    """Estimation grossière (~4 caractères par token)."""
    return max(1, len(text or "") // 4)


def trim_history(history: list[dict], budget_tokens: int, *,
                 system_tokens: int = 0) -> list[dict]:
    """Ne garde que les messages les plus récents qui tiennent dans le budget.

    On garde toujours au moins le dernier message. Évite qu'un long historique
    fasse dépasser la fenêtre de contexte (et tronque silencieusement la persona).
    """
    kept: list[dict] = []
    total = system_tokens
    for m in reversed(history):
        t = estimate_tokens(m.get("content", ""))
        if kept and total + t > budget_tokens:
            break
        total += t
        kept.append(m)
    return list(reversed(kept))


# ------------------------------------------------------------------- réseau

def _backoff(attempt: int) -> float:
    return min(8.0, 0.5 * (2 ** attempt)) * (0.7 + 0.6 * random.random())


def _open(req: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)


def _read_openai_stream(r, on_token) -> str:
    parts: list[str] = []
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        choices = obj.get("choices") or [{}]
        piece = (choices[0].get("delta") or {}).get("content")
        if piece:
            parts.append(piece)
            if on_token:
                on_token(piece)
    return "".join(parts).strip()


def _read_ollama_stream(r, on_token) -> str:
    parts: list[str] = []
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("error"):
            raise LLMError(f"Ollama : {obj['error']}")
        piece = (obj.get("message") or {}).get("content")
        if piece:
            parts.append(piece)
            if on_token:
                on_token(piece)
        if obj.get("done"):
            break
    return "".join(parts).strip()


def call_openai(messages: list[dict], model: str, temperature: float = 0.8, *,
                max_tokens: int | None = None, seed: int | None = None,
                stream: bool = False, on_token=None, timeout: float = 120,
                retries: int = 2) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY manquante.")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload: dict = {"model": model, "messages": messages,
                     "temperature": temperature, "stream": stream}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = seed
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    last = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(f"{base}/chat/completions", data=body, headers=headers)
        try:
            with _open(req, timeout) as r:
                if stream:
                    return _read_openai_stream(r, on_token)
                return json.load(r)["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            if e.code in RETRY_CODES and attempt < retries:
                last = f"HTTP {e.code}"
                time.sleep(_backoff(attempt))
                continue
            raise LLMError(f"API OpenAI : HTTP {e.code} — {detail}")
        except urllib.error.URLError as e:
            if attempt < retries:
                last = str(e.reason)
                time.sleep(_backoff(attempt))
                continue
            raise LLMError(f"API OpenAI injoignable : {e.reason}")
    raise LLMError(f"API OpenAI : échec après {retries + 1} essais ({last})")


def call_ollama(messages: list[dict], model: str, temperature: float = 0.8, *,
                max_tokens: int | None = None, seed: int | None = None,
                stream: bool = False, on_token=None, timeout: float = 300,
                retries: int = 2) -> str:
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    # num_ctx : le défaut peut être trop petit et tronquer la persona dès que
    # l'historique grandit (échec silencieux). On le fixe explicitement.
    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
    options: dict = {"temperature": temperature, "num_ctx": num_ctx}
    if max_tokens:
        options["num_predict"] = max_tokens
    if seed is not None:
        options["seed"] = seed
    payload: dict = {"model": model, "messages": messages, "stream": stream,
                     "options": options}
    # Les modèles « reasoning » (qwen3, deepseek-r1…) brûlent le budget en réflexion
    # avant de répondre. On la coupe par défaut ; OLLAMA_THINK=1 pour la réactiver.
    think = os.environ.get("OLLAMA_THINK", "").lower()
    if think:
        payload["think"] = think not in ("0", "false", "off", "no")
    elif any(tag in model.lower() for tag in ("qwen3", "deepseek-r1", "qwq", "reason")):
        payload["think"] = False
    body = json.dumps(payload).encode()
    last = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(base + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with _open(req, timeout) as r:
                if stream:
                    return _read_ollama_stream(r, on_token)
                return json.load(r)["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 404:
                raise LLMError(f"Modèle Ollama introuvable : {model} (ollama pull {model})")
            if e.code in RETRY_CODES and attempt < retries:
                last = f"HTTP {e.code}"
                time.sleep(_backoff(attempt))
                continue
            raise LLMError(f"Ollama : HTTP {e.code} — {detail}")
        except urllib.error.URLError as e:
            if attempt < retries:
                last = str(e.reason)
                time.sleep(_backoff(attempt))
                continue
            raise LLMError(f"Ollama injoignable ({base}) : {e.reason}. Lance `ollama serve`.")
    raise LLMError(f"Ollama : échec après {retries + 1} essais ({last})")


def call_mock(agent: dict, messages: list[dict]) -> str:
    """Réponse factice mais déterministe : vérifie le câblage sans réseau."""
    question = messages[-1]["content"]
    return (f"[mock] {agent['prenom']} {agent['nom']} "
            f"(identité {agent['identity_key']}) : je t'entends bien — « {question[:120]} ». "
            "Branche un vrai fournisseur (--provider openai|ollama) pour une vraie réponse.")


def respond(agent: dict, history: list[dict], provider: str, model: str,
            temperature: float = 0.8, max_tokens: int | None = None,
            seed: int | None = None, stream: bool = False, on_token=None) -> str:
    messages = build_messages(agent, history)
    if provider == "mock":
        answer = call_mock(agent, messages)
        if on_token:
            on_token(answer)
        return answer
    if provider == "openai":
        return call_openai(messages, model, temperature, max_tokens=max_tokens,
                           seed=seed, stream=stream, on_token=on_token)
    if provider == "ollama":
        return call_ollama(messages, model, temperature, max_tokens=max_tokens,
                           seed=seed, stream=stream, on_token=on_token)
    raise LLMError(f"Fournisseur inconnu : {provider}")


def repl(agent: dict, agents: list[dict], provider: str, model: str, max_turns: int,
         *, stream: bool = True, max_tokens: int | None = None, seed: int | None = None,
         budget_tokens: int = 3000) -> None:
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
            agent = select_agent(agents, line.split(maxsplit=1)[1], seed)
            history.clear()
            print(describe(agent))
            continue
        history.append({"role": "user", "content": line})
        if max_turns and len(history) > 2 * max_turns:
            del history[: len(history) - 2 * max_turns]
        sent = trim_history(history, budget_tokens,
                            system_tokens=estimate_tokens(agent["persona"]))
        try:
            if stream:
                print(f"\n{agent['prenom']}> ", end="", flush=True)
                answer = respond(agent, sent, provider, model, max_tokens=max_tokens,
                                 seed=seed, stream=True,
                                 on_token=lambda piece: print(piece, end="", flush=True))
                print("\n")
            else:
                answer = respond(agent, sent, provider, model, max_tokens=max_tokens,
                                 seed=seed)
                print(f"\n{agent['prenom']}> {answer}\n")
        except LLMError as e:
            history.pop()
            print(f"\nerreur : {e}\n")
            continue
        history.append({"role": "assistant", "content": answer})


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
    p.add_argument("--max-tokens", type=int, default=None,
                   help="longueur maximale de la réponse")
    p.add_argument("--seed", type=int, default=None,
                   help="graine (sélection 'random' et échantillonnage du modèle)")
    p.add_argument("--budget-tokens", type=int, default=3000,
                   help="budget de contexte pour l'historique")
    p.add_argument("--no-stream", action="store_true", help="désactive le flux en direct")
    args = p.parse_args()

    agents = load_agents(args.agents)
    if args.list:
        for a in agents:
            print(describe(a, with_attrs=False))
        return 0
    if args.agent is None:
        p.error("--agent est requis (ou --list). Exemple : --agent 1")

    agent = select_agent(agents, args.agent, args.seed)
    model = args.model or os.environ.get("OPENAI_MODEL") or os.environ.get("OLLAMA_MODEL")
    if not model:
        model = {"openai": "gpt-4o-mini", "ollama": "llama3.2:1b"}.get(args.provider, "mock")

    if args.message:
        history = [{"role": "user", "content": args.message}]
        try:
            print(respond(agent, history, args.provider, model,
                          max_tokens=args.max_tokens, seed=args.seed))
        except LLMError as e:
            print(f"erreur : {e}", file=sys.stderr)
            return 2
        return 0

    repl(agent, agents, args.provider, model, args.max_turns,
         stream=not args.no_stream, max_tokens=args.max_tokens, seed=args.seed,
         budget_tokens=args.budget_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
