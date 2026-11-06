#!/usr/bin/env python3
"""Simulation sociale — Phases A à F : conversations dans un espace 2D.

Chaque tick, les agents se déplacent légèrement. Avec une probabilité
`spark_rate`, un agent initiateur choisit un partenaire d'autant plus probable
qu'il est proche et que leur affinité est élevée, puis les deux discutent d'un
sujet tiré dans une grande liste. Les sujets récents ne sont pas réutilisés
avant un tour de liste. Un agent invité peut refuser de discuter
(`refusal_rate`), même quand la paire est imposée en mode manuel ; l'affinité
module ce refus et évolue après chaque conversation.

Des groupes persistants de 3 à 6 agents se forment par affinité et proximité.
Ils admettent des membres, en perdent, se dissolvent, et discutent ensemble à
plusieurs avec une session vierge par participant. `--resume` recharge un état
précédent et continue le même run sans tronquer les journaux.

Chaque paire garde une mémoire de ses sujets et de sa dernière humeur ; un
sujet déjà abordé avec la même personne est moins probable. Chaque agent a une
opinion par sujet, initialisée de façon déterministe et rapprochée de celle de
ses interlocuteurs après chaque échange.

Le débit se règle avec ``--conversations-per-tick`` (plusieurs conversations
par tick) et ``--parallel`` (génération simultanée). Les répliques sont
produites en parallèle, puis leurs effets sont appliqués séquentiellement :
le résultat reste identique à un run séquentiel, à seed égal.

Sorties (dans le dossier du run) :
  config.json          paramètres du run
  events.jsonl         un événement par conversation (ou erreur)
  transcripts/<id>.json  transcript complet de chaque conversation
  state.json           positions, tick, affinités, groupes, opinions, mémoire

Usage :
  uv run python tools/social_sim.py --provider mock --max-ticks 5
  uv run python tools/social_sim.py --provider ollama --model llama3.2:3b \
      --spark-rate 0.3 --tick-seconds 2 --max-ticks 50
  uv run python tools/social_sim.py --limit 12 --spark-rate 1 --max-ticks 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_agent  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENTS = ROOT / "agents" / "agents.jsonl"
DEFAULT_TOPICS = Path(__file__).resolve().parent / "topics.fr.json"
DEFAULT_OUT = ROOT / "social" / "runs"
DEFAULT_AFFINITY = 0.5
DEFAULT_OPINION = 0.5


# --------------------------------------------------------------------- sujets

def load_topics(path: Path) -> list[dict]:
    """Charge et valide la liste de sujets.

    Le fichier peut être soit une liste, soit un objet ``{"topics": [...]}``.
    Chaque sujet a au minimum un ``text`` ; ``id``, ``category``, ``tone`` et
    ``sensitive`` sont complétés par des valeurs par défaut.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("topics") if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"Aucun sujet dans {path}")
    topics: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(raw, start=1):
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            raise SystemExit(f"Sujet {i} sans texte dans {path}")
        tid = str(item.get("id") or f"t{i}")
        if tid in seen:
            raise SystemExit(f"Identifiant de sujet en double : {tid}")
        seen.add(tid)
        topics.append({
            "id": tid,
            "text": str(item["text"]).strip(),
            "category": str(item.get("category") or "divers"),
            "tone": str(item.get("tone") or "neutre"),
            "sensitive": bool(item.get("sensitive", False)),
        })
    return topics


def load_agents(path: Path) -> list[dict]:
    return chat_agent.load_agents(path)


# ------------------------------------------------------------------- espace 2D

def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ------------------------------------------------------------------- moteur

class Simulation:
    def __init__(
        self,
        agents: list[dict],
        topics: list[dict],
        out_dir: Path,
        *,
        seed: int = 0,
        provider: str = "mock",
        model: str = "mock",
        spark_rate: float = 0.2,
        tick_seconds: float = 0.0,
        max_ticks: int = 50,
        min_turns: int = 4,
        max_turns: int = 8,
        distance_scale: float = 0.15,
        move_sigma: float = 0.01,
        temperature: float = 0.8,
        max_tokens: int | None = None,
        budget_tokens: int = 3000,
        max_context_turns: int = 12,
        stream: bool = True,
        refusal_rate: float = 0.1,
        group_min: int = 3,
        group_max: int = 5,
        group_affinity: float = 0.5,
        group_radius: float = 0.25,
        group_rate: float = 0.35,
        max_groups: int = 20,
        memory_size: int = 5,
        opinion_influence: float = 0.15,
        repeat_penalty: float = 0.7,
        state_every: int = 0,
        conversations_per_tick: int = 1,
        parallel: int = 1,
        resume: dict | None = None,
        resume_source: str | None = None,
        agent_models: dict[int, str] | None = None,
        verbose: bool = True,
    ):
        if len(agents) < 2:
            raise SystemExit("Il faut au moins deux agents pour simuler une conversation.")
        if not topics:
            raise SystemExit("Il faut au moins un sujet.")
        if max_turns < 1 or min_turns < 1 or min_turns > max_turns:
            raise SystemExit("Bornes de tours invalides (1 <= min_turns <= max_turns).")
        if distance_scale <= 0:
            raise SystemExit("distance_scale doit être strictement positif.")
        if group_min < 2 or group_max < group_min:
            raise SystemExit("Bornes de groupe invalides (2 <= group_min <= group_max).")
        if not 0.0 <= group_affinity <= 1.0:
            raise SystemExit("group_affinity doit être dans 0..1.")
        if group_radius <= 0:
            raise SystemExit("group_radius doit être strictement positif.")
        if memory_size < 1:
            raise SystemExit("memory_size doit être au moins 1.")
        if not 0.0 <= opinion_influence <= 1.0:
            raise SystemExit("opinion_influence doit être dans 0..1.")
        if not 0.0 <= repeat_penalty <= 1.0:
            raise SystemExit("repeat_penalty doit être dans 0..1.")
        if conversations_per_tick < 1:
            raise SystemExit("conversations_per_tick doit être au moins 1.")
        if parallel < 1:
            raise SystemExit("parallel doit être au moins 1.")

        self.agents = agents
        self.topics = topics
        self.out_dir = Path(out_dir)
        self.seed = seed
        self.provider = provider
        self.model = model
        self.spark_rate = max(0.0, min(1.0, float(spark_rate)))
        self.tick_seconds = max(0.0, float(tick_seconds))
        self.max_ticks = max(0, int(max_ticks))
        self.min_turns = int(min_turns)
        self.max_turns = int(max_turns)
        self.distance_scale = float(distance_scale)
        self.move_sigma = max(0.0, float(move_sigma))
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.budget_tokens = max(200, int(budget_tokens))
        self.max_context_turns = max(1, int(max_context_turns))
        self.stream = bool(stream)
        self.refusal_rate = max(0.0, min(1.0, float(refusal_rate)))
        self.group_min = max(2, int(group_min))
        self.group_max = max(self.group_min, int(group_max))
        self.group_affinity = float(group_affinity)
        self.group_radius = float(group_radius)
        self.group_rate = max(0.0, min(1.0, float(group_rate)))
        self.max_groups = max(0, int(max_groups))
        self.memory_size = max(1, int(memory_size))
        self.opinion_influence = max(0.0, min(1.0, float(opinion_influence)))
        self.repeat_penalty = max(0.0, min(1.0, float(repeat_penalty)))
        self.state_every = max(0, int(state_every))
        self.conversations_per_tick = max(1, int(conversations_per_tick))
        self.parallel = max(1, int(parallel))
        self.verbose = verbose

        self.rng = random.Random(seed)
        self.spatial_rng = random.Random(seed ^ 0x5EED)
        self.by_id = {a["id"]: a for a in agents}
        self.topic_text = {t["id"]: t["text"] for t in topics}
        self.agent_models: dict[int, str] = {}
        for key, value in (agent_models or {}).items():
            aid = int(key)
            if aid not in self.by_id:
                raise SystemExit(f"Agent inconnu dans agent_models : {aid}")
            self.agent_models[aid] = str(value)
        self.positions: dict[int, tuple[float, float]] = {
            a["id"]: (self.spatial_rng.random(), self.spatial_rng.random())
            for a in agents
        }
        self.recent_topics: deque[str] = deque(maxlen=max(1, min(40, len(topics) // 3)))
        self.tick = 0
        self.conversation_no = 0
        self.stats = {"conversations": 0, "refusals": 0, "turns": 0,
                      "errors": 0, "group_conversations": 0}
        # Affinité symétrique entre deux agents, 0 (froide) à 1 (forte).
        # Les paires non modifiées valent DEFAULT_AFFINITY.
        self.affinity: dict[tuple[int, int], float] = {}
        # Groupes persistants : id stable -> membres et cohésion.
        self.groups: dict[str, dict] = {}
        self.group_no = 0
        # Opinions par agent et par sujet, 0 (opposé) à 1 (favorable).
        self.opinions: dict[int, dict[str, float]] = {}
        # Mémoire sociale par paire : sujets abordés et dernière humeur.
        self.memory: dict[tuple[int, int], dict] = {}

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir = self.out_dir / "transcripts"
        self.transcripts_dir.mkdir(exist_ok=True)
        self.events_path = self.out_dir / "events.jsonl"
        if resume is None:
            self.events_path.write_text("", encoding="utf-8")
        elif not self.events_path.exists():
            self.events_path.write_text("", encoding="utf-8")
        self.resumed_from = resume_source
        if resume:
            self._apply_state(resume)
        self._write_config()

    # ----- configuration et journalisation

    def _write_config(self) -> None:
        config = {
            "phase": "A+B+C+D+E+F",
            "seed": self.seed,
            "provider": self.provider,
            "model": self.model,
            "resumed_from": self.resumed_from,
            "spark_rate": self.spark_rate,
            "tick_seconds": self.tick_seconds,
            "max_ticks": self.max_ticks,
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "distance_scale": self.distance_scale,
            "move_sigma": self.move_sigma,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "budget_tokens": self.budget_tokens,
            "max_context_turns": self.max_context_turns,
            "stream": self.stream,
            "refusal_rate": self.refusal_rate,
            "group_min": self.group_min,
            "group_max": self.group_max,
            "group_affinity": self.group_affinity,
            "group_radius": self.group_radius,
            "group_rate": self.group_rate,
            "max_groups": self.max_groups,
            "memory_size": self.memory_size,
            "opinion_influence": self.opinion_influence,
            "repeat_penalty": self.repeat_penalty,
            "state_every": self.state_every,
            "conversations_per_tick": self.conversations_per_tick,
            "parallel": self.parallel,
            "agent_models": {str(k): v for k, v in sorted(self.agent_models.items())},
            "n_agents": len(self.agents),
            "n_topics": len(self.topics),
        }
        (self.out_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _log_event(self, event: dict) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_transcript(self, conversation_id: str, event: dict) -> None:
        (self.transcripts_dir / f"{conversation_id}.json").write_text(
            json.dumps(event, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _apply_state(self, data: dict) -> None:
        """Recharge positions, affinités, sujets récents et groupes."""
        for key, value in (data.get("positions") or {}).items():
            try:
                aid = int(key)
                x, y = (float(v) for v in value)
            except (TypeError, ValueError):
                continue
            if aid in self.by_id:
                self.positions[aid] = (_clip01(x), _clip01(y))
        for key, value in (data.get("affinity") or {}).items():
            try:
                aid, bid = (int(part) for part in str(key).split(":", 1))
                score = float(value)
            except (TypeError, ValueError):
                continue
            if aid in self.by_id and bid in self.by_id and aid != bid:
                self.set_affinity(aid, bid, score)
        for tid in (data.get("recent_topics") or []):
            if isinstance(tid, str) and tid:
                self.recent_topics.append(tid)
        for key, raw in (data.get("opinions") or {}).items():
            try:
                aid = int(key)
            except (TypeError, ValueError):
                continue
            if aid not in self.by_id or not isinstance(raw, dict):
                continue
            values: dict[str, float] = {}
            for tid, value in raw.items():
                try:
                    score = float(value)
                except (TypeError, ValueError):
                    continue
                values[str(tid)] = max(0.0, min(1.0, score))
            if values:
                self.opinions[aid] = values
        for key, raw in (data.get("memory") or {}).items():
            if not isinstance(raw, dict):
                continue
            try:
                aid, bid = (int(part) for part in str(key).split(":", 1))
            except (TypeError, ValueError):
                continue
            if aid not in self.by_id or bid not in self.by_id or aid == bid:
                continue
            topics = [str(t) for t in (raw.get("topics") or [])
                      if isinstance(t, str)][-self.memory_size:]
            self.memory[self._pair(aid, bid)] = {
                "topics": topics,
                "last_tick": int(raw.get("last_tick") or 0),
                "conversations": int(raw.get("conversations") or 0),
                "refusals": int(raw.get("refusals") or 0),
                "last_delta": float(raw.get("last_delta") or 0),
                "last_topic": raw.get("last_topic"),
            }
        for key, value in (data.get("stats") or {}).items():
            if key in self.stats and isinstance(value, (int, float)):
                self.stats[key] = value
        self.tick = max(0, int(data.get("tick") or 0))
        self.conversation_no = max(0, int(data.get("conversation_no") or 0))
        groups: dict[str, dict] = {}
        for gid, raw in (data.get("groups") or {}).items():
            if not isinstance(raw, dict):
                continue
            members: list[int] = []
            for member in raw.get("members") or []:
                try:
                    mid = int(member)
                except (TypeError, ValueError):
                    continue
                if mid in self.by_id:
                    members.append(mid)
            members = sorted(set(members))
            if len(members) < 2:
                continue
            try:
                cohesion = float(raw.get("cohesion") or 0)
            except (TypeError, ValueError):
                cohesion = 0.0
            groups[str(gid)] = {
                "id": str(gid),
                "members": members,
                "created_tick": int(raw.get("created_tick") or 0),
                "cohesion": max(0.0, min(1.0, cohesion)),
                "last_active_tick": int(raw.get("last_active_tick") or 0),
                "conversations": int(raw.get("conversations") or 0),
            }
        self.groups = groups
        numbers = [int(gid[1:]) for gid in groups
                   if gid.startswith("g") and gid[1:].isdigit()]
        self.group_no = max(numbers, default=len(groups))

    def _group_state(self) -> dict:
        return {
            gid: {
                "members": list(group["members"]),
                "created_tick": group["created_tick"],
                "cohesion": round(group["cohesion"], 4),
                "last_active_tick": group["last_active_tick"],
                "conversations": group["conversations"],
            }
            for gid, group in sorted(self.groups.items())
        }

    def write_state(self) -> None:
        state = {
            "phase": "A+B+C+D+E+F",
            "seed": self.seed,
            "tick": self.tick,
            "conversation_no": self.conversation_no,
            "positions": {str(aid): [round(x, 6), round(y, 6)]
                          for aid, (x, y) in sorted(self.positions.items())},
            "recent_topics": list(self.recent_topics),
            "affinity": {f"{a}:{b}": round(v, 4)
                         for (a, b), v in sorted(self.affinity.items())},
            "groups": self._group_state(),
            "opinions": {
                str(aid): {tid: round(value, 4)
                           for tid, value in sorted(opinions.items())}
                for aid, opinions in sorted(self.opinions.items())
                if opinions
            },
            "memory": {
                f"{a}:{b}": {
                    "topics": list(entry["topics"]),
                    "last_tick": entry["last_tick"],
                    "conversations": entry["conversations"],
                    "refusals": entry.get("refusals", 0),
                    "last_delta": round(entry["last_delta"], 4),
                    "last_topic": entry["last_topic"],
                }
                for (a, b), entry in sorted(self.memory.items())
            },
            "stats": dict(self.stats),
        }
        (self.out_dir / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ----- affinité

    @staticmethod
    def _pair(aid: int, bid: int) -> tuple[int, int]:
        return (aid, bid) if aid <= bid else (bid, aid)

    def affinity_between(self, aid: int, bid: int) -> float:
        """Affinité symétrique entre deux agents (0..1)."""
        return self.affinity.get(self._pair(aid, bid), DEFAULT_AFFINITY)

    def set_affinity(self, aid: int, bid: int, value: float) -> float:
        """Fixe et renvoie l'affinité d'une paire, bornée à 0..1."""
        value = max(0.0, min(1.0, float(value)))
        self.affinity[self._pair(aid, bid)] = value
        return value

    def adjust_affinity(self, aid: int, bid: int, delta: float) -> float:
        return self.set_affinity(aid, bid, self.affinity_between(aid, bid) + delta)

    @staticmethod
    def _affinity_factor(affinity: float) -> float:
        """Multiplicateur de proximité : faible affinité reste possible."""
        return 0.25 + 1.5 * affinity

    def refusal_probability(self, aid: int, bid: int) -> float:
        """Probabilité de refus, augmentée quand l'affinité est basse.

        ``refusal_rate <= 0`` désactive totalement le refus.
        """
        if self.refusal_rate <= 0:
            return 0.0
        affinity = self.affinity_between(aid, bid)
        probability = self.refusal_rate + (DEFAULT_AFFINITY - affinity) * 0.4
        entry = self.memory.get(self._pair(aid, bid))
        if entry and entry.get("last_delta", 0.0) < 0:
            probability += min(0.05, -entry["last_delta"] * 0.5)
        return max(0.0, min(1.0, probability))

    def _affinity_delta(self, topic: dict, transcript: list[dict],
                        turns: int, errors: list[dict]) -> float:
        """Variation d'affinité après une conversation.

        Heuristique déterministe : une conversation complétée et cordiale
        rapproche, une conversation interrompue ou conflictuelle éloigne.
        """
        if errors:
            return -0.03
        if not transcript:
            return -0.02
        delta = 0.02
        delta += 0.02 if len(transcript) >= turns else -0.02
        if topic.get("sensitive"):
            delta -= 0.005
        text = " ".join(str(t.get("text", "")).lower() for t in transcript)
        positive = ("d'accord", "je te rejoins", "tu as raison", "intéressant",
                    "je comprends", "bonne idée", "tout à fait")
        negative = ("pas d'accord", "je ne suis pas", "au contraire",
                    "désaccord", "je ne crois pas", "pas sûr", "autrement",
                    "faux")
        score = sum(text.count(word) for word in positive)
        score -= sum(text.count(word) for word in negative)
        delta += max(-0.03, min(0.03, score * 0.005))
        return max(-0.08, min(0.08, delta))

    # ----- opinions et mémoire sociale

    def _initial_opinion(self, aid: int, topic_id: str) -> float:
        """Opinion initiale stable, propre au couple agent/sujet."""
        digest = hashlib.sha256(
            f"{self.seed}:{aid}:{topic_id}".encode("utf-8")).digest()
        return 0.15 + (digest[0] / 255.0) * 0.7

    def opinion_of(self, aid: int, topic_id: str) -> float:
        """Opinion d'un agent sur un sujet (0 opposé, 1 favorable)."""
        stored = self.opinions.get(aid, {}).get(topic_id)
        return stored if stored is not None else self._initial_opinion(
            aid, topic_id)

    def set_opinion(self, aid: int, topic_id: str, value: float) -> float:
        score = max(0.0, min(1.0, float(value)))
        self.opinions.setdefault(aid, {})[topic_id] = score
        return score

    def _opinion_spread(self, people: list[dict],
                        topic_id: str) -> tuple[float, float]:
        values = [self.opinion_of(a["id"], topic_id) for a in people]
        if not values:
            return 0.0, DEFAULT_OPINION
        return max(values) - min(values), sum(values) / len(values)

    def _update_opinions(self, people: list[dict], topic: dict,
                         affinity: float) -> tuple[dict, dict]:
        """Rapproche les opinions du sujet, sans retournement brutal."""
        tid = topic["id"]
        before = {a["id"]: self.opinion_of(a["id"], tid) for a in people}
        if len(people) < 2:
            return before, dict(before)
        # Fige la position initiale des participants pour la rendre visible et
        # persistante, même si l'influence est nulle.
        for aid, value in before.items():
            self.opinions.setdefault(aid, {})[tid] = value
        if self.opinion_influence <= 0:
            return before, dict(before)
        _, mean = self._opinion_spread(people, tid)
        confidence = 0.5 + 0.5 * max(0.0, min(1.0, affinity))
        for agent in people:
            aid = agent["id"]
            old = before[aid]
            self.set_opinion(
                aid, tid, old + self.opinion_influence * confidence * (mean - old))
        after = {a["id"]: self.opinion_of(a["id"], tid) for a in people}
        return before, after

    def _pair_memory(self, aid: int, bid: int) -> dict:
        return self.memory.setdefault(self._pair(aid, bid), {
            "topics": [], "last_tick": 0, "conversations": 0,
            "refusals": 0, "last_delta": 0.0, "last_topic": None,
        })

    def _remember_conversation(self, people: list[dict], topic: dict,
                               delta: float) -> None:
        for index, left in enumerate(people):
            for right in people[index + 1:]:
                entry = self._pair_memory(left["id"], right["id"])
                tid = topic["id"]
                if tid in entry["topics"]:
                    entry["topics"].remove(tid)
                entry["topics"].append(tid)
                del entry["topics"][:-self.memory_size]
                entry["last_tick"] = self.tick
                entry["conversations"] += 1
                entry["last_delta"] = delta
                entry["last_topic"] = tid

    def _remember_refusal(self, aid: int, bid: int) -> None:
        entry = self._pair_memory(aid, bid)
        entry["last_tick"] = self.tick
        entry["refusals"] += 1
        entry["last_delta"] = -0.02

    def _memory_note(self, people: list[dict], topic_id: str) -> str:
        """Note courte injectée dans le prompt pour éviter les redites."""
        lines: list[str] = []
        seen: set[str] = set()
        for index, left in enumerate(people):
            for right in people[index + 1:]:
                entry = self.memory.get(self._pair(left["id"], right["id"]))
                if not entry or not entry["topics"]:
                    continue
                names = f"{left['prenom']} et {right['prenom']}"
                if topic_id in entry["topics"]:
                    line = (f"Vous avez déjà parlé de ce sujet avec "
                            f"{names} ({entry['conversations']} fois) : "
                            "rebondis dessus ou apporte un angle nouveau.")
                else:
                    known = ", ".join(self.topic_text.get(t, t)
                                      for t in entry["topics"][-2:])
                    line = f"Sujets déjà abordés avec {names} : {known}."
                if line not in seen:
                    seen.add(line)
                    lines.append(line)
        return ("\n\nMémoire : " + " ".join(lines)) if lines else ""

    def _topic_weights(self, participants: list[dict],
                       available: list[dict]) -> list[float]:
        weights = []
        for topic in available:
            weight = 1.0
            for index, left in enumerate(participants):
                for right in participants[index + 1:]:
                    entry = self.memory.get(self._pair(left["id"], right["id"]))
                    if entry and topic["id"] in entry["topics"]:
                        weight *= self.repeat_penalty
            weights.append(max(1e-9, weight))
        return weights

    # ----- espace et sélection

    def model_for(self, agent: dict) -> str:
        """Modèle utilisé pour cet agent (surcharge possible par agent)."""
        return self.agent_models.get(agent["id"], self.model)

    def _move_agents(self) -> None:
        if self.move_sigma <= 0:
            return
        for aid, (x, y) in list(self.positions.items()):
            self.positions[aid] = (
                _clip01(x + self.spatial_rng.gauss(0.0, self.move_sigma)),
                _clip01(y + self.spatial_rng.gauss(0.0, self.move_sigma)),
            )

    def select_topic(self, participants: list[dict] | None = None) -> dict:
        available = [t for t in self.topics if t["id"] not in self.recent_topics]
        if not available:
            self.recent_topics.clear()
            available = list(self.topics)
        if participants:
            topic = self.rng.choices(
                available, weights=self._topic_weights(participants, available),
                k=1)[0]
        else:
            topic = self.rng.choice(available)
        self.recent_topics.append(topic["id"])
        return dict(topic)

    def _resolve_topic(self, topic: dict | str | None,
                       participants: list[dict] | None = None) -> dict:
        if topic is None:
            return self.select_topic(participants)
        if isinstance(topic, dict):
            return {
                "id": str(topic.get("id") or "custom"),
                "text": str(topic["text"]),
                "category": str(topic.get("category") or "divers"),
                "tone": str(topic.get("tone") or "neutre"),
                "sensitive": bool(topic.get("sensitive", False)),
            }
        text = str(topic).strip()
        for t in self.topics:
            if t["id"] == text or t["text"].lower() == text.lower():
                self.recent_topics.append(t["id"])
                return dict(t)
        tid = "custom-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        self.recent_topics.append(tid)
        return {"id": tid, "text": text, "category": "libre",
                "tone": "libre", "sensitive": False}

    def _partner_weights(self, initiator_id: int) -> tuple[list[int], list[float]]:
        others = [a["id"] for a in self.agents if a["id"] != initiator_id]
        here = self.positions[initiator_id]
        weights = [
            max(1e-12, math.exp(-distance(here, self.positions[aid]) /
                                self.distance_scale) *
                self._affinity_factor(self.affinity_between(initiator_id, aid)))
            for aid in others
        ]
        return others, weights

    def select_partner(self, initiator_id: int) -> int:
        others, weights = self._partner_weights(initiator_id)
        return self.rng.choices(others, weights=weights, k=1)[0]

    def select_participants(self) -> tuple[dict, dict, float]:
        initiator = self.rng.choice(self.agents)
        partner_id = self.select_partner(initiator["id"])
        partner = self.by_id[partner_id]
        return (initiator, partner,
                distance(self.positions[initiator["id"]], self.positions[partner_id]))

    # ----- conversation

    def _mock_reply(self, speaker: dict, topic: dict, transcript: list[dict],
                    turn: int) -> str:
        """Réponse factice mais interactive : elle réagit à la réplique précédente."""
        rng = random.Random(f"{self.seed}:{self.conversation_no}:{turn}:{speaker['id']}")
        if not transcript:
            templates = [
                "Sur {topic}, je trouve qu'on en parle souvent sans entrer dans les détails.",
                "Franchement, {topic}, ça dépend beaucoup de la situation de chacun.",
                "Je n'ai pas forcément le même avis que toi sur {topic}, mais ça m'intéresse d'en parler.",
                "Pour {topic}, je pense qu'il faut surtout écouter les personnes concernées.",
                "C'est un sujet qui revient souvent dans les discussions : {topic}.",
            ]
            return f"[mock] {speaker['prenom']} : " + rng.choice(templates).format(
                topic=topic["text"])
        previous = transcript[-1]
        prev_name = previous["name"].split()[0]
        prev_text = previous["text"]
        if prev_text.startswith("[mock]") and " : " in prev_text:
            prev_text = prev_text.split(" : ", 1)[1]
        quote = prev_text.strip().replace("\n", " ")[:100]
        templates = [
            "{prev_name}, je te rejoins sur « {quote} ». Sur {topic}, j'ajouterais "
            "quand même une nuance.",
            "Je ne suis pas sûr de te suivre, {prev_name}, quand tu dis « {quote} ». "
            "Pour {topic}, je vois les choses autrement.",
            "D'accord avec toi sur « {quote} », mais {topic} mérite qu'on regarde "
            "les détails du quotidien.",
            "Tu as raison de pointer « {quote} ». Sur {topic}, je nuancerais "
            "quand même un peu.",
            "Je comprends ton point de vue, {prev_name}. Sur {topic}, je ne "
            "trancherais pas aussi vite.",
        ]
        text = rng.choice(templates).format(
            prev_name=prev_name, quote=quote, topic=topic["text"])
        return f"[mock] {speaker['prenom']} : {text}"

    def _session_agent(self, agent: dict, topic: dict,
                       participants: list[dict],
                       memory_note: str = "") -> dict:
        """Copie de l'agent avec les règles de conversation dans son persona."""
        others = ", ".join(a["prenom"] for a in participants
                           if a["id"] != agent["id"])
        persona = agent.get("persona", "") + (
            "\n\nRègles de conversation : tu discutes avec " + others +
            " sur le sujet « " + topic["text"] + " ». "
            "Réponds directement à la dernière réplique de ton interlocuteur, "
            "en 1 à 3 phrases, à la première personne. Ne répète pas ce qu'il "
            "vient de dire : fais avancer la discussion. Ne dis jamais que tu "
            "es une IA, un modèle de langage ou un agent synthétique."
        ) + memory_note
        session_agent = dict(agent)
        session_agent["persona"] = persona
        return session_agent

    def _opening_prompt(self, speaker: dict, topic: dict,
                       participants: list[dict],
                       memory_note: str = "") -> str:
        others = ", ".join(a["prenom"] for a in participants
                           if a["id"] != speaker["id"])
        return (
            f"Tu ouvres une conversation spontanée avec {others} sur le sujet : "
            f"{topic['text']}. Présente ton point de vue en 1 à 3 phrases, à la "
            "première personne, dans un français naturel. Ne dis pas que tu es "
            "une IA." + memory_note
        )

    def _resolve_agent(self, ref: dict | int | str | None) -> dict:
        if isinstance(ref, dict):
            return ref
        if isinstance(ref, str):
            try:
                ref = int(ref)
            except ValueError:
                return chat_agent.select_agent(self.agents, ref, self.seed)
        try:
            return self.by_id[int(ref)]
        except (KeyError, TypeError):
            raise SystemExit(f"Agent inconnu : {ref}") from None

    def _refusal_reason(self) -> str:
        """Raison courte et lisible d'un refus, tirée au hasard."""
        return self.rng.choice([
            "n'a pas envie de parler maintenant",
            "préfère rester seul pour l'instant",
            "est occupé et n'a pas le temps",
            "ne se sent pas concerné par le sujet",
            "n'a pas la tête à discuter",
        ])

    def _refusal_event(self, cid: str, initiator: dict, partner: dict,
                       dist: float, mode: str, reason: str,
                       affinity_before: float, affinity_after: float,
                       refusal_probability: float) -> dict:
        """Événement de refus, sans appel au modèle ni transcript."""
        return {
            "type": "refusal",
            "conversation_id": cid,
            "tick": self.tick,
            "mode": mode,
            "topic": None,
            "participants": [
                {"id": a["id"], "name": f"{a['prenom']} {a['nom']}",
                 "model": self.model_for(a)}
                for a in (initiator, partner)
            ],
            "refuser": {
                "id": partner["id"],
                "name": f"{partner['prenom']} {partner['nom']}",
                "model": self.model_for(partner),
            },
            "invited_by": {
                "id": initiator["id"],
                "name": f"{initiator['prenom']} {initiator['nom']}",
            },
            "distance": round(dist, 4),
            "affinity_before": round(affinity_before, 4),
            "affinity_after": round(affinity_after, 4),
            "affinity_delta": round(affinity_after - affinity_before, 4),
            "refusal_probability": round(refusal_probability, 4),
            "reason": reason,
            "transcript": [],
            "completed": False,
        }

    # ----- groupes

    def _mean_affinity(self, people: list[dict]) -> float:
        ids = [a["id"] for a in people]
        if len(ids) < 2:
            return DEFAULT_AFFINITY
        values = [self.affinity_between(ids[i], ids[j])
                  for i in range(len(ids)) for j in range(i + 1, len(ids))]
        return sum(values) / len(values)

    def _mean_distance(self, people: list[dict]) -> float:
        points = [self.positions[a["id"]] for a in people]
        if len(points) < 2:
            return 0.0
        values = [distance(points[i], points[j])
                  for i in range(len(points)) for j in range(i + 1, len(points))]
        return sum(values) / len(values)

    def _centroid(self, members: list[int]) -> tuple[float, float]:
        points = [self.positions[m] for m in members if m in self.positions]
        if not points:
            return (0.5, 0.5)
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))

    def _affinity_with(self, aid: int, members: list[int]) -> float:
        others = [m for m in members if m != aid and m in self.by_id]
        if not others:
            return DEFAULT_AFFINITY
        return sum(self.affinity_between(aid, m) for m in others) / len(others)

    def _group_cohesion(self, members: list[int]) -> float:
        """Cohésion 0..1 : 60 % affinité moyenne, 40 % proximité moyenne."""
        members = sorted({m for m in members if m in self.positions})
        if len(members) < 2:
            return 0.0
        people = [self.by_id[m] for m in members]
        affinity = self._mean_affinity(people)
        points = [self.positions[m] for m in members]
        values = [math.exp(-distance(points[i], points[j]) / self.group_radius)
                  for i in range(len(points)) for j in range(i + 1, len(points))]
        proximity = sum(values) / len(values)
        return max(0.0, min(1.0, 0.6 * affinity + 0.4 * proximity))

    def _update_groups(self) -> None:
        """Fait vivre les groupes : départs, dissolution, adhésions, création."""
        if self.max_groups <= 0 or self.group_max < 2:
            return

        # 1. Un membre trop faiblement lié aux autres quitte ; un groupe trop
        #    petit se dissout.
        for gid in sorted(self.groups):
            group = self.groups[gid]
            members = [m for m in group["members"] if m in self.positions]
            kept = [m for m in members
                    if self._affinity_with(m, members)
                    >= self.group_affinity - 0.15]
            if len(kept) < self.group_min:
                del self.groups[gid]
                continue
            group["members"] = kept
            group["cohesion"] = self._group_cohesion(kept)

        grouped = {m for g in self.groups.values() for m in g["members"]}

        # 2. Adhésion d'agents libres proches du centre et bien liés au groupe.
        for gid in sorted(self.groups):
            group = self.groups[gid]
            if len(group["members"]) >= self.group_max:
                continue
            members = list(group["members"])
            centroid = self._centroid(members)
            free = [a for a in self.agents if a["id"] not in grouped]
            free.sort(key=lambda a: distance(self.positions[a["id"]], centroid))
            for agent in free:
                if len(members) >= self.group_max:
                    break
                aid = agent["id"]
                if distance(self.positions[aid], centroid) > self.group_radius:
                    continue
                if self._affinity_with(aid, members) < self.group_affinity:
                    continue
                members.append(aid)
                grouped.add(aid)
            if members != group["members"]:
                group["members"] = sorted(members)
                group["cohesion"] = self._group_cohesion(members)

        # 3. Création déterministe de nouveaux groupes autour d'un agent libre.
        if len(self.groups) >= self.max_groups:
            return
        free = sorted((a for a in self.agents if a["id"] not in grouped),
                      key=lambda a: a["id"])
        for seed in free:
            if len(self.groups) >= self.max_groups:
                break
            sid = seed["id"]
            if sid in grouped:
                continue
            members = [sid]
            for other in free:
                if len(members) >= self.group_max:
                    break
                oid = other["id"]
                if oid == sid or oid in grouped:
                    continue
                if distance(self.positions[sid], self.positions[oid]) > self.group_radius:
                    continue
                if self.affinity_between(sid, oid) < self.group_affinity:
                    continue
                if self._affinity_with(oid, members) < self.group_affinity:
                    continue
                members.append(oid)
            if len(members) < self.group_min:
                continue
            self.group_no += 1
            gid = f"g{self.group_no:03d}"
            self.groups[gid] = {
                "id": gid,
                "members": sorted(members),
                "created_tick": self.tick,
                "cohesion": self._group_cohesion(members),
                "last_active_tick": self.tick,
                "conversations": 0,
            }
            grouped.update(members)

    def _pick_group(self) -> dict | None:
        groups = [g for g in self.groups.values() if len(g["members"]) >= 2]
        if not groups:
            return None
        weights = [max(1e-9, len(g["members"]) * (0.25 + g["cohesion"]))
                   for g in groups]
        return self.rng.choices(groups, weights=weights, k=1)[0]

    def run_group_conversation(self, group: dict | None = None, topic=None,
                               turns: int | None = None) -> dict | None:
        """Conversation d'un groupe existant (ou tiré au sort)."""
        if group is None:
            group = self._pick_group()
            if group is None:
                return None
        members = [self.by_id[m] for m in group["members"] if m in self.by_id]
        if len(members) < 2:
            return None
        if turns is None:
            turns = max(len(members),
                        self.rng.randint(self.min_turns, self.max_turns))
        return self.run_conversation(participants=members, topic=topic,
                                     turns=turns, mode="group", group=group)

    def run_conversation(self, initiator=None, partner=None, topic=None,
                         turns: int | None = None, mode: str = "random",
                         participants=None, group: dict | None = None) -> dict:
        """Lance une conversation et renvoie son événement.

        ``initiator`` et ``partner`` acceptent un agent (dict), un id, une clé
        d'identité, un nom « Prénom Nom », ou None pour un tirage aléatoire.
        ``participants`` permet une conversation de groupe (2+ agents).
        ``topic`` peut être un dict, un id, un texte libre, ou None pour un
        tirage dans la liste. ``mode`` vaut ``random``, ``manual`` ou ``group``.
        """
        if participants is not None:
            people: list[dict] = []
            seen: set[int] = set()
            for ref in participants:
                agent = self._resolve_agent(ref)
                if agent["id"] not in seen:
                    people.append(agent)
                    seen.add(agent["id"])
            if len(people) < 2:
                raise SystemExit("Il faut au moins deux agents distincts.")
            initiator, partner = people[0], people[1]
            dist = self._mean_distance(people)
            affinity_before = self._mean_affinity(people)
            is_group = group is not None or len(people) > 2
            if is_group and mode == "manual":
                mode = "group"
            refusal_probability = 0.0
        else:
            if initiator is None and partner is None:
                initiator, partner, dist = self.select_participants()
            else:
                if initiator is None:
                    initiator = self.rng.choice(self.agents)
                initiator = self._resolve_agent(initiator)
                if partner is None:
                    partner = self.by_id[self.select_partner(initiator["id"])]
                partner = self._resolve_agent(partner)
                dist = distance(self.positions[initiator["id"]],
                                self.positions[partner["id"]])
            if initiator["id"] == partner["id"]:
                raise SystemExit("Un agent ne peut pas discuter avec lui-même.")
            people = [initiator, partner]
            affinity_before = self.affinity_between(initiator["id"],
                                                    partner["id"])
            is_group = False
            refusal_probability = 0.0

        job = self._prepare_conversation(
            initiator=initiator, partner=partner, topic=topic, turns=turns,
            mode=mode, participants=participants, group=group)
        if job.get("refused"):
            return job["event"]
        return self._finish_conversation(job, self._generate_conversation(job))

    def _prepare_conversation(self, *, initiator=None, partner=None,
                              topic=None, turns=None, mode: str = "random",
                              participants=None, group: dict | None = None,
                              ) -> dict:
        """Résout les participants, le sujet et le refus, sans appeler le LLM.

        La génération des répliques est faite séparément par
        ``_generate_conversation``, ce qui permet de lancer plusieurs
        conversations en parallèle avant d'appliquer leurs effets.
        """
        if participants is not None:
            people: list[dict] = []
            seen: set[int] = set()
            for ref in participants:
                agent = self._resolve_agent(ref)
                if agent["id"] not in seen:
                    people.append(agent)
                    seen.add(agent["id"])
            if len(people) < 2:
                raise SystemExit("Il faut au moins deux agents distincts.")
            initiator, partner = people[0], people[1]
            dist = self._mean_distance(people)
            affinity_before = self._mean_affinity(people)
            is_group = group is not None or len(people) > 2
            if is_group and mode == "manual":
                mode = "group"
            refusal_probability = 0.0
        else:
            if initiator is None and partner is None:
                initiator, partner, dist = self.select_participants()
            else:
                if initiator is None:
                    initiator = self.rng.choice(self.agents)
                initiator = self._resolve_agent(initiator)
                if partner is None:
                    partner = self.by_id[self.select_partner(initiator["id"])]
                partner = self._resolve_agent(partner)
                dist = distance(self.positions[initiator["id"]],
                                self.positions[partner["id"]])
            if initiator["id"] == partner["id"]:
                raise SystemExit("Un agent ne peut pas discuter avec lui-même.")
            people = [initiator, partner]
            affinity_before = self.affinity_between(initiator["id"],
                                                    partner["id"])
            is_group = False
            refusal_probability = 0.0

        self.conversation_no += 1
        number = self.conversation_no
        cid = f"c{number:04d}"

        if not is_group:
            # Refus possible même quand la paire est imposée (--talk, --spark-rate 1).
            refusal_probability = self.refusal_probability(
                initiator["id"], partner["id"])
            if (refusal_probability > 0
                    and self.rng.random() < refusal_probability):
                reason = self._refusal_reason()
                affinity_after = self.adjust_affinity(
                    initiator["id"], partner["id"], -0.02)
                event = self._refusal_event(
                    cid, initiator, partner, dist, mode, reason,
                    affinity_before, affinity_after, refusal_probability)
                self._log_event(event)
                self.stats["refusals"] += 1
                self._remember_refusal(initiator["id"], partner["id"])
                if self.verbose:
                    print(f"\n[tick {self.tick}] {cid} · "
                          f"{initiator['prenom']} & {partner['prenom']} · "
                          f"distance {dist:.3f} · "
                          f"affinité {affinity_before:.2f}")
                    print(f"  {partner['prenom']}> refuse de discuter avec "
                          f"{initiator['prenom']} ({reason}) · "
                          f"affinité {affinity_before:.2f} → "
                          f"{affinity_after:.2f}")
                return {"refused": True, "event": event}

        topic = self._resolve_topic(topic, people)
        memory_note = self._memory_note(people, topic["id"])
        if turns is None:
            turns = self.rng.randint(self.min_turns, self.max_turns)
        else:
            turns = max(1, min(self.max_turns, int(turns)))

        order = list(people)
        if is_group:
            self.rng.shuffle(order)
        elif self.rng.random() < 0.5:
            order.reverse()

        sessions = {
            a["id"]: {
                "agent": self._session_agent(a, topic, order, memory_note),
                "model": self.model_for(a),
                "history": [],
                "session_id": f"{cid}-a{a['id']}",
            }
            for a in order
        }
        # Chaque agent a sa propre session. Le premier à parler reçoit le prompt
        # d'ouverture ; les autres reçoivent les répliques comme messages.
        first = order[0]
        sessions[first["id"]]["history"].append({
            "role": "user",
            "content": self._opening_prompt(first, topic, order, memory_note),
        })
        if self.verbose:
            if is_group:
                label = (f"groupe {group['id']} ({len(order)}) · "
                         f"{', '.join(a['prenom'] for a in order)}")
            else:
                label = f"{initiator['prenom']} & {partner['prenom']}"
            print(f"\n[tick {self.tick}] {cid} · {label} · "
                  f"sujet : {topic['text']} [{topic['category']}] · "
                  f"distance {dist:.3f} · affinité {affinity_before:.2f}")
        return {
            "refused": False,
            "cid": cid,
            "number": number,
            "people": people,
            "order": order,
            "sessions": sessions,
            "topic": topic,
            "turns": turns,
            "dist": dist,
            "affinity_before": affinity_before,
            "is_group": is_group,
            "mode": mode,
            "group": group,
            "initiator": initiator,
            "partner": partner,
            "quiet": False,
        }

    def _generate_conversation(self, job: dict) -> dict:
        """Joue les tours d'une conversation préparée, sans effet de bord.

        Cette méthode est appelée dans un thread de travail quand plusieurs
        conversations tournent en parallèle : elle ne touche ni aux compteurs,
        ni aux affinités, ni aux fichiers.
        """
        order = job["order"]
        topic = job["topic"]
        sessions = job["sessions"]
        turns = job["turns"]
        quiet = job.get("quiet", False)
        verbose = self.verbose and not quiet
        transcript: list[dict] = []
        errors: list[dict] = []
        for turn in range(turns):
            speaker = order[turn % len(order)]
            session = sessions[speaker["id"]]
            model = session["model"]
            window = self.max_context_turns * 2
            sent = chat_agent.trim_history(
                session["history"][-window:], self.budget_tokens,
                system_tokens=chat_agent.estimate_tokens(
                    session["agent"]["persona"]))
            try:
                if self.provider == "mock":
                    text = self._mock_reply(speaker, topic, transcript, turn)
                    if verbose:
                        print(f"  {speaker['prenom']}> {text}")
                else:
                    do_stream = self.stream and verbose
                    if do_stream:
                        print(f"  {speaker['prenom']}> ", end="", flush=True)
                    text = chat_agent.respond(
                        session["agent"], sent, self.provider, model,
                        temperature=self.temperature, max_tokens=self.max_tokens,
                        seed=self.seed + job["number"] * 1000 + turn,
                        stream=do_stream,
                        on_token=(lambda piece: print(piece, end="", flush=True))
                        if do_stream else None)
                    if do_stream:
                        print()
                    elif verbose:
                        print(f"  {speaker['prenom']}> {text}")
            except chat_agent.LLMError as exc:
                if verbose and self.provider != "mock" and self.stream:
                    print()
                errors.append({"turn": turn + 1, "speaker": speaker["id"],
                               "error": str(exc)})
                if verbose:
                    print(f"  erreur : {exc}", file=sys.stderr)
                break
            text = (text or "").strip()
            if not text:
                text = "(silence)"
            transcript.append({"speaker": speaker["id"],
                               "name": f"{speaker['prenom']} {speaker['nom']}",
                               "model": model,
                               "session": session["session_id"],
                               "text": text})
            # Met à jour toutes les sessions : l'orateur garde sa réplique en
            # assistant, les autres la reçoivent en user.
            for aid, other in sessions.items():
                if aid == speaker["id"]:
                    other["history"].append({"role": "assistant", "content": text})
                else:
                    other["history"].append({
                        "role": "user",
                        "content": f"{speaker['prenom']} : {text}",
                    })
        return {"transcript": transcript, "errors": errors}

    def _finish_conversation(self, job: dict, generation: dict) -> dict:
        """Applique les effets d'une conversation et journalise son événement."""
        people = job["people"]
        topic = job["topic"]
        group = job["group"]
        transcript = generation["transcript"]
        errors = generation["errors"]
        turns = job["turns"]
        dist = job["dist"]
        affinity_before = job["affinity_before"]
        is_group = job["is_group"]
        initiator, partner = job["initiator"], job["partner"]
        cid = job["cid"]
        quiet = job.get("quiet", False)
        verbose = self.verbose and not quiet

        affinity_delta = self._affinity_delta(topic, transcript, turns, errors)
        spread, _ = self._opinion_spread(people, topic["id"])
        agreement = 1.0 - spread
        affinity_delta = max(-0.08, min(0.08,
                                        affinity_delta + (agreement - 0.5) * 0.03))
        opinions_before, opinions_after = self._update_opinions(
            people, topic, affinity_before)
        self._remember_conversation(people, topic, affinity_delta)
        cohesion_after = None
        if is_group:
            for index, left in enumerate(people):
                for right in people[index + 1:]:
                    self.adjust_affinity(left["id"], right["id"], affinity_delta)
            affinity_after = self._mean_affinity(people)
            if group is not None:
                group["last_active_tick"] = self.tick
                group["conversations"] = group.get("conversations", 0) + 1
                group["cohesion"] = self._group_cohesion(group["members"])
                cohesion_after = group["cohesion"]
            else:
                cohesion_after = affinity_after
        else:
            affinity_after = self.adjust_affinity(
                initiator["id"], partner["id"], affinity_delta)
        if verbose:
            suffix = (f" · cohésion {cohesion_after:.2f}"
                      if cohesion_after is not None else "")
            print(f"  ── fin · {len(transcript)}/{turns} tour(s) · "
                  f"sujet : {topic['text']} · "
                  f"affinité {affinity_before:.2f} → {affinity_after:.2f} "
                  f"({affinity_after - affinity_before:+.2f}){suffix}")

        event = {
            "type": "conversation",
            "conversation_id": cid,
            "tick": self.tick,
            "mode": job["mode"],
            "topic": topic,
            "participants": [
                {"id": a["id"], "name": f"{a['prenom']} {a['nom']}",
                 "model": self.model_for(a),
                 "session": job["sessions"][a["id"]]["session_id"]}
                for a in people
            ],
            "sessions": {
                str(aid): {"id": s["session_id"], "model": s["model"],
                           "messages": len(s["history"])}
                for aid, s in sorted(job["sessions"].items())
            },
            "group_id": group["id"] if group is not None else None,
            "group": {
                "id": group["id"],
                "members": list(group["members"]),
                "size": len(people),
                "cohesion": round(cohesion_after, 4),
            } if group is not None else None,
            "distance": round(dist, 4),
            "affinity_before": round(affinity_before, 4),
            "affinity_after": round(affinity_after, 4),
            "affinity_delta": round(affinity_after - affinity_before, 4),
            "agreement": round(agreement, 4),
            "opinions_before": {str(aid): round(v, 4)
                                for aid, v in sorted(opinions_before.items())},
            "opinions_after": {str(aid): round(v, 4)
                               for aid, v in sorted(opinions_after.items())},
            "turns_requested": turns,
            "transcript": transcript,
            "completed": len(transcript) == turns and not errors,
            "errors": errors,
        }
        self._write_transcript(cid, event)
        self._log_event(event)
        self.stats["conversations"] += 1
        self.stats["turns"] += len(transcript)
        self.stats["errors"] += len(errors)
        if is_group:
            self.stats["group_conversations"] += 1
        return event

    def _prepare_group_job(self) -> dict | None:
        group = self._pick_group()
        if group is None:
            return None
        members = [self.by_id[m] for m in group["members"] if m in self.by_id]
        if len(members) < 2:
            return None
        turns = max(len(members),
                    self.rng.randint(self.min_turns, self.max_turns))
        return self._prepare_conversation(
            participants=members, turns=turns, mode="group", group=group)

    def _generate_batch(self, jobs: list[dict]) -> list[dict]:
        workers = min(max(1, self.parallel), len(jobs))
        if workers <= 1:
            return [self._generate_conversation(job) for job in jobs]
        # En parallèle, les répliques ne sont pas affichées : les flux
        # s'entrelaceraient. Les événements restent complets.
        for job in jobs:
            job["quiet"] = True
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._generate_conversation, jobs))

    def step(self) -> dict | None:
        self.tick += 1
        self._move_agents()
        self._update_groups()
        if len(self.agents) < 2:
            return None
        events: list[dict] = []
        jobs: list[dict] = []
        group_done = False
        for _ in range(self.conversations_per_tick):
            if self.rng.random() >= self.spark_rate:
                continue
            if (not group_done and self.groups and self.group_rate > 0
                    and self.rng.random() < self.group_rate):
                job = self._prepare_group_job()
                if job is not None:
                    group_done = True
                    if job.get("refused"):
                        events.append(job["event"])
                    else:
                        jobs.append(job)
                    continue
            job = self._prepare_conversation()
            if job.get("refused"):
                events.append(job["event"])
            else:
                jobs.append(job)
        if jobs:
            generations = self._generate_batch(jobs)
            for job, generation in zip(jobs, generations):
                events.append(self._finish_conversation(job, generation))
        return events[-1] if events else None

    def run(self, max_ticks: int | None = None) -> dict:
        ticks = self.max_ticks if max_ticks is None else max(0, int(max_ticks))
        for index in range(ticks):
            self.step()
            if self.state_every and (index + 1) % self.state_every == 0:
                self.write_state()
            if self.tick_seconds:
                time.sleep(self.tick_seconds)
        self.write_state()
        return dict(self.stats)


# ---------------------------------------------------------------------- CLI

def _default_model(provider: str, requested: str | None) -> str:
    if requested:
        return requested
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("OLLAMA_MODEL")
    if model:
        return model
    return {"openai": "gpt-4o-mini", "ollama": "llama3.2:1b"}.get(provider, "mock")


def _sample_agents(agents: list[dict], limit: int | None, seed: int) -> list[dict]:
    if not limit or limit >= len(agents):
        return agents
    chosen = random.Random(seed).sample(agents, limit)
    return sorted(chosen, key=lambda a: a["id"])


def _parse_agent_models(values: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--agent-model attendu sous la forme ID=MODEL, reçu {item!r}")
        key, model = item.split("=", 1)
        try:
            out[int(key)] = model.strip()
        except ValueError:
            raise SystemExit(f"Identifiant d'agent invalide : {key!r}") from None
    return out


def _manual_agents(agents: list[dict], a_ref: str, b_ref: str,
                   seed: int) -> tuple[dict, dict]:
    """Résout deux agents pour une conversation manuelle.

    ``random`` est accepté pour l'un ou les deux. Deux tirages aléatoires sont
    distincts ; un tirage aléatoire exclut l'agent explicite déjà choisi.
    """
    rng = random.Random(f"{seed}:manual")
    a_random = a_ref.strip().lower() == "random"
    b_random = b_ref.strip().lower() == "random"
    if a_random and b_random:
        if len(agents) < 2:
            raise SystemExit("Il faut au moins deux agents.")
        a, b = rng.sample(agents, 2)
        return a, b
    if a_random:
        b = chat_agent.select_agent(agents, b_ref, seed)
        pool = [x for x in agents if x["id"] != b["id"]]
        return rng.choice(pool), b
    if b_random:
        a = chat_agent.select_agent(agents, a_ref, seed)
        pool = [x for x in agents if x["id"] != a["id"]]
        return a, rng.choice(pool)
    a = chat_agent.select_agent(agents, a_ref, seed)
    b = chat_agent.select_agent(agents, b_ref, seed)
    if a["id"] == b["id"]:
        raise SystemExit("Les deux agents doivent être différents.")
    return a, b


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=Path, default=DEFAULT_AGENTS)
    p.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="dossier de base des runs (défaut : social/runs)")
    p.add_argument("--run-id", default=None,
                   help="nom du sous-dossier ; défaut : run-<horodatage>-s<seed>")
    p.add_argument("--provider", choices=["mock", "openai", "ollama"], default="mock")
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ticks", type=int, default=50)
    p.add_argument("--spark-rate", type=float, default=0.2,
                   help="probabilité de conversation à chaque tick (0..1)")
    p.add_argument("--tick-seconds", type=float, default=1.0,
                   help="pause entre deux ticks (0 = aucune)")
    p.add_argument("--min-turns", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--distance-scale", type=float, default=0.15,
                   help="portée spatiale : plus grand = voisinage plus large")
    p.add_argument("--move-sigma", type=float, default=0.01,
                   help="amplitude du déplacement aléatoire par tick")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--budget-tokens", type=int, default=3000)
    p.add_argument("--max-context-turns", type=int, default=12)
    p.add_argument("--refusal-rate", type=float, default=0.1,
                   help="probabilité qu'un agent invité refuse de discuter (0..1)")
    p.add_argument("--resume", type=Path, default=None, metavar="PATH",
                   help="reprend un run existant (dossier de run ou state.json)")
    p.add_argument("--reset", action="store_true",
                   help="efface l'état et les transcripts du run avant de démarrer")
    p.add_argument("--group-min", type=int, default=3,
                   help="taille minimale d'un groupe (défaut : 3)")
    p.add_argument("--group-max", type=int, default=5,
                   help="taille maximale d'un groupe (défaut : 5)")
    p.add_argument("--group-affinity", type=float, default=0.5,
                   help="affinité minimale pour former/admettre dans un groupe")
    p.add_argument("--group-radius", type=float, default=0.25,
                   help="rayon spatial maximal autour du centre du groupe")
    p.add_argument("--group-rate", type=float, default=0.35,
                   help="probabilité qu'une conversation soit collective (0..1)")
    p.add_argument("--max-groups", type=int, default=20,
                   help="nombre maximal de groupes simultanés (0 = aucun)")
    p.add_argument("--memory-size", type=int, default=5,
                   help="sujets mémorisés par paire (défaut : 5)")
    p.add_argument("--opinion-influence", type=float, default=0.15,
                   help="influence des interlocuteurs sur les opinions (0..1)")
    p.add_argument("--repeat-penalty", type=float, default=0.7,
                   help="poids d'un sujet déjà abordé avec la même personne (0..1)")
    p.add_argument("--state-every", type=int, default=0, metavar="N",
                   help="écrit state.json tous les N ticks (0 = seulement à la fin)")
    p.add_argument("--conversations-per-tick", type=int, default=1, metavar="N",
                   help="nombre maximal de conversations par tick (défaut : 1)")
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="conversations générées en parallèle (défaut : 1)")
    p.add_argument("--talk", nargs=2, metavar=("AGENT_A", "AGENT_B"),
                   help="fait discuter deux agents puis quitte (id, clé, nom ou 'random')")
    topic_group = p.add_mutually_exclusive_group()
    topic_group.add_argument("--topic", default=None,
                             help="sujet manuel : id, texte exact ou texte libre")
    topic_group.add_argument("--random-topic", action="store_true",
                             help="tire un sujet au hasard (défaut en mode --talk)")
    p.add_argument("--turns", type=int, default=None,
                   help="nombre de tours en mode --talk (défaut : aléatoire)")
    p.add_argument("--no-stream", action="store_true",
                   help="désactive l'affichage des tokens en direct")
    p.add_argument("--agent-model", action="append", default=[], metavar="ID=MODEL",
                   help="modèle propre à un agent (répétable), ex. --agent-model 12=qwen2.5:3b")
    p.add_argument("--limit", type=int, default=None,
                   help="utiliser seulement N agents (échantillon seedé)")
    p.add_argument("--list-topics", action="store_true",
                   help="liste les sujets puis quitte")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    topics = load_topics(args.topics)
    if args.list_topics:
        print(f"{len(topics)} sujets :")
        for t in topics:
            flag = " [sensible]" if t["sensitive"] else ""
            print(f"  {t['id']:<28} {t['category']:<14} {t['text']}{flag}")
        return 0

    all_agents = load_agents(args.agents)
    model = _default_model(args.provider, args.model)
    agent_models = _parse_agent_models(args.agent_model)
    talk = args.talk is not None
    run_id = args.run_id or (
        f"manual-{time.strftime('%Y%m%d-%H%M%S')}-s{args.seed}" if talk
        else f"run-{time.strftime('%Y%m%d-%H%M%S')}-s{args.seed}")
    out_dir = args.out / run_id

    if args.reset and args.resume is not None:
        raise SystemExit("--reset et --resume sont incompatibles.")
    if args.reset and out_dir.exists():
        for name in ("events.jsonl", "state.json", "config.json"):
            (out_dir / name).unlink(missing_ok=True)
        shutil.rmtree(out_dir / "transcripts", ignore_errors=True)

    resume_data = None
    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = (args.resume / "state.json"
                       if args.resume.is_dir() else args.resume)
        if not resume_path.exists():
            raise SystemExit(f"État introuvable : {resume_path}")
        try:
            resume_data = json.loads(resume_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise SystemExit(f"État illisible : {resume_path} ({exc})") from None
        out_dir = resume_path.parent
        run_id = out_dir.name

    group_kwargs = dict(
        group_min=args.group_min, group_max=args.group_max,
        group_affinity=args.group_affinity, group_radius=args.group_radius,
        group_rate=args.group_rate, max_groups=args.max_groups,
        memory_size=args.memory_size,
        opinion_influence=args.opinion_influence,
        repeat_penalty=args.repeat_penalty,
        state_every=args.state_every,
        conversations_per_tick=args.conversations_per_tick,
        parallel=args.parallel,
        resume=resume_data,
        resume_source=str(resume_path) if resume_path is not None else None)

    if talk:
        first, second = _manual_agents(all_agents, *args.talk, args.seed)
        sim = Simulation(
            all_agents, topics, out_dir,
            seed=args.seed, provider=args.provider, model=model,
            spark_rate=args.spark_rate, tick_seconds=0,
            max_ticks=args.max_ticks, min_turns=args.min_turns,
            max_turns=args.max_turns, distance_scale=args.distance_scale,
            move_sigma=args.move_sigma, temperature=args.temperature,
            max_tokens=args.max_tokens, budget_tokens=args.budget_tokens,
            max_context_turns=args.max_context_turns, stream=not args.no_stream,
            refusal_rate=args.refusal_rate,
            agent_models=agent_models, verbose=not args.quiet, **group_kwargs)
        if resume_data is None:
            sim.tick = 1
        if not args.quiet:
            print(f"Conversation manuelle · {first['prenom']} {first['nom']} "
                  f"& {second['prenom']} {second['nom']} · seed {args.seed} · "
                  f"{args.provider}/{model}")
            if args.provider == "mock":
                print("Mode mock : réponses factices déterministes. Pour un vrai "
                      "modèle : --provider ollama --model llama3.2:3b")
            print(f"Run : {out_dir}")
        event = sim.run_conversation(
            initiator=first, partner=second, topic=args.topic,
            turns=args.turns, mode="manual")
        sim.write_state()
        if args.quiet:
            print(json.dumps(event, ensure_ascii=False))
        elif event["type"] == "refusal":
            print(f"\nrefus · {event['refuser']['name']} "
                  f"({event['reason']}) · distance {event['distance']}")
            print(f"  logs        : {out_dir / 'events.jsonl'}")
        else:
            print(f"\n{len(event['transcript'])} tour(s) · "
                  f"distance {event['distance']} · "
                  f"affinité {event['affinity_before']:.2f} → "
                  f"{event['affinity_after']:.2f} "
                  f"({event['affinity_delta']:+.2f}) · "
                  f"complété={event['completed']}")
            print(f"  logs        : {out_dir / 'events.jsonl'}")
            print(f"  transcript  : {out_dir / 'transcripts' / (event['conversation_id'] + '.json')}")
        return 0

    agents = _sample_agents(all_agents, args.limit, args.seed)
    if resume_data is not None:
        known = {int(k) for k in (resume_data.get("positions") or {})
                 if str(k).isdigit()}
        if known:
            agents = [a for a in all_agents if a["id"] in known]
    sim = Simulation(
        agents, topics, out_dir,
        seed=args.seed, provider=args.provider, model=model,
        spark_rate=args.spark_rate, tick_seconds=args.tick_seconds,
        max_ticks=args.max_ticks, min_turns=args.min_turns,
        max_turns=args.max_turns, distance_scale=args.distance_scale,
        move_sigma=args.move_sigma, temperature=args.temperature,
        max_tokens=args.max_tokens, budget_tokens=args.budget_tokens,
        max_context_turns=args.max_context_turns, stream=not args.no_stream,
        refusal_rate=args.refusal_rate,
        agent_models=agent_models, verbose=not args.quiet, **group_kwargs)
    if not args.quiet:
        print(f"{len(agents)} agents · {len(topics)} sujets · seed {args.seed} · "
              f"{args.provider}/{model}")
        if resume_data is not None:
            print(f"Reprise : {resume_path} · tick {sim.tick} · "
                  f"{len(sim.groups)} groupe(s)")
        if args.provider == "mock":
            print("Mode mock : réponses factices déterministes. Pour un vrai "
                  "modèle : --provider ollama --model llama3.2:3b")
        print(f"Run : {out_dir}")
    stats = sim.run()
    if not args.quiet:
        print(f"\n{stats['conversations']} conversation(s), "
              f"{stats['refusals']} refus, "
              f"{stats['group_conversations']} collective(s), "
              f"{stats['turns']} tour(s), "
              f"{stats['errors']} erreur(s)")
        print(f"  groupes     : {len(sim.groups)}")
        print(f"  logs        : {out_dir / 'events.jsonl'}")
        print(f"  transcripts : {out_dir / 'transcripts'}")
    else:
        print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
