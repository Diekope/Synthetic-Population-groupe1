"""Tests de la simulation sociale — Phase A.

Couvre : liste de sujets, positions 2D, choix du partenaire par proximité,
non-répétition des sujets, conversations mock, journaux et déterminisme.

Lancer :  uv run python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import chat_agent  # noqa: E402
import social_sim  # noqa: E402

AGENTS = ROOT / "agents" / "agents.jsonl"
TOPICS = ROOT / "tools" / "topics.fr.json"


class TopicListTest(unittest.TestCase):
    def test_topics_are_varied(self):
        topics = social_sim.load_topics(TOPICS)
        self.assertGreaterEqual(len(topics), 100)
        self.assertEqual(len({t["id"] for t in topics}), len(topics))
        self.assertGreaterEqual(len({t["category"] for t in topics}), 12)
        self.assertGreaterEqual(sum(1 for t in topics if t["sensitive"]), 5)
        for t in topics:
            self.assertTrue(t["text"])
            self.assertTrue(t["id"])

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "topics.json"
            path.write_text(json.dumps({"topics": [
                {"id": "a", "text": "sujet A"},
                {"id": "a", "text": "sujet B"},
            ]}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                social_sim.load_topics(path)

    def test_accepts_plain_list(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "topics.json"
            path.write_text(json.dumps(["premier sujet", "deuxième sujet"]),
                            encoding="utf-8")
            topics = social_sim.load_topics(path)
            self.assertEqual([t["text"] for t in topics],
                             ["premier sujet", "deuxième sujet"])


class SimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agents = chat_agent.load_agents(AGENTS)[:12]
        cls.topics = social_sim.load_topics(TOPICS)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name) / self._testMethodName

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, **overrides):
        params = dict(
            seed=0, provider="mock", model="mock", spark_rate=1.0,
            tick_seconds=0, max_ticks=3, min_turns=4, max_turns=6,
            verbose=False, refusal_rate=0.0, group_rate=0.0,
        )
        params.update(overrides)
        out_dir = params.pop("out_dir", self.base)
        agents = params.pop("agents", self.agents)
        return social_sim.Simulation(agents, self.topics, out_dir, **params)

    def test_positions_are_seeded_and_bounded(self):
        a = self.make()
        b = self.make(out_dir=self.base / "b")
        self.assertEqual(a.positions, b.positions)
        for x, y in a.positions.values():
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)

    def test_near_partner_is_preferred(self):
        sim = self.make(agents=self.agents[:3])
        ids = [a["id"] for a in self.agents[:3]]
        sim.positions[ids[0]] = (0.50, 0.50)
        sim.positions[ids[1]] = (0.51, 0.50)
        sim.positions[ids[2]] = (0.99, 0.99)
        draws = [sim.select_partner(ids[0]) for _ in range(200)]
        self.assertGreater(draws.count(ids[1]), draws.count(ids[2]))
        self.assertGreater(draws.count(ids[1]), 150)

    def test_recent_topics_do_not_repeat_before_cycle(self):
        sim = self.make()
        n = sim.recent_topics.maxlen
        ids = [sim.select_topic()["id"] for _ in range(n)]
        self.assertEqual(len(set(ids)), n)
        # Au-delà de la fenêtre, la liste se recycle sans erreur.
        again = sim.select_topic()
        self.assertIn(again["id"], {t["id"] for t in self.topics})

    def test_run_writes_logs_and_transcripts(self):
        sim = self.make(max_ticks=4)
        stats = sim.run()
        self.assertEqual(stats["conversations"], 4)
        self.assertEqual(stats["errors"], 0)

        events = [json.loads(line) for line in
                  sim.events_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(events), 4)
        for event in events:
            self.assertEqual(event["type"], "conversation")
            self.assertTrue(event["completed"])
            ids = [p["id"] for p in event["participants"]]
            self.assertEqual(len(ids), 2)
            self.assertNotEqual(ids[0], ids[1])
            self.assertTrue(4 <= len(event["transcript"]) <= 6)
            self.assertEqual({t["speaker"] for t in event["transcript"]}, set(ids))
            # Les prises de parole alternent.
            speakers = [t["speaker"] for t in event["transcript"]]
            for left, right in zip(speakers, speakers[1:]):
                self.assertNotEqual(left, right)
            self.assertTrue(event["transcript"][0]["text"].startswith("[mock]"))

        self.assertEqual(len(list(sim.transcripts_dir.glob("*.json"))), 4)
        self.assertTrue((sim.out_dir / "config.json").exists())
        self.assertTrue((sim.out_dir / "state.json").exists())

    def test_mock_is_deterministic(self):
        a = self.make(out_dir=self.base / "a")
        b = self.make(out_dir=self.base / "b")
        a.run()
        b.run()
        self.assertEqual(a.events_path.read_text(encoding="utf-8"),
                         b.events_path.read_text(encoding="utf-8"))
        self.assertEqual((a.out_dir / "state.json").read_text(encoding="utf-8"),
                         (b.out_dir / "state.json").read_text(encoding="utf-8"))

    def test_spark_rate_zero_makes_no_conversation(self):
        sim = self.make(spark_rate=0.0, max_ticks=5)
        stats = sim.run()
        self.assertEqual(stats["conversations"], 0)
        self.assertEqual(sim.events_path.read_text(encoding="utf-8"), "")

    def test_manual_conversation_with_chosen_topic(self):
        sim = self.make()
        first, second = self.agents[0], self.agents[1]
        event = sim.run_conversation(
            initiator=first, partner=second, topic="le télétravail", turns=2,
            mode="manual")
        self.assertEqual(event["topic"]["id"], "teletravail")
        self.assertEqual(event["mode"], "manual")
        self.assertEqual([p["id"] for p in event["participants"]],
                         [first["id"], second["id"]])
        self.assertEqual(len(event["transcript"]), 2)

    def _talk_cli(self, *extra):
        out = self.base / "cli"
        cmd = [
            sys.executable, str(ROOT / "tools" / "social_sim.py"),
            "--agents", str(AGENTS), "--topics", str(TOPICS),
            "--out", str(out), "--run-id", "manual",
            "--provider", "mock", "--tick-seconds", "0", "--quiet",
            "--refusal-rate", "0",
            "--talk", "1", "5", "--turns", "2",
            *extra,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        event = json.loads(result.stdout)
        self.assertEqual(event["mode"], "manual")
        self.assertEqual([p["id"] for p in event["participants"]], [1, 5])
        self.assertEqual(len(event["transcript"]), 2)
        return event, out / "manual"

    def test_manual_cli_random_topic(self):
        event, run = self._talk_cli("--random-topic")
        topic_ids = {t["id"] for t in self.topics}
        self.assertIn(event["topic"]["id"], topic_ids)
        self.assertTrue((run / "events.jsonl").exists())
        self.assertTrue((run / "transcripts" / "c0001.json").exists())

    def test_manual_cli_chosen_topic(self):
        event, _ = self._talk_cli("--topic", "teletravail")
        self.assertEqual(event["topic"]["id"], "teletravail")
        self.assertEqual(event["topic"]["text"], "le télétravail")

    def test_manual_cli_free_topic(self):
        event, _ = self._talk_cli(
            "--topic", "faut-il interdire les trottinettes ?")
        self.assertTrue(event["topic"]["id"].startswith("custom-"))
        self.assertEqual(event["topic"]["text"],
                         "faut-il interdire les trottinettes ?")
        self.assertEqual(event["topic"]["category"], "libre")

    def test_manual_cli_per_agent_models(self):
        event, _ = self._talk_cli(
            "--random-topic", "--agent-model", "1=alpha",
            "--agent-model", "5=beta")
        models = {p["id"]: p["model"] for p in event["participants"]}
        self.assertEqual(models, {1: "alpha", 5: "beta"})
        for turn in event["transcript"]:
            self.assertEqual(turn["model"], models[turn["speaker"]])

    def test_manual_cli_prints_topic_and_log_paths(self):
        out = self.base / "cli-verbose"
        cmd = [
            sys.executable, str(ROOT / "tools" / "social_sim.py"),
            "--agents", str(AGENTS), "--topics", str(TOPICS),
            "--out", str(out), "--run-id", "manual",
            "--provider", "mock", "--tick-seconds", "0",
            "--talk", "1", "5", "--topic", "teletravail", "--turns", "2",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sujet : le télétravail", result.stdout)
        self.assertIn("logs", result.stdout)
        self.assertIn("transcript", result.stdout)
        self.assertIn("Mode mock", result.stdout)

    def test_manual_agents_by_name(self):
        first, second = self.agents[0], self.agents[1]
        a, b = social_sim._manual_agents(
            self.agents, f"{first['prenom']} {first['nom']}",
            f"{second['prenom']} {second['nom']}", 0)
        self.assertEqual((a["id"], b["id"]), (first["id"], second["id"]))

    def test_manual_random_agents_are_distinct(self):
        a, b = social_sim._manual_agents(self.agents, "random", "random", 0)
        self.assertNotEqual(a["id"], b["id"])
        with self.assertRaises(SystemExit):
            social_sim._manual_agents(self.agents, "1", "1", 0)

    def test_mock_reply_is_readable_and_topic_aware(self):
        sim = self.make()
        event = sim.run_conversation(
            initiator=self.agents[0], partner=self.agents[1],
            topic="teletravail", turns=2)
        for turn in event["transcript"]:
            self.assertTrue(turn["text"].startswith("[mock]"))
            self.assertIn("le télétravail", turn["text"])
            self.assertNotIn("je t'entends bien", turn["text"])
            self.assertNotIn("Tu participes à une conversation", turn["text"])

    def test_agent_models_are_used_per_speaker(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(agent_models={first["id"]: "model-a",
                                      second["id"]: "model-b"})
        event = sim.run_conversation(
            initiator=first, partner=second, topic="teletravail", turns=2)
        models = {p["id"]: p["model"] for p in event["participants"]}
        self.assertEqual(models, {first["id"]: "model-a",
                                  second["id"]: "model-b"})
        for turn in event["transcript"]:
            self.assertEqual(turn["model"], models[turn["speaker"]])

    def test_each_agent_has_its_own_session_and_receives_replies(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(provider="ollama", model="fake", stream=False,
                        verbose=False)
        replies = {
            first["id"]: f"réplique propre à {first['prenom']}",
            second["id"]: f"réplique propre à {second['prenom']}",
        }
        calls: list[dict] = []

        def fake_respond(agent, history, provider, model, **kwargs):
            calls.append({
                "agent": agent["id"],
                "persona": agent.get("persona", ""),
                "history": [dict(m) for m in history],
            })
            return replies[agent["id"]]

        with mock.patch.object(social_sim.chat_agent, "respond",
                               side_effect=fake_respond):
            event = sim.run_conversation(
                initiator=first, partner=second, topic="teletravail",
                turns=4)

        self.assertEqual(len(calls), 4)
        session_ids = {p["session"] for p in event["participants"]}
        self.assertEqual(len(session_ids), 2)
        self.assertEqual(set(event["sessions"]),
                         {str(first["id"]), str(second["id"])})

        for index, call in enumerate(calls):
            own_id = call["agent"]
            history = call["history"]
            # Le persona de conversation est bien transmis à l'appel.
            self.assertIn("Règles de conversation", call["persona"])
            # Aucun agent ne reçoit sa propre réplique comme message utilisateur.
            for message in history:
                if message["role"] == "user":
                    self.assertNotIn(replies[own_id], message["content"])
            if index == 0:
                # Le premier orateur reçoit un prompt d'ouverture.
                self.assertEqual(history[-1]["role"], "user")
                self.assertIn("Tu ouvres une conversation", history[-1]["content"])
                continue
            previous = calls[index - 1]["agent"]
            self.assertNotEqual(previous, own_id)
            self.assertEqual(history[-1]["role"], "user")
            self.assertIn(replies[previous], history[-1]["content"])
            if index >= 2:
                # Après sa première prise de parole, l'agent la retrouve en
                # assistant dans son propre historique.
                self.assertTrue(any(
                    message["role"] == "assistant"
                    and message["content"] == replies[own_id]
                    for message in history))

    def test_self_conversation_is_rejected(self):
        sim = self.make()
        with self.assertRaises(SystemExit):
            sim.run_conversation(initiator=self.agents[0],
                                 partner=self.agents[0])

    def test_refusal_is_logged_without_transcript(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(refusal_rate=1.0)
        event = sim.run_conversation(
            initiator=first, partner=second, topic="teletravail", turns=2)
        self.assertEqual(event["type"], "refusal")
        self.assertEqual(event["refuser"]["id"], second["id"])
        self.assertEqual(event["invited_by"]["id"], first["id"])
        self.assertTrue(event["reason"])
        self.assertIsNone(event["topic"])
        self.assertEqual(event["transcript"], [])
        self.assertFalse(event["completed"])
        self.assertEqual(sim.stats["refusals"], 1)
        self.assertEqual(sim.stats["conversations"], 0)
        self.assertEqual(sim.stats["turns"], 0)
        events = [json.loads(line) for line in
                  sim.events_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["type"] for e in events], ["refusal"])
        self.assertFalse((sim.transcripts_dir / f"{event['conversation_id']}.json").exists())
        # Un refus ne consomme pas de sujet récent.
        self.assertEqual(len(sim.recent_topics), 0)

    def test_refusal_happens_even_for_a_forced_pair(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(refusal_rate=1.0)
        with mock.patch.object(
                social_sim.chat_agent, "respond",
                side_effect=AssertionError("le modèle ne doit pas être appelé")):
            event = sim.run_conversation(
                initiator=first, partner=second, topic="teletravail",
                turns=2, mode="manual")
        self.assertEqual(event["type"], "refusal")
        self.assertEqual(event["mode"], "manual")

    def test_refusal_rate_zero_always_talks(self):
        sim = self.make(refusal_rate=0.0)
        event = sim.run_conversation(
            initiator=self.agents[0], partner=self.agents[1],
            topic="teletravail", turns=2)
        self.assertEqual(event["type"], "conversation")
        self.assertEqual(len(event["transcript"]), 2)
        self.assertEqual(sim.stats["refusals"], 0)

    def test_refusals_are_seeded(self):
        def run(out_dir):
            sim = self.make(refusal_rate=1.0, out_dir=out_dir)
            return sim.run_conversation(
                initiator=self.agents[0], partner=self.agents[1],
                topic="teletravail", turns=2)
        a = run(self.base / "a")
        b = run(self.base / "b")
        self.assertEqual(a["reason"], b["reason"])
        self.assertEqual(a["conversation_id"], b["conversation_id"])

    def test_affinity_is_neutral_symmetric_and_clamped(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make()
        self.assertEqual(sim.affinity_between(first["id"], second["id"]),
                         social_sim.DEFAULT_AFFINITY)
        sim.set_affinity(first["id"], second["id"], 0.9)
        self.assertEqual(sim.affinity_between(second["id"], first["id"]), 0.9)
        self.assertEqual(sim.set_affinity(first["id"], second["id"], 2.0), 1.0)
        self.assertEqual(sim.set_affinity(first["id"], second["id"], -1.0), 0.0)

    def test_low_affinity_reduces_partner_selection(self):
        sim = self.make(agents=self.agents[:3])
        ids = [a["id"] for a in self.agents[:3]]
        for aid in ids:
            sim.positions[aid] = (0.5, 0.5)
        sim.set_affinity(ids[0], ids[1], 0.9)
        sim.set_affinity(ids[0], ids[2], 0.1)
        draws = [sim.select_partner(ids[0]) for _ in range(300)]
        self.assertGreater(draws.count(ids[1]), draws.count(ids[2]) * 2)

    def test_low_affinity_increases_refusal_probability(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(refusal_rate=0.2)
        sim.set_affinity(first["id"], second["id"], 0.9)
        high = sim.refusal_probability(first["id"], second["id"])
        sim.set_affinity(first["id"], second["id"], 0.1)
        low = sim.refusal_probability(first["id"], second["id"])
        self.assertGreater(low, high)
        self.assertAlmostEqual(high, 0.04)
        self.assertAlmostEqual(low, 0.36)

    def test_zero_refusal_rate_ignores_low_affinity(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(refusal_rate=0.0)
        sim.set_affinity(first["id"], second["id"], 0.0)
        self.assertEqual(sim.refusal_probability(first["id"], second["id"]), 0.0)
        event = sim.run_conversation(
            initiator=first, partner=second, topic="teletravail", turns=2)
        self.assertEqual(event["type"], "conversation")

    def test_conversation_increases_affinity_and_records_it(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make()
        event = sim.run_conversation(
            initiator=first, partner=second, topic="teletravail", turns=2)
        self.assertEqual(event["type"], "conversation")
        self.assertGreater(event["affinity_after"], event["affinity_before"])
        self.assertEqual(
            event["affinity_after"],
            round(sim.affinity_between(first["id"], second["id"]), 4))
        self.assertEqual(
            event["affinity_delta"],
            round(event["affinity_after"] - event["affinity_before"], 4))

    def test_conversation_affinity_is_written_to_state(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make()
        sim.run_conversation(initiator=first, partner=second,
                             topic="teletravail", turns=2)
        sim.write_state()
        state = json.loads((sim.out_dir / "state.json").read_text(
            encoding="utf-8"))
        value = sim.affinity_between(first["id"], second["id"])
        key = f"{min(first['id'], second['id'])}:{max(first['id'], second['id'])}"
        self.assertAlmostEqual(state["affinity"][key], round(value, 4))

    def test_refusal_lowers_affinity(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(refusal_rate=1.0)
        event = sim.run_conversation(
            initiator=first, partner=second, topic="teletravail", turns=2)
        self.assertEqual(event["type"], "refusal")
        self.assertLess(event["affinity_after"], event["affinity_before"])
        self.assertEqual(event["refusal_probability"], 1.0)

    def test_cli_refusal_is_visible(self):
        out = self.base / "cli-refusal"
        cmd = [
            sys.executable, str(ROOT / "tools" / "social_sim.py"),
            "--agents", str(AGENTS), "--topics", str(TOPICS),
            "--out", str(out), "--run-id", "manual",
            "--provider", "mock", "--tick-seconds", "0",
            "--refusal-rate", "1",
            "--talk", "1", "5", "--topic", "teletravail", "--turns", "2",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("refuse de discuter", result.stdout)
        run = out / "manual"
        event = json.loads(
            (run / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(event["type"], "refusal")
        self.assertFalse((run / "transcripts" / "c0001.json").exists())

    def test_cli_run(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "runs"
            cmd = [
                sys.executable, str(ROOT / "tools" / "social_sim.py"),
                "--agents", str(AGENTS), "--topics", str(TOPICS),
                "--out", str(out), "--run-id", "test",
                "--provider", "mock", "--max-ticks", "2",
                "--spark-rate", "1", "--tick-seconds", "0",
                "--limit", "3", "--quiet", "--refusal-rate", "0",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            run = out / "test"
            self.assertTrue((run / "events.jsonl").exists())
            self.assertTrue((run / "config.json").exists())
            events = [json.loads(line) for line in
                      (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
                      if line]
            self.assertEqual(len(events), 2)


    # ----- Phase E : groupes persistants

    def _place(self, sim, ids, center=(0.3, 0.3), spread=0.005):
        for index, aid in enumerate(ids):
            sim.positions[aid] = (center[0] + index * spread, center[1])

    def _affine(self, sim, ids, value):
        for index, left in enumerate(ids):
            for right in ids[index + 1:]:
                sim.set_affinity(left, right, value)

    def _form_one_group(self, sim, ids):
        self._place(sim, ids)
        self._affine(sim, ids, 0.9)
        sim.tick = max(1, sim.tick)
        sim._update_groups()
        self.assertEqual(len(sim.groups), 1)
        return next(iter(sim.groups.values()))

    def test_groups_form_by_affinity_and_proximity(self):
        sim = self.make(agents=self.agents[:6], group_min=3, group_max=3,
                        group_affinity=0.6, group_radius=0.2, max_groups=10)
        ids = [a["id"] for a in self.agents[:6]]
        left, right = ids[:3], ids[3:]
        self._place(sim, left, center=(0.2, 0.2))
        self._place(sim, right, center=(0.8, 0.8))
        self._affine(sim, left, 0.9)
        self._affine(sim, right, 0.9)
        for a in left:
            for b in right:
                sim.set_affinity(a, b, 0.1)
        sim.tick = 1
        sim._update_groups()
        self.assertEqual(len(sim.groups), 2)
        members = sorted(tuple(g["members"]) for g in sim.groups.values())
        self.assertEqual(members, sorted([tuple(sorted(left)), tuple(sorted(right))]))
        for group in sim.groups.values():
            self.assertGreaterEqual(len(group["members"]), 3)
            self.assertLessEqual(len(group["members"]), 3)
            self.assertGreater(group["cohesion"], 0.5)

    def test_group_member_leaves_then_group_dissolves(self):
        sim = self.make(agents=self.agents[:4], group_min=3, group_max=4,
                        group_affinity=0.6, group_radius=0.2)
        ids = [a["id"] for a in self.agents[:4]]
        group = self._form_one_group(sim, ids)
        self.assertEqual(len(group["members"]), 4)

        leaving = ids[3]
        for other in ids[:3]:
            sim.set_affinity(leaving, other, 0.1)
        sim.tick += 1
        sim._update_groups()
        self.assertEqual(len(sim.groups), 1)
        group = next(iter(sim.groups.values()))
        self.assertNotIn(leaving, group["members"])
        self.assertEqual(len(group["members"]), 3)

        self._affine(sim, ids[:3], 0.1)
        sim.tick += 1
        sim._update_groups()
        self.assertEqual(sim.groups, {})

    def test_group_admits_a_close_well_connected_agent(self):
        sim = self.make(agents=self.agents[:4], group_min=3, group_max=4,
                        group_affinity=0.6, group_radius=0.2)
        ids = [a["id"] for a in self.agents[:4]]
        candidate = ids[3]
        sim.positions[candidate] = (0.9, 0.9)
        group = self._form_one_group(sim, ids[:3])
        self.assertEqual(len(group["members"]), 3)

        sim.positions[candidate] = (0.305, 0.305)
        for member in ids[:3]:
            sim.set_affinity(candidate, member, 0.9)
        sim.tick += 1
        sim._update_groups()
        group = next(iter(sim.groups.values()))
        self.assertIn(candidate, group["members"])
        self.assertEqual(len(group["members"]), 4)

    def test_group_conversation_uses_one_session_per_member(self):
        sim = self.make(agents=self.agents[:3], group_min=3, group_max=3,
                        group_affinity=0.6, group_radius=0.2)
        ids = [a["id"] for a in self.agents[:3]]
        group = self._form_one_group(sim, ids)
        event = sim.run_group_conversation(group, turns=6)

        self.assertEqual(event["mode"], "group")
        self.assertEqual(event["group_id"], group["id"])
        self.assertEqual(event["type"], "conversation")
        self.assertEqual({p["id"] for p in event["participants"]}, set(ids))
        self.assertEqual(set(event["sessions"]), {str(i) for i in ids})
        self.assertEqual(len({t["session"] for t in event["transcript"]}), 3)
        speakers = [t["speaker"] for t in event["transcript"]]
        for aid in ids:
            self.assertEqual(speakers.count(aid), 2)
        self.assertEqual(event["group"]["size"], 3)
        self.assertEqual(group["conversations"], 1)
        self.assertEqual(sim.stats["group_conversations"], 1)
        self.assertEqual(sim.stats["conversations"], 1)

    def test_group_state_is_persisted(self):
        sim = self.make(agents=self.agents[:3], group_min=3, group_max=3,
                        group_affinity=0.6, group_radius=0.2)
        ids = [a["id"] for a in self.agents[:3]]
        group = self._form_one_group(sim, ids)
        sim.run_group_conversation(group, turns=4)
        sim.write_state()
        state = json.loads((sim.out_dir / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["phase"], "A+B+C+D+E+F")
        self.assertEqual(state["conversation_no"], 1)
        self.assertEqual(len(state["groups"]), 1)
        saved = state["groups"][group["id"]]
        self.assertEqual(saved["members"], group["members"])
        self.assertEqual(saved["conversations"], 1)
        self.assertAlmostEqual(saved["cohesion"], round(group["cohesion"], 4))

    def test_resume_restores_state_and_keeps_events(self):
        sim = self.make(agents=self.agents[:3], group_min=3, group_max=3,
                        group_affinity=0.6, group_radius=0.2)
        ids = [a["id"] for a in self.agents[:3]]
        group = self._form_one_group(sim, ids)
        sim.run_group_conversation(group, turns=4)
        sim.tick = 9
        sim.write_state()
        data = json.loads((sim.out_dir / "state.json").read_text(encoding="utf-8"))
        events_before = len(sim.events_path.read_text(encoding="utf-8").splitlines())

        resumed = self.make(agents=self.agents[:3], resume=data,
                            resume_source=str(sim.out_dir / "state.json"))
        self.assertEqual(resumed.tick, 9)
        self.assertEqual(resumed.conversation_no, sim.conversation_no)
        self.assertEqual(sorted(resumed.groups), sorted(sim.groups))
        saved = resumed.groups[group["id"]]
        self.assertEqual(saved["members"], group["members"])
        self.assertEqual(saved["conversations"], 1)
        self.assertAlmostEqual(saved["cohesion"], group["cohesion"], places=4)
        for (aid, bid), value in sim.affinity.items():
            self.assertAlmostEqual(resumed.affinity_between(aid, bid), value,
                                   places=4)

        resumed.run_conversation(topic="teletravail", turns=2)
        events_after = len(resumed.events_path.read_text(
            encoding="utf-8").splitlines())
        self.assertEqual(events_after, events_before + 1)

    def test_cli_resume_missing_state_fails(self):
        missing = Path(self.tmp.name) / "absent"
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "social_sim.py"),
             "--resume", str(missing), "--quiet"],
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("État introuvable", result.stderr)

    def test_cli_resume_continues_run(self):
        out = Path(self.tmp.name) / "runs"
        base = [sys.executable, str(ROOT / "tools" / "social_sim.py"),
                "--provider", "mock", "--out", str(out), "--run-id", "r1",
                "--spark-rate", "1", "--tick-seconds", "0", "--limit", "6",
                "--refusal-rate", "0", "--group-rate", "0", "--quiet"]
        first = subprocess.run(base + ["--max-ticks", "2"],
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(first.returncode, 0, first.stderr)
        run = out / "r1"
        state = json.loads((run / "state.json").read_text(encoding="utf-8"))
        tick = state["tick"]
        events = len((run / "events.jsonl").read_text(encoding="utf-8").splitlines())
        self.assertEqual(events, 2)

        second = subprocess.run(
            base + ["--resume", str(run), "--max-ticks", "2"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(second.returncode, 0, second.stderr)
        state = json.loads((run / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["tick"], tick + 2)
        events = len((run / "events.jsonl").read_text(encoding="utf-8").splitlines())
        self.assertEqual(events, 4)
        self.assertEqual(state["phase"], "A+B+C+D+E+F")


    def test_cli_reset_clears_state_and_transcripts(self):
        out = Path(self.tmp.name) / "runs"
        base = [sys.executable, str(ROOT / "tools" / "social_sim.py"),
                "--provider", "mock", "--out", str(out), "--run-id", "r2",
                "--spark-rate", "1", "--tick-seconds", "0", "--limit", "6",
                "--refusal-rate", "0", "--group-rate", "0", "--quiet",
                "--max-ticks", "3"]
        first = subprocess.run(base, capture_output=True, text=True, timeout=60)
        self.assertEqual(first.returncode, 0, first.stderr)
        run = out / "r2"
        self.assertEqual(len(list((run / "transcripts").glob("*.json"))), 3)

        second = subprocess.run(base + ["--reset"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(second.returncode, 0, second.stderr)
        state = json.loads((run / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["stats"]["conversations"], 3)
        self.assertEqual(state["conversation_no"], 3)
        self.assertEqual(state["tick"], 3)
        self.assertEqual(len(list((run / "transcripts").glob("*.json"))), 3)
        events = (run / "events.jsonl").read_text(
            encoding="utf-8").splitlines()
        self.assertEqual(len(events), 3)

    def test_cli_reset_conflicts_with_resume(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "social_sim.py"),
             "--reset", "--resume", str(Path(self.tmp.name) / "absent"),
             "--quiet"],
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incompatibles", result.stderr)


    # ----- Phase F : mémoire sociale et opinions

    def test_opinions_are_deterministic_and_clamped(self):
        first = self.agents[0]["id"]
        sim = self.make(agents=self.agents[:2])
        other = self.make(agents=self.agents[:2], out_dir=self.base / "b")
        value = sim.opinion_of(first, "teletravail")
        self.assertGreaterEqual(value, 0.15)
        self.assertLessEqual(value, 0.85)
        self.assertEqual(value, other.opinion_of(first, "teletravail"))
        self.assertEqual(sim.set_opinion(first, "teletravail", 1.7), 1.0)
        self.assertEqual(sim.set_opinion(first, "teletravail", -0.4), 0.0)

    def test_conversation_moves_opinions_toward_the_mean(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(agents=[first, second], opinion_influence=0.5,
                        refusal_rate=0.0)
        sim.set_opinion(first["id"], "teletravail", 0.9)
        sim.set_opinion(second["id"], "teletravail", 0.1)
        event = sim.run_conversation(topic="teletravail", turns=2)
        self.assertEqual(event["opinions_before"][str(first["id"])], 0.9)
        self.assertEqual(event["opinions_before"][str(second["id"])], 0.1)
        after_first = sim.opinion_of(first["id"], "teletravail")
        after_second = sim.opinion_of(second["id"], "teletravail")
        self.assertLess(after_first, 0.9)
        self.assertGreater(after_second, 0.1)
        # Les deux se rapprochent du même point sans se croiser.
        self.assertGreater(after_first, after_second)
        self.assertAlmostEqual(event["opinions_after"][str(first["id"])],
                               after_first, places=4)
        self.assertLessEqual(event["agreement"], 1.0)

    def test_opinion_influence_zero_disables_updates(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(agents=[first, second], opinion_influence=0.0,
                        refusal_rate=0.0)
        sim.set_opinion(first["id"], "teletravail", 0.9)
        sim.set_opinion(second["id"], "teletravail", 0.1)
        sim.run_conversation(topic="teletravail", turns=2)
        self.assertEqual(sim.opinion_of(first["id"], "teletravail"), 0.9)
        self.assertEqual(sim.opinion_of(second["id"], "teletravail"), 0.1)

    def test_pair_memory_records_and_deduplicates_topics(self):
        first, second = self.agents[0], self.agents[1]
        other = next(t for t in self.topics if t["id"] != "teletravail")
        sim = self.make(agents=[first, second], refusal_rate=0.0)
        sim.run_conversation(topic="teletravail", turns=2)
        sim.run_conversation(topic="teletravail", turns=2)
        sim.run_conversation(topic=other["id"], turns=2)
        entry = sim.memory[sim._pair(first["id"], second["id"])]
        self.assertEqual(entry["topics"], ["teletravail", other["id"]])
        self.assertEqual(entry["conversations"], 3)
        self.assertEqual(entry["last_topic"], other["id"])
        self.assertEqual(entry["last_tick"], sim.tick)

    def test_repeat_penalty_lowers_topic_weight(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(agents=[first, second], repeat_penalty=0.25)
        pair = [first, second]
        available = [{"id": "teletravail"}, {"id": "cuisine"}]
        self.assertEqual(sim._topic_weights(pair, available), [1.0, 1.0])
        sim.memory[sim._pair(first["id"], second["id"])] = {
            "topics": ["teletravail"], "last_tick": 1, "conversations": 1,
            "refusals": 0, "last_delta": 0.02, "last_topic": "teletravail",
        }
        self.assertEqual(sim._topic_weights(pair, available), [0.25, 1.0])

    def test_memory_note_mentions_previous_and_other_topics(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(agents=[first, second], refusal_rate=0.0)
        sim.run_conversation(topic="teletravail", turns=2)
        people = [first, second]
        repeat = sim._memory_note(people, "teletravail")
        self.assertIn("déjà parlé", repeat)
        fresh = sim._memory_note(people, "cuisine")
        self.assertIn("déjà abordés", fresh)
        self.assertIn("télétravail", fresh)

    def test_memory_and_opinions_persist_and_resume(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(agents=[first, second], opinion_influence=0.5,
                        refusal_rate=0.0)
        sim.set_opinion(first["id"], "teletravail", 0.8)
        sim.run_conversation(topic="teletravail", turns=2)
        sim.write_state()
        data = json.loads((sim.out_dir / "state.json").read_text(encoding="utf-8"))
        self.assertIn(str(first["id"]), data["opinions"])
        self.assertIn("teletravail", data["opinions"][str(first["id"])])
        pair_key = f"{min(first['id'], second['id'])}:{max(first['id'], second['id'])}"
        self.assertEqual(data["memory"][pair_key]["topics"], ["teletravail"])

        resumed = self.make(agents=[first, second], resume=data,
                            resume_source=str(sim.out_dir / "state.json"))
        self.assertAlmostEqual(
            resumed.opinion_of(first["id"], "teletravail"),
            sim.opinion_of(first["id"], "teletravail"), places=4)
        entry = resumed.memory[resumed._pair(first["id"], second["id"])]
        self.assertEqual(entry["topics"], ["teletravail"])
        self.assertEqual(entry["conversations"], 1)

    def test_refusal_memory_raises_again_probability(self):
        first, second = self.agents[0], self.agents[1]
        sim = self.make(agents=[first, second], refusal_rate=0.1)
        base = sim.refusal_probability(first["id"], second["id"])
        self.assertAlmostEqual(base, 0.1)
        entry = sim._pair_memory(first["id"], second["id"])
        entry["last_delta"] = -0.08
        self.assertAlmostEqual(
            sim.refusal_probability(first["id"], second["id"]), 0.14)

    def test_state_every_writes_periodically(self):
        sim = self.make(state_every=1)
        with mock.patch.object(sim, "write_state",
                               wraps=sim.write_state) as writer:
            sim.run(3)
        self.assertEqual(writer.call_count, 4)  # 3 ticks + écriture finale


if __name__ == "__main__":
    unittest.main()
