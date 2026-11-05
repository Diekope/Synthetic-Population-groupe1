"""Tests du pipeline agents : marges exactes, identités uniques, conversation.

Lancer :  uv run python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_agents  # noqa: E402
import chat_agent  # noqa: E402

CONSTRAINTS = ROOT / "constraint_cells.csv"
POPULATION = ROOT / "population.csv"


class AgentPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "agents"
        cls.manifest = build_agents.build(CONSTRAINTS, POPULATION, cls.out, seed=0, labels_path=None)
        cls.agents = chat_agent.load_agents(cls.out / "agents.jsonl")
        cls.tables = build_agents.load_tables(CONSTRAINTS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_size(self):
        self.assertEqual(self.manifest["n_agents"], 500)
        self.assertEqual(len(self.agents), 500)

    def test_margins_exact_after_enrichment(self):
        pop = [a["attributes"] for a in self.agents]
        error, violated = build_agents.margin_error(pop, self.tables)
        self.assertEqual(error, 0)
        self.assertEqual(violated, 0)
        self.assertEqual(self.manifest["margin_error"], 0)

    def test_unique_identities(self):
        keys = {a["identity_key"] for a in self.agents}
        self.assertEqual(len(keys), 500)
        names = {(a["prenom"], a["nom"]) for a in self.agents}
        self.assertEqual(len(names), 500)

    def test_unique_full_attribute_vectors(self):
        vectors = {tuple(sorted(a["attributes"].items())) for a in self.agents}
        self.assertEqual(len(vectors), 500)

    def test_exact_copies_are_exposed(self):
        from collections import defaultdict
        by_cluster = defaultdict(list)
        for a in self.agents:
            by_cluster[a["cluster_id"]].append(a)
        # Un cluster = un seul profil contraint.
        for members in by_cluster.values():
            profile = tuple(members[0]["attributes"][f"A{i}"] for i in range(12))
            self.assertTrue(all(
                tuple(m["attributes"][f"A{i}"] for i in range(12)) == profile
                for m in members))
            self.assertEqual(len(members), members[0]["copies"])
            self.assertEqual(sorted(m["copy_index"] for m in members),
                             list(range(1, len(members) + 1)))
        self.assertEqual(len(by_cluster), self.manifest["distinct_profiles"])
        self.assertEqual(len(by_cluster), 302)
        self.assertEqual(sum(1 for a in self.agents if a["copies"] > 1),
                         self.manifest["duplicated_agents"])
        self.assertEqual(self.manifest["duplicated_agents"], 227)
        self.assertEqual(self.manifest["exact_copies"], 198)
        self.assertEqual(self.manifest["max_copies"],
                         max(a["copies"] for a in self.agents))

    def test_base_personas_do_not_suffice(self):
        # Preuve que les noms sont nécessaires : seules ~302 personas contraintes sont distinctes.
        self.assertLess(self.manifest["distinct_base_personas"], 500)
        self.assertGreater(self.manifest["distinct_base_personas"], 250)

    def test_prompt_is_complete(self):
        agent = self.agents[0]
        self.assertIn(agent["prenom"], agent["persona"])
        self.assertIn(agent["nom"], agent["persona"])
        self.assertIn("EXACTEMENT", agent["persona"])
        self.assertIn("modèle de langage", agent["persona"])
        for value in agent["attributes"].values():
            self.assertIn(value, agent["persona"])

    def test_default_labels_are_applied(self):
        with tempfile.TemporaryDirectory() as d:
            labels = ROOT / "tools" / "agent_labels.fr.json"
            build_agents.build(CONSTRAINTS, POPULATION, Path(d), seed=0, labels_path=labels)
            agent = json.loads((Path(d) / "agents.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("classe d'âge", agent["persona"])
            self.assertIn("revenu mensuel du ménage", agent["persona"])
            self.assertTrue((Path(d) / "labels.json").exists())

    def test_names_match_declared_sex(self):
        labels = json.loads((ROOT / "agents" / "labels.json").read_text(encoding="utf-8"))
        committed = chat_agent.load_agents(ROOT / "agents" / "agents.jsonl")
        female = set(build_agents.PRENOMS_F)
        male = set(build_agents.PRENOMS_M)
        for a in committed:
            sex = labels.get("A1=" + a["attributes"]["A1"], "")
            if sex == "femme":
                self.assertIn(a["prenom"], female, a["prenom"])
            elif sex == "homme":
                self.assertIn(a["prenom"], male, a["prenom"])

    def test_csv_matches_jsonl(self):
        with (self.out / "agents.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(self.agents))
        self.assertEqual({r["identity_key"] for r in rows},
                         {a["identity_key"] for a in self.agents})
        for col in ("stable_id", "cluster_id", "copy_index", "copies"):
            self.assertIn(col, rows[0])
        self.assertEqual({r["cluster_id"] for r in rows},
                         {a["cluster_id"] for a in self.agents})
        by_key = {a["identity_key"]: a for a in self.agents}
        for r in rows:
            a = by_key[r["identity_key"]]
            self.assertEqual(r["stable_id"], a["stable_id"])
            self.assertEqual(r["cluster_id"], a["cluster_id"])
            self.assertEqual(int(r["copy_index"]), a["copy_index"])
            self.assertEqual(int(r["copies"]), a["copies"])

    def test_mock_chat(self):
        agent = self.agents[0]
        answer = chat_agent.respond(
            agent, [{"role": "user", "content": "Bonjour"}], "mock", "mock")
        self.assertIn(agent["prenom"], answer)
        self.assertIn("Bonjour", answer)

    def test_history_is_included(self):
        agent = self.agents[0]
        history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        msgs = chat_agent.build_messages(agent, history)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual([m["role"] for m in msgs[1:]], ["user", "assistant"])

    def test_selection(self):
        agent = self.agents[7]
        self.assertEqual(chat_agent.select_agent(self.agents, str(agent["id"]))["id"], agent["id"])
        self.assertEqual(
            chat_agent.select_agent(self.agents, agent["identity_key"].upper())["id"], agent["id"])
        self.assertEqual(
            chat_agent.select_agent(self.agents, f"{agent['prenom']} {agent['nom']}")["id"],
            agent["id"])

    def test_cli_list_and_one_shot(self):
        base = [sys.executable, str(ROOT / "tools" / "chat_agent.py"),
                "--agents", str(self.out / "agents.jsonl")]
        listed = subprocess.run(base + ["--list"], capture_output=True, text=True, check=True)
        self.assertEqual(len(listed.stdout.strip().splitlines()), 500)

        answer = subprocess.run(base + ["--agent", "1", "--message", "Salut"],
                                capture_output=True, text=True, check=True)
        self.assertIn(self.agents[0]["prenom"], answer.stdout)


class ReproducibilityTest(unittest.TestCase):
    def test_stable_ids_are_uuid_and_unique(self):
        import uuid as _uuid
        agents = chat_agent.load_agents(ROOT / "agents" / "agents.jsonl")
        ids = [a["stable_id"] for a in agents]
        self.assertEqual(len(set(ids)), len(ids))
        for sid in ids[:20]:
            _uuid.UUID(sid)  # lève si invalide

    def test_deterministic_output(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            m1 = build_agents.build(CONSTRAINTS, POPULATION, Path(d1), seed=0, labels_path=None)
            m2 = build_agents.build(CONSTRAINTS, POPULATION, Path(d2), seed=0, labels_path=None)
            self.assertEqual(m1, m2)
            for name in ("agents.jsonl", "agents.csv"):
                self.assertEqual((Path(d1) / name).read_bytes(),
                                 (Path(d2) / name).read_bytes())

    def test_different_seed_changes_names(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            build_agents.build(CONSTRAINTS, POPULATION, Path(d1), seed=0, labels_path=None)
            build_agents.build(CONSTRAINTS, POPULATION, Path(d2), seed=7, labels_path=None)
            a = (Path(d1) / "agents.jsonl").read_bytes()
            b = (Path(d2) / "agents.jsonl").read_bytes()
            self.assertNotEqual(a, b)

    def test_check_mode(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "agents"
            build_agents.build(CONSTRAINTS, POPULATION, out, seed=0, labels_path=None)
            self.assertEqual(build_agents.check(CONSTRAINTS, POPULATION, out, 0, None), 0)
            # on altère un fichier -> la vérification doit échouer
            (out / "agents.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertEqual(build_agents.check(CONSTRAINTS, POPULATION, out, 0, None), 1)

    def test_manifest_provenance(self):
        manifest = json.loads((ROOT / "agents" / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("inputs", manifest)
        self.assertEqual(len(manifest["inputs"]["population_sha256"]), 64)
        self.assertEqual(manifest["margin_error"], 0)


class PlumbingTest(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertEqual(chat_agent.estimate_tokens("abcd"), 1)
        self.assertEqual(chat_agent.estimate_tokens("a" * 400), 100)
        self.assertEqual(chat_agent.estimate_tokens(""), 1)

    def test_trim_history_keeps_recent_within_budget(self):
        history = [{"role": "user", "content": "a" * 400},   # ~100 tokens
                   {"role": "assistant", "content": "b" * 400},
                   {"role": "user", "content": "c" * 40}]     # ~10 tokens
        kept = chat_agent.trim_history(history, 120, system_tokens=0)
        self.assertEqual(kept[-1]["content"], "c" * 40)
        self.assertLessEqual(sum(chat_agent.estimate_tokens(m["content"]) for m in kept), 120)

    def test_trim_history_always_keeps_last(self):
        history = [{"role": "user", "content": "x" * 4000}]
        kept = chat_agent.trim_history(history, 10)
        self.assertEqual(len(kept), 1)

    def test_seeded_random_selection(self):
        agents = [{"id": i, "prenom": "P", "nom": "N", "identity_key": str(i),
                   "attributes": {}} for i in range(50)]
        self.assertEqual(chat_agent.select_agent(agents, "random", seed=1)["id"],
                         chat_agent.select_agent(agents, "random", seed=1)["id"])

    def test_stream_parsers(self):
        ollama_lines = [
            json.dumps({"message": {"content": "Bon"}, "done": False}).encode(),
            json.dumps({"message": {"content": "jour"}, "done": True}).encode(),
        ]
        got = []
        self.assertEqual(chat_agent._read_ollama_stream(ollama_lines, got.append), "Bonjour")
        self.assertEqual(got, ["Bon", "jour"])

        openai_lines = [
            b'data: {"choices":[{"delta":{"content":"Sal"}}]}',
            b'data: {"choices":[{"delta":{"content":"ut"}}]}',
            b"data: [DONE]",
        ]
        got2 = []
        self.assertEqual(chat_agent._read_openai_stream(openai_lines, got2.append), "Salut")
        self.assertEqual(got2, ["Sal", "ut"])

    def test_retry_on_429_then_success(self):
        calls = {"n": 0}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, *a): return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}).encode()

        def fake_open(req, timeout):
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.HTTPError(req.full_url, 429, "rate", {}, io.BytesIO(b"slow"))
            return FakeResp()

        os.environ["OPENAI_API_KEY"] = "test"
        with mock.patch.object(chat_agent, "_open", fake_open), \
                mock.patch.object(chat_agent.time, "sleep", lambda *_: None):
            out = chat_agent.call_openai([{"role": "user", "content": "hi"}], "m", retries=2)
        self.assertEqual(out, "ok")
        self.assertEqual(calls["n"], 3)

    def test_ollama_404_is_not_retried(self):
        calls = {"n": 0}

        def fake_open(req, timeout):
            calls["n"] += 1
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, io.BytesIO(b"nope"))

        with mock.patch.object(chat_agent, "_open", fake_open):
            with self.assertRaises(chat_agent.LLMError) as cm:
                chat_agent.call_ollama([{"role": "user", "content": "hi"}], "ghost", retries=2)
        self.assertIn("introuvable", str(cm.exception))
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
