"""Tests du pipeline agents : marges exactes, identités uniques, conversation.

Lancer :  uv run python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_unique_full_attribute_vectors(self):
        vectors = {tuple(sorted(a["attributes"].items())) for a in self.agents}
        self.assertEqual(len(vectors), 500)

    def test_base_personas_do_not_suffice(self):
        # Preuve que les noms sont nécessaires : seules ~302 personas contraintes sont distinctes.
        self.assertLess(self.manifest["distinct_base_personas"], 500)
        self.assertGreater(self.manifest["distinct_base_personas"], 250)

    def test_prompt_is_complete(self):
        agent = self.agents[0]
        self.assertIn(agent["prenom"], agent["persona"])
        self.assertIn(agent["nom"], agent["persona"])
        self.assertIn(agent["identity_key"], agent["persona"])
        for attr, value in agent["attributes"].items():
            self.assertIn(value, agent["persona"])

    def test_csv_matches_jsonl(self):
        with (self.out / "agents.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(self.agents))
        self.assertEqual({r["identity_key"] for r in rows},
                         {a["identity_key"] for a in self.agents})

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


if __name__ == "__main__":
    unittest.main()
