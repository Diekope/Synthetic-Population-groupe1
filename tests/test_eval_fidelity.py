"""Tests de l'évaluation de fidélité : appariement, harnais mock, CLI.

Lancer :  uv run python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import chat_agent  # noqa: E402
import eval_fidelity  # noqa: E402

PROBES = ROOT / "tools" / "fidelity_probes.fr.json"
LABELS = ROOT / "agents" / "labels.json"


class MatcherTest(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(eval_fidelity.norm("1 500"), "1500")
        self.assertEqual(eval_fidelity.norm("Très bon !"), "tres bon")
        self.assertEqual(eval_fidelity.norm("outre-mer"), "outre mer")

    def test_token_in_boundaries(self):
        self.assertTrue(eval_fidelity.token_in("j ai une voiture", "une voiture"))
        self.assertFalse(eval_fidelity.token_in("bonjour", "bon"))

    def test_matches_accept(self):
        self.assertTrue(eval_fidelity.matches("Je possède une seule voiture.",
                                              {"accept": ["une seule voiture"]}))
        self.assertTrue(eval_fidelity.matches("Non, aucune voiture.",
                                              {"accept": ["aucune voiture"]}))
        self.assertFalse(eval_fidelity.matches("J'ai deux voitures.",
                                               {"accept": ["une seule voiture"]}))

    def test_matches_range(self):
        self.assertTrue(eval_fidelity.matches("J'ai 30 ans.", {"range": [18, 39]}))
        self.assertFalse(eval_fidelity.matches("J'ai 45 ans.", {"range": [18, 39]}))
        self.assertTrue(eval_fidelity.matches("environ 1 500 euros", {"range": [1500, 2500]}))

    def test_probes_cover_all_attributes(self):
        labels = json.loads(LABELS.read_text(encoding="utf-8"))
        probes = eval_fidelity.load_probes(PROBES, labels)
        self.assertEqual(set(probes), {f"A{i}" for i in range(12)})
        for attr, probe in probes.items():
            self.assertEqual(set(probe["values"]), {"v0", "v1", "v2"}, attr)
            # le libellé est ajouté aux motifs
            self.assertIn(labels[f"{attr}=v1"], probes[attr]["values"]["v1"]["accept"])


class HarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = json.loads(LABELS.read_text(encoding="utf-8"))
        cls.probes = eval_fidelity.load_probes(PROBES, cls.labels)
        cls.agents = chat_agent.load_agents(ROOT / "agents" / "agents.jsonl")[:6]

    def test_mock_is_perfect(self):
        results = eval_fidelity.evaluate(
            self.agents, self.probes, provider="mock", model="mock",
            labels=self.labels, progress=False)
        self.assertEqual(len(results), len(self.agents) * 12)
        report = eval_fidelity.aggregate(results, self.agents)
        self.assertEqual(report["overall"], 1.0)
        self.assertEqual(report["errors"], 0)

    def test_aggregate_counts(self):
        results = eval_fidelity.evaluate(
            self.agents, self.probes, provider="mock", model="mock",
            labels=self.labels, progress=False)
        report = eval_fidelity.aggregate(results, self.agents)
        self.assertEqual(report["n_questions"], len(self.agents) * 12)
        self.assertEqual(len(report["per_agent"]), len(self.agents))
        self.assertEqual(set(report["per_attribute"]), {f"A{i}" for i in range(12)})

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            rep = Path(d) / "r.json"
            out = Path(d) / "a.jsonl"
            r = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "eval_fidelity.py"),
                 "--provider", "mock", "--limit", "3", "--seed", "0",
                 "--out", str(out), "--report", str(rep)],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 3 * 12)
            data = json.loads(rep.read_text(encoding="utf-8"))
            self.assertEqual(data["mock"]["overall"], 1.0)


if __name__ == "__main__":
    unittest.main()
