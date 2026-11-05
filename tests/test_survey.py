"""Tests de la sonde : sélection, exécution mock, reprise, agrégation.

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

import build_agents  # noqa: E402
import chat_agent  # noqa: E402
import survey  # noqa: E402

CONSTRAINTS = ROOT / "constraint_cells.csv"
POPULATION = ROOT / "population.csv"
DEMO = ROOT / "surveys" / "demo.fr.json"

QUESTIONNAIRE = {
    "title": "Test",
    "questions": [
        {"id": "a", "text": "Favorable ?", "choices": ["oui", "non", "nsp"]},
        {"id": "b", "text": "Raconte ta journée."},
    ],
}


class SurveySelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "agents"
        build_agents.build(CONSTRAINTS, POPULATION, cls.out, seed=0, labels_path=None)
        cls.agents = chat_agent.load_agents(cls.out / "agents.jsonl")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_and_one_per_cluster(self):
        self.assertEqual(len(survey.select_agents(self.agents)), 500)
        one = survey.select_agents(self.agents, one_per_cluster=True)
        self.assertEqual(len(one), 302)
        self.assertEqual(len({a["cluster_id"] for a in one}), 302)

    def test_where_and_ids_and_limit(self):
        self.assertEqual(len(survey.select_agents(self.agents, where=["A1=v2"])), 254)
        self.assertEqual(
            len(survey.select_agents(self.agents, where=survey.parse_where(["A1=v2,A9=v1"]))), 47)
        self.assertEqual([a["id"] for a in survey.select_agents(self.agents, ids=[1, 5, 23])],
                         [1, 5, 23])
        sample = survey.select_agents(self.agents, limit=10, seed=0)
        self.assertEqual(len(sample), 10)
        self.assertEqual({a["id"] for a in sample},
                         {a["id"] for a in survey.select_agents(self.agents, limit=10, seed=0)})

    def test_match_choice(self):
        self.assertEqual(survey.match_choice("  Oui. ", ["oui", "non"]), "oui")
        self.assertEqual(survey.match_choice("NON", ["oui", "non", "nsp"]), "non")
        self.assertEqual(survey.match_choice("Vélo ou marche", ["voiture", "velo ou marche"]),
                         "velo ou marche")
        self.assertIsNone(survey.match_choice("peut-être", ["oui", "non"]))
        self.assertIsNone(survey.match_choice("", ["oui"]))

    def test_load_survey_demo(self):
        s = survey.load_survey(DEMO)
        self.assertTrue(s["title"])
        self.assertEqual(len(s["questions"]), 5)
        ids = [q["id"] for q in s["questions"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIsNone(s["questions"][-1]["choices"])  # question ouverte

    def test_load_survey_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.json"
            bad.write_text(json.dumps({"questions": [{"id": "x", "text": "1"},
                                                     {"id": "x", "text": "2"}]}),
                           encoding="utf-8")
            with self.assertRaises(SystemExit):
                survey.load_survey(bad)


class SurveyRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "agents"
        build_agents.build(CONSTRAINTS, POPULATION, cls.out, seed=0, labels_path=None)
        cls.agents = survey.select_agents(
            chat_agent.load_agents(cls.out / "agents.jsonl"), limit=5)
        cls.survey = QUESTIONNAIRE

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, out):
        return survey.run_survey(self.agents, self.survey, provider="mock",
                                 model="mock", out=out, concurrency=3, progress=False)

    def test_run_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "ans.jsonl"
            ans = self._run(out)
            self.assertEqual(len(ans), len(self.agents) * len(self.survey["questions"]))
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(ans))
            # choix valides
            qa = next(q for q in self.survey["questions"] if q["id"] == "a")
            for r in ans:
                if r["question_id"] == "a":
                    self.assertIn(r["choice"], qa["choices"])
            # reprise : rien de nouveau
            again = self._run(out)
            self.assertEqual(len(again), len(ans))
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), len(ans))

    def test_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "ans.jsonl"
            ans = self._run(out)
            rep = survey.aggregate(self.survey, ans, self.agents, crosstab=["A1"])
            self.assertEqual(rep["n_agents"], len(self.agents))
            qa = next(q for q in rep["questions"] if q["id"] == "a")
            self.assertEqual(sum(qa["counts"].values()), len(self.agents))
            self.assertIn("A1", qa["crosstab"])
            # chaque agent apparaît une fois par valeur d'attribut
            total = sum(sum(v.values()) for v in qa["crosstab"]["A1"].values())
            self.assertEqual(total, len(self.agents))
            # la question ouverte n'a pas de choix
            qb = next(q for q in rep["questions"] if q["id"] == "b")
            self.assertNotIn("counts", qb)
            self.assertTrue(qb["samples"])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "c.jsonl"
            r = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "survey.py"), str(DEMO),
                 "--provider", "mock", "--limit", "3", "--seed", "0",
                 "--out", str(out), "--no-report"],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 3 * 5)


if __name__ == "__main__":
    unittest.main()
