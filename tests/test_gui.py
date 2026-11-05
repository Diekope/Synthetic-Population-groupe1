"""Tests de la GUI : serveur local, API JSON, chat mock.

Lancer :  uv run python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gui  # noqa: E402


class GuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        surveys = Path(cls.tmp.name) / "surveys"
        surveys.mkdir()
        shutil.copy(ROOT / "surveys" / "demo.fr.json", surveys / "demo.fr.json")
        cls.surveys_dir = surveys
        cls.httpd = gui.create_server(
            ROOT / "agents" / "agents.jsonl", "127.0.0.1", 0, surveys)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)
        cls.tmp.cleanup()

    def _get(self, path, raw=False):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
            body = r.read()
            return body if raw else json.loads(body)

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_index(self):
        html = self._get("/", raw=True).decode()
        self.assertIn("Population synthétique", html)
        self.assertIn("/api/chat", html)
        self.assertIn('id="model"', html)
        self.assertIn('id="animate"', html)
        self.assertIn('id="atlasinfo"', html)
        self.assertNotIn('id="jitter"', html)
        self.assertNotIn("<datalist", html)

    def test_agents(self):
        data = self._get("/api/agents")
        self.assertEqual(len(data["agents"]), 500)
        a = data["agents"][0]
        self.assertEqual(len(a["attributes"]), 12)
        self.assertGreaterEqual(a["mult"], 1)
        for field in ("cluster_id", "copy_index", "copies"):
            self.assertIn(field, a)
        self.assertEqual(len({x["cluster_id"] for x in data["agents"]}), 302)

    def test_exact_copies_exposed(self):
        meta = self._get("/api/meta")
        s = meta["stats"]
        self.assertEqual(s["identities"], 500)
        self.assertEqual(s["distinct_profiles"], 302)
        self.assertEqual(s["duplicated_agents"], 227)
        self.assertEqual(s["exact_copies"], 198)
        self.assertEqual(s["max_mult"], max(d["count"] for d in meta["duplicates"]))
        d = meta["duplicates"][0]
        self.assertIn("cluster_id", d)
        self.assertEqual(len(d["agent_ids"]), d["count"])
        by_id = {x["id"]: x for x in self._get("/api/agents")["agents"]}
        self.assertTrue(all(by_id[i]["cluster_id"] == d["cluster_id"]
                            for i in d["agent_ids"]))

    def test_meta(self):
        m = self._get("/api/meta")
        self.assertEqual(m["stats"]["n"], 500)
        self.assertEqual(m["stats"]["distinct"], 302)
        self.assertAlmostEqual(m["stats"]["entropy"], 5.0222, places=3)
        self.assertEqual(len(m["attrs"]), 12)
        self.assertTrue(m["duplicates"])
        self.assertIn("qwen2.5:0.5b", m["suggestions"])
        self.assertEqual(m["labels"]["A1"], "sexe")
        self.assertEqual(m["labels"]["A1=v2"], "femme")
        self.assertIn("classe d'âge", m["labels"]["A9"])

    def test_chat_mock(self):
        agents = self._get("/api/agents")["agents"]
        status, data = self._post("/api/chat", {
            "agent_id": agents[0]["id"],
            "message": "Bonjour",
            "history": [],
            "provider": "mock",
        })
        self.assertEqual(status, 200)
        self.assertIn(agents[0]["name"].split()[0], data["answer"])
        self.assertEqual(data["provider"], "mock")

    def test_chat_stream_mock(self):
        agents = self._get("/api/agents")["agents"]
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/chat/stream",
            data=json.dumps({"agent_id": agents[0]["id"], "message": "Salut",
                             "provider": "mock"}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode()
        events = [json.loads(l) for l in raw.splitlines() if l.strip()]
        self.assertTrue(any(e["type"] == "token" for e in events))
        self.assertEqual(events[-1]["type"], "done")
        text = "".join(e["t"] for e in events if e["type"] == "token")
        self.assertIn("Salut", text)

    def test_chat_unknown_agent(self):
        status, data = self._post("/api/chat", {"agent_id": 99999, "message": "x"})
        self.assertEqual(status, 404)
        self.assertIn("inconnu", data["error"])

    def test_chat_empty_message(self):
        status, _ = self._post("/api/chat", {"agent_id": 1, "message": "  "})
        self.assertEqual(status, 400)

    def test_models(self):
        data = self._get("/api/models")
        self.assertIn("suggestions", data)
        self.assertIn("huihui_ai/qwen3-abliterated:4b", data["suggestions"])
        self.assertIsInstance(data["ollama"], list)

    def test_index_has_survey(self):
        html = self._get("/", raw=True).decode()
        self.assertIn('data-view="survey"', html)
        self.assertIn('id="svRun"', html)
        self.assertIn('id="svSurvey"', html)

    def test_surveys_list(self):
        data = self._get("/api/surveys")
        names = [s["name"] for s in data["surveys"]]
        self.assertIn("demo.fr", names)
        demo = next(s for s in data["surveys"] if s["name"] == "demo.fr")
        self.assertEqual(demo["n_questions"], 5)

    def test_survey_run_mock(self):
        status, data = self._post("/api/survey", {
            "survey": "demo.fr", "provider": "mock", "limit": 4, "seed": 0,
            "crosstab": ["A1"], "concurrency": 2})
        self.assertEqual(status, 200)
        job = data["job"]
        s = None
        for _ in range(200):
            s = self._get("/api/survey/status?id=" + job)
            if not s["running"]:
                break
            time.sleep(0.05)
        self.assertIsNone(s["error"])
        self.assertEqual(s["report"]["n_agents"], 4)
        q = next(x for x in s["report"]["questions"] if x["choices"])
        self.assertEqual(sum(q["counts"].values()), 4)
        self.assertIn("A1", q["crosstab"])

    def test_survey_unknown(self):
        status, data = self._post("/api/survey", {"survey": "../etc/passwd"})
        self.assertEqual(status, 400)
        self.assertIn("inconnu", data["error"])


if __name__ == "__main__":
    unittest.main()
