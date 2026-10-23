"""Tests de la GUI : serveur local, API JSON, chat mock.

Lancer :  uv run python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import threading
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
        cls.httpd = gui.create_server(ROOT / "agents" / "agents.jsonl", "127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

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

    def test_agents(self):
        data = self._get("/api/agents")
        self.assertEqual(len(data["agents"]), 500)
        a = data["agents"][0]
        self.assertEqual(len(a["attributes"]), 12)
        self.assertGreaterEqual(a["mult"], 1)

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


if __name__ == "__main__":
    unittest.main()
