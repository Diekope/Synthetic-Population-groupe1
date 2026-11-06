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

    def test_state_loads_social_affinity(self):
        run = Path(self.tmp.name) / "social" / "runs" / "affinity-run"
        run.mkdir(parents=True)
        (run / "state.json").write_text(json.dumps({
            "affinity": {"1:5": 0.73, "5:9": 0.2, "1:99999": 0.9,
                         "7:7": 1.0},
            "stats": {"conversations": 3, "refusals": 1},
        }), encoding="utf-8")
        state = gui.State(ROOT / "agents" / "agents.jsonl", self.surveys_dir,
                          run / "state.json")
        meta = state.meta()
        self.assertEqual(meta["affinity"], {"1:5": 0.73, "5:9": 0.2})
        self.assertTrue(meta["affinity_source"].endswith("state.json"))
        self.assertEqual(meta["affinity_stats"]["refusals"], 1)
        self.assertEqual(meta["affinity_default"], 0.5)

    def test_state_finds_latest_affinity_run_by_default(self):
        root = Path(self.tmp.name) / "auto"
        (root / "agents").mkdir(parents=True)
        shutil.copy(ROOT / "agents" / "agents.jsonl",
                    root / "agents" / "agents.jsonl")
        runs = root / "social" / "runs"
        older = runs / "older-affinity"
        newer = runs / "newer-affinity"
        empty = runs / "newest-empty"
        for p in (older, newer, empty):
            p.mkdir(parents=True)
        (older / "state.json").write_text(json.dumps(
            {"affinity": {"1:5": 0.41}}), encoding="utf-8")
        (newer / "state.json").write_text(json.dumps(
            {"affinity": {"1:5": 0.72}}), encoding="utf-8")
        (empty / "state.json").write_text(json.dumps(
            {"stats": {"conversations": 0}}), encoding="utf-8")
        import os
        os.utime(older / "state.json", (100, 100))
        os.utime(newer / "state.json", (200, 200))
        os.utime(empty / "state.json", (300, 300))
        state = gui.State(root / "agents" / "agents.jsonl", self.surveys_dir)
        self.assertEqual(state.affinity, {"1:5": 0.72})
        self.assertIn("newer-affinity", state.affinity_source)

    def test_state_loads_social_groups(self):
        run = Path(self.tmp.name) / "social" / "runs" / "group-run"
        run.mkdir(parents=True)
        (run / "state.json").write_text(json.dumps({
            "affinity": {"1:5": 0.7},
            "tick": 12,
            "conversation_no": 9,
            "opinions": {"1": {"teletravail": 0.72}, "99999": {"x": 0.9}},
            "groups": {
                "g001": {"members": [1, 5, 9], "cohesion": 0.66,
                         "conversations": 2, "created_tick": 4},
                "g002": {"members": [7, 99999], "cohesion": 0.4},
            },
        }), encoding="utf-8")
        state = gui.State(ROOT / "agents" / "agents.jsonl", self.surveys_dir,
                          run / "state.json")
        meta = state.meta()
        self.assertEqual(list(meta["groups"]), ["g001"])
        self.assertEqual(meta["groups"]["g001"]["members"], [1, 5, 9])
        self.assertEqual(meta["groups"]["g001"]["conversations"], 2)
        self.assertEqual(meta["social_tick"], 12)
        self.assertEqual(meta["conversation_no"], 9)
        self.assertEqual(meta["opinions"], {"1": {"teletravail": 0.72}})

    def test_social_reset_clears_displayed_state(self):
        run = Path(self.tmp.name) / "social" / "runs" / "reset-run"
        run.mkdir(parents=True)
        (run / "state.json").write_text(json.dumps({
            "affinity": {"1:5": 0.8},
            "groups": {"g001": {"members": [1, 5, 9], "cohesion": 0.6}},
            "opinions": {"1": {"teletravail": 0.7}},
            "tick": 7, "conversation_no": 4,
        }), encoding="utf-8")
        state = gui.State(ROOT / "agents" / "agents.jsonl", self.surveys_dir,
                          run / "state.json")
        self.assertTrue(state.meta()["affinity"])
        reset = state.social_reset()
        self.assertEqual(reset["affinity"], {})
        self.assertEqual(reset["groups"], {})
        self.assertEqual(reset["opinions"], {})
        meta = state.meta()
        self.assertEqual(meta["affinity"], {})
        self.assertEqual(meta["groups"], {})
        self.assertEqual(meta["opinions"], {})
        self.assertIsNone(meta["affinity_source"])
        self.assertEqual(meta["social_tick"], 0)
        self.assertEqual(meta["conversation_no"], 0)

    def test_sim_export_returns_state_file(self):
        run_id = "unittest-gui-export"
        run_dir = ROOT / "social" / "runs" / run_id
        try:
            run_dir.mkdir(parents=True)
            (run_dir / "state.json").write_text(json.dumps({
                "affinity": {}, "positions": {}, "groups": {}, "opinions": {},
                "tick": 0, "conversation_no": 0,
                "stats": {"conversations": 0, "refusals": 0},
            }), encoding="utf-8")
            status, _ = self._post("/api/sim/load", {"run_id": run_id})
            self.assertEqual(status, 200)
            raw = self._get("/api/sim/export", raw=True)
            data = json.loads(raw)
            self.assertIn("positions", data)
            self.assertEqual(data["stats"]["conversations"], 0)
        finally:
            self._post("/api/runs/delete", {"run_id": run_id})
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_run_list_load_resume_delete(self):
        run_id = "unittest-gui-runs"
        run_dir = ROOT / "social" / "runs" / run_id
        try:
            status, data = self._post("/api/sim/start", {
                "provider": "mock", "model": "mock", "max_ticks": 2,
                "tick_seconds": 0, "spark_rate": 1, "group_rate": 0,
                "refusal_rate": 0, "limit": 6, "run_id": run_id,
                "reset": True,
            })
            self.assertEqual(status, 200, data)
            deadline = time.time() + 30
            while time.time() < deadline:
                state = self._get("/api/sim/status")
                if not state["running"]:
                    break
                time.sleep(0.2)
            runs = self._get("/api/runs")["runs"]
            entry = next(r for r in runs if r["run_id"] == run_id)
            self.assertEqual(entry["tick"], 2)
            self.assertEqual(entry["conversations"], 2)

            status, loaded = self._post("/api/sim/load", {"run_id": run_id})
            self.assertEqual(status, 200)
            self.assertEqual(loaded["run_id"], run_id)
            self.assertEqual(loaded["tick"], 2)
            self.assertIn("positions", loaded)

            status, resumed = self._post("/api/sim/start", {
                "provider": "mock", "model": "mock", "max_ticks": 1,
                "tick_seconds": 0, "spark_rate": 1, "group_rate": 0,
                "refusal_rate": 0, "limit": 6, "run_id": run_id,
                "resume": run_id, "reset": False,
            })
            self.assertEqual(status, 200, resumed)
            deadline = time.time() + 30
            while time.time() < deadline:
                state = self._get("/api/sim/status")
                if not state["running"]:
                    break
                time.sleep(0.2)
            self.assertEqual(state["tick"], 3)
            self.assertEqual(state["stats"]["conversations"], 3)

            status, listed = self._post("/api/runs/delete", {"run_id": run_id})
            self.assertEqual(status, 200)
            self.assertNotIn(run_id, [r["run_id"] for r in listed["runs"]])
            self.assertFalse(run_dir.exists())
        finally:
            self._post("/api/sim/stop", {})
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_index_has_group_and_opinion_panels(self):
        html = self._get("/", raw=True).decode()
        for marker in ('id="group"', 'id="opinions"', "renderGroup",
                       "renderOpinions", "groupColor"):
            self.assertIn(marker, html)

    def test_meta_exposes_topic_labels(self):
        meta = self._get("/api/meta")
        self.assertIn("teletravail", meta["topic_labels"])
        self.assertIn("opinions", meta)

    def test_index_has_affinity_hover(self):
        html = self._get("/", raw=True).decode()
        self.assertIn("affinité avec", html)
        self.assertIn("renderAffinity", html)
        self.assertIn('id="affinity"', html)
        self.assertIn('id="affstat"', html)

    def test_index_has_simulation_controls(self):
        html = self._get("/", raw=True).decode()
        for control in ('id="simstart"', 'id="simstop"', 'id="simticks"',
                        'id="simspeed"', 'id="simlimit"', 'id="simfresh"',
                        'id="simreset"', 'id="simburst"', 'id="simpar"',
                        'id="simpause"',
                        'id="simruns"', 'id="simload"', 'id="simresume"',
                        'id="simdelete"', 'id="simexport"',
                        "/api/sim/start", "/api/sim/reset", "/api/runs/delete",
                        "/api/sim/pause", "/api/sim/resume"):
            self.assertIn(control, html)
        self.assertIn('placeholder="tous"', html)

    def test_sim_start_and_finish(self):
        run_id = "unittest-gui-sim"
        run_dir = ROOT / "social" / "runs" / run_id
        try:
            status, data = self._post("/api/sim/start", {
                "provider": "mock", "model": "mock", "max_ticks": 3,
                "tick_seconds": 0, "spark_rate": 1, "group_rate": 0,
                "refusal_rate": 0, "limit": 6, "run_id": run_id,
                "reset": True,
            })
            self.assertEqual(status, 200, data)
            self.assertTrue(data["running"])
            deadline = time.time() + 30
            while time.time() < deadline:
                state = self._get("/api/sim/status")
                if not state["running"]:
                    break
                time.sleep(0.2)
            self.assertFalse(state["running"])
            self.assertEqual(state["run_id"], run_id)
            self.assertEqual(state["tick"], 3)
            self.assertEqual(state["stats"]["conversations"], 3)
            self.assertEqual(len(state["events"]), 3)
            self.assertTrue(state["affinity"])
            self.assertIsNone(state["error"])
            meta = self._get("/api/meta")
            self.assertEqual(meta["social_tick"], 3)
            self.assertTrue(meta["affinity"])
        finally:
            self._post("/api/sim/stop", {})
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_sim_runs_on_every_agent_by_default(self):
        run_id = "unittest-gui-sim-all"
        run_dir = ROOT / "social" / "runs" / run_id
        try:
            status, data = self._post("/api/sim/start", {
                "provider": "mock", "model": "mock", "max_ticks": 2,
                "tick_seconds": 0, "spark_rate": 0, "group_rate": 0,
                "refusal_rate": 0, "run_id": run_id, "reset": True,
            })
            self.assertEqual(status, 200, data)
            self.assertIsNone(data["config"]["limit"])
            deadline = time.time() + 30
            while time.time() < deadline:
                state = self._get("/api/sim/status")
                if not state["running"]:
                    break
                time.sleep(0.2)
            self.assertFalse(state["running"])
            self.assertEqual(len(state["positions"]), 500)
        finally:
            self._post("/api/sim/stop", {})
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_sim_pause_and_resume(self):
        run_id = "unittest-gui-pause"
        run_dir = ROOT / "social" / "runs" / run_id
        try:
            status, data = self._post("/api/sim/start", {
                "provider": "mock", "model": "mock", "max_ticks": 5000,
                "tick_seconds": 0.05, "spark_rate": 1, "group_rate": 0,
                "refusal_rate": 0, "limit": 6, "run_id": run_id,
                "reset": True,
            })
            self.assertEqual(status, 200, data)
            time.sleep(0.3)
            status, paused = self._post("/api/sim/pause", {})
            self.assertEqual(status, 200, paused)
            self.assertTrue(paused["running"])
            self.assertTrue(paused["paused"])
            time.sleep(0.3)
            still = self._get("/api/sim/status")
            self.assertTrue(still["paused"])
            status, resumed = self._post("/api/sim/resume", {})
            self.assertEqual(status, 200, resumed)
            self.assertFalse(resumed["paused"])
            self.assertTrue(resumed["running"])
            status, stopped = self._post("/api/sim/stop", {})
            self.assertFalse(stopped["running"])
            self.assertFalse(stopped["paused"])
        finally:
            self._post("/api/sim/stop", {})
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_sim_stop_interrupts_a_long_run(self):
        run_id = "unittest-gui-stop"
        run_dir = ROOT / "social" / "runs" / run_id
        try:
            status, data = self._post("/api/sim/start", {
                "provider": "mock", "model": "mock", "max_ticks": 5000,
                "tick_seconds": 0.2, "spark_rate": 0.5, "group_rate": 0,
                "refusal_rate": 0, "limit": 6, "run_id": run_id,
                "reset": True,
            })
            self.assertEqual(status, 200, data)
            time.sleep(0.5)
            status, stopped = self._post("/api/sim/stop", {})
            self.assertEqual(status, 200, stopped)
            self.assertFalse(stopped["running"])
            self.assertLess(stopped["tick"], 5000)
        finally:
            self._post("/api/sim/stop", {})
            shutil.rmtree(run_dir, ignore_errors=True)

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
