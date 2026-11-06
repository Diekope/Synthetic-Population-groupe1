#!/usr/bin/env python3
"""GUI locale légère : visualise la population et discute avec un agent.

Zéro dépendance (stdlib only). Sert une page HTML + une petite API JSON.

  uv run python tools/gui.py                     # http://127.0.0.1:8765
  uv run python tools/gui.py --open              # ouvre le navigateur
  uv run python tools/gui.py --port 0            # port libre choisi par l'OS
  uv run python tools/gui.py --provider ollama --model llama3.2:3b

L'interface permet de :
  - voir les 500 agents, leurs attributs, les doublons et les distributions ;
  - filtrer / chercher un agent ;
  - tracer deux attributs et colorer les points par multiplicité de persona ;
  - discuter avec l'agent sélectionné (mock, OpenAI ou Ollama).

Sécurité : écoute sur 127.0.0.1 par défaut, aucune authentification. Ne l'expose
pas sur un réseau public.
"""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_agent  # noqa: E402
import survey as survey_mod  # noqa: E402

MODEL_SUGGESTIONS = [
    "qwen3:0.6b",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "llama3.2:1b",
    "llama3.2:3b",
    "huihui_ai/qwen3-abliterated:4b",
    "huihui_ai/llama3.2-abliterate:3b",
    "gemma3:1b",
    "phi3:mini",
]


class State:
    def __init__(self, agents_path: Path, surveys_dir: Path | None = None,
                 social_state: Path | None = None):
        self.agents = chat_agent.load_agents(agents_path)
        self.agents_path = Path(agents_path).resolve()
        self.root = self.agents_path.parent.parent
        self.surveys_dir = surveys_dir or (self.root / "surveys")
        self.jobs: dict[str, dict] = {}
        self.jobs_lock = threading.Lock()
        self.by_id = {a["id"]: a for a in self.agents}
        self.affinity, self.affinity_source, self.affinity_stats, self.groups = \
            self._load_affinity(agents_path, social_state)
        # Simulation lancée depuis la GUI (sous-processus).
        self.sim_lock = threading.RLock()
        self.sim_proc: subprocess.Popen | None = None
        self.sim_log_file = None
        self.sim_run_id: str | None = None
        self.sim_run_dir: Path | None = None
        self.sim_error: str | None = None
        self.sim_config: dict = {}
        self.topic_labels = self._load_topic_labels()
        self.attrs = [a for a in self.agents[0]["attributes"] if a.startswith("A")]
        self.domains = {a: sorted({ag["attributes"][a] for ag in self.agents}) for a in self.attrs}
        self.counts = Counter(tuple(ag["attributes"][a] for a in self.attrs) for ag in self.agents)
        self.mult = {ag["id"]: self.counts[tuple(ag["attributes"][a] for a in self.attrs)]
                     for ag in self.agents}
        n = len(self.agents)
        self.entropy = -sum((m / n) * math.log(m / n) for m in self.counts.values())
        self.max_mult = max(self.counts.values())
        self.distinct = len(self.counts)
        self.distributions = {
            a: dict(Counter(ag["attributes"][a] for ag in self.agents)) for a in self.attrs
        }
        self.cluster_of = {
            tuple(ag["attributes"][a] for a in self.attrs): ag["cluster_id"]
            for ag in self.agents
        }
        reps = {}
        members: dict[tuple[str, ...], list[int]] = {}
        for ag in self.agents:
            key = tuple(ag["attributes"][a] for a in self.attrs)
            reps.setdefault(key, ag["id"])
            members.setdefault(key, []).append(ag["id"])
        self.duplicates = sorted(
            ({"key": "|".join(k), "cluster_id": self.cluster_of[k], "count": c,
              "agent_id": reps[k], "agent_ids": members[k],
              "attrs": dict(zip(self.attrs, k))} for k, c in self.counts.items() if c > 1),
            key=lambda d: -d["count"])
        labels_path = agents_path.parent / "labels.json"
        self.labels = {}
        if labels_path.exists():
            try:
                self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
            except ValueError:
                pass

    # ----- affinité sociale ---------------------------------------------

    def _load_topic_labels(self) -> dict[str, str]:
        path = self.root / "tools" / "topics.fr.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        raw = data.get("topics") if isinstance(data, dict) else data
        labels = {}
        for topic in raw or []:
            if isinstance(topic, dict) and topic.get("id") and topic.get("text"):
                labels[str(topic["id"])] = str(topic["text"])
        return labels

    @staticmethod
    def _social_state_file(agents_path: Path,
                           explicit: Path | None) -> tuple[Path | None, dict]:
        """Trouve le state.json social : explicite, sinon le plus récent.

        On privilégie un état qui contient déjà des affinités ; à défaut on
        garde le state le plus récent même sans affinité.
        """
        if explicit:
            path = Path(explicit)
            if path.is_dir():
                path = path / "state.json"
            if path.exists():
                try:
                    return path, json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    return path, {}
            return None, {}

        runs = agents_path.parent.parent / "social" / "runs"
        candidates = sorted(runs.glob("*/state.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        fallback: tuple[Path | None, dict] = (None, {})
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if data.get("affinity"):
                return path, data
            if fallback[0] is None:
                fallback = (path, data)
        return fallback

    def _normalise_affinity(self, raw) -> dict[str, float]:
        affinity: dict[str, float] = {}
        for key, value in (raw or {}).items():
            try:
                aid, bid = (int(x) for x in str(key).split(":", 1))
                score = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
            if aid in self.by_id and bid in self.by_id and aid != bid:
                affinity[f"{min(aid, bid)}:{max(aid, bid)}"] = round(score, 4)
        return affinity

    def _normalise_groups(self, raw) -> dict[str, dict]:
        groups: dict[str, dict] = {}
        for gid, entry in (raw or {}).items():
            if not isinstance(entry, dict):
                continue
            members = sorted({int(m) for m in entry.get("members") or []
                              if str(m).isdigit() and int(m) in self.by_id})
            if len(members) < 2:
                continue
            try:
                cohesion = float(entry.get("cohesion") or 0)
            except (TypeError, ValueError):
                cohesion = 0.0
            groups[str(gid)] = {
                "id": str(gid),
                "members": members,
                "cohesion": round(max(0.0, min(1.0, cohesion)), 4),
                "conversations": int(entry.get("conversations") or 0),
                "created_tick": int(entry.get("created_tick") or 0),
                "last_active_tick": int(entry.get("last_active_tick") or 0),
            }
        return groups

    def _normalise_opinions(self, raw) -> dict[int, dict[str, float]]:
        opinions: dict[int, dict[str, float]] = {}
        for key, values in (raw or {}).items():
            try:
                aid = int(key)
            except (TypeError, ValueError):
                continue
            if aid not in self.by_id or not isinstance(values, dict):
                continue
            clean = {}
            for tid, value in values.items():
                try:
                    clean[str(tid)] = round(max(0.0, min(1.0, float(value))), 4)
                except (TypeError, ValueError):
                    continue
            if clean:
                opinions[aid] = clean
        return opinions

    def _load_affinity(self, agents_path: Path, explicit: Path | None
                       ) -> tuple[dict, str | None, dict, dict]:
        path, data = self._social_state_file(agents_path, explicit)
        affinity = self._normalise_affinity(data.get("affinity"))
        groups = self._normalise_groups(data.get("groups"))
        self.opinions = self._normalise_opinions(data.get("opinions"))
        self.social_tick = int(data.get("tick") or 0)
        self.conversation_no = int(data.get("conversation_no") or 0)
        source = str(path) if path else None
        return affinity, source, data.get("stats") or {}, groups

    def meta(self) -> dict:
        return {
            "stats": {
                "n": len(self.agents),
                "identities": len(self.agents),
                "distinct": self.distinct,
                "distinct_profiles": self.distinct,
                "entropy": self.entropy,
                "effective": math.exp(self.entropy),
                "max_mult": self.max_mult,
                "extra_copies": len(self.agents) - self.distinct,
                "duplicated_agents": sum(1 for m in self.mult.values() if m > 1),
                "exact_copies": len(self.agents) - self.distinct,
            },
            "attrs": self.attrs,
            "domains": self.domains,
            "distributions": self.distributions,
            "duplicates": self.duplicates,
            "suggestions": MODEL_SUGGESTIONS,
            "labels": self.labels,
            "affinity": self.affinity,
            "affinity_source": self.affinity_source,
            "affinity_stats": self.affinity_stats,
            "affinity_default": 0.5,
            "groups": self.groups,
            "social_tick": self.social_tick,
            "conversation_no": self.conversation_no,
            "opinions": {str(aid): values
                         for aid, values in sorted(self.opinions.items())},
            "topic_labels": self.topic_labels,
        }

    # ----- simulation sociale en direct --------------------------------

    @staticmethod
    def _tail(path: Path, nbytes: int = 65536) -> list[str]:
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - nbytes))
                return handle.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return []

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        for _ in range(3):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (ValueError, OSError):
                time.sleep(0.02)
        return None

    def _sim_close_log(self) -> None:
        if self.sim_log_file is not None:
            try:
                self.sim_log_file.close()
            except OSError:
                pass
            self.sim_log_file = None

    def _sim_events(self) -> list[dict]:
        if self.sim_run_dir is None:
            return []
        path = self.sim_run_dir / "events.jsonl"
        events = []
        for line in self._tail(path)[-40:]:
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            topic = raw.get("topic") or {}
            group = raw.get("group") or {}
            events.append({
                "type": raw.get("type"),
                "tick": raw.get("tick"),
                "id": raw.get("conversation_id"),
                "mode": raw.get("mode"),
                "topic": topic.get("text"),
                "names": [p.get("name") for p in raw.get("participants") or []],
                "group": group.get("id"),
                "affinity_before": raw.get("affinity_before"),
                "affinity_after": raw.get("affinity_after"),
                "reason": raw.get("reason"),
            })
        return events[-10:]

    def sim_status(self) -> dict:
        with self.sim_lock:
            proc = self.sim_proc
            running = proc is not None and proc.poll() is None
            data = None
            if self.sim_run_dir is not None:
                data = self._read_json(self.sim_run_dir / "state.json")
            if data:
                self.affinity = self._normalise_affinity(data.get("affinity"))
                self.groups = self._normalise_groups(data.get("groups"))
                self.opinions = self._normalise_opinions(data.get("opinions"))
                self.affinity_stats = data.get("stats") or self.affinity_stats
                self.social_tick = int(data.get("tick") or 0)
                self.conversation_no = int(data.get("conversation_no") or 0)
                if self.sim_run_dir is not None:
                    self.affinity_source = str(self.sim_run_dir / "state.json")

            error = self.sim_error
            if not running and proc is not None and proc.returncode not in (0, None):
                self._sim_close_log()
                tail = [line for line in self._tail(
                    self.sim_run_dir / "gui.log", 4096) if line.strip()]
                error = "\n".join(tail[-6:]) or f"code de sortie {proc.returncode}"
                self.sim_error = error
            elif not running:
                self._sim_close_log()

            memory_pairs = len((data or {}).get("memory") or {})
            return {
                "running": running,
                "run_id": self.sim_run_id,
                "run_dir": str(self.sim_run_dir) if self.sim_run_dir else None,
                "tick": self.social_tick,
                "conversation_no": self.conversation_no,
                "stats": self.affinity_stats,
                "affinity": self.affinity,
                "groups": self.groups,
                "opinions": {str(aid): values
                             for aid, values in sorted(self.opinions.items())},
                "memory_pairs": memory_pairs,
                "positions": (data or {}).get("positions") or {},
                "events": self._sim_events(),
                "error": error,
                "config": self.sim_config,
            }

    def sim_start(self, body: dict) -> dict:
        with self.sim_lock:
            if self.sim_proc is not None and self.sim_proc.poll() is None:
                raise ValueError("une simulation tourne déjà")
            provider = str(body.get("provider") or "mock")
            if provider not in ("mock", "openai", "ollama"):
                raise ValueError("fournisseur invalide")
            default_model = {"mock": "mock", "ollama": "llama3.2:1b",
                             "openai": "gpt-4o-mini"}[provider]
            model = str(body.get("model") or default_model).strip() or default_model
            try:
                max_ticks = max(1, min(5000, int(body.get("max_ticks") or 20)))
                tick_seconds = max(0.0, min(60.0,
                                            float(body.get("tick_seconds", 0.5))))
                spark = max(0.0, min(1.0, float(body.get("spark_rate", 0.5))))
                group_rate = max(0.0, min(1.0, float(body.get("group_rate", 0.35))))
                refusal = max(0.0, min(1.0, float(body.get("refusal_rate", 0.1))))
                burst = max(1, min(64, int(body.get("conversations_per_tick") or 1)))
                parallel = max(1, min(32, int(body.get("parallel") or 1)))
                limit = body.get("limit")
                limit = max(2, min(len(self.agents), int(limit))) if limit else None
            except (TypeError, ValueError):
                raise ValueError("paramètres numériques invalides") from None

            raw_id = str(body.get("run_id") or "").strip()
            resume = str(body.get("resume") or "").strip()
            run_id = resume or raw_id or time.strftime("gui-%Y%m%d-%H%M%S")
            safe = "".join(c if (c.isalnum() or c in "-_.") else "-"
                           for c in run_id)[:80]
            if not safe:
                raise ValueError("identifiant de run invalide")
            runs_dir = self.root / "social" / "runs"
            run_dir = runs_dir / safe
            if resume and not (run_dir / "state.json").exists():
                raise ValueError(f"run à reprendre introuvable : {safe}")
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(self.root / "tools" / "social_sim.py"),
                "--agents", str(self.agents_path),
                "--out", str(runs_dir), "--run-id", safe,
                "--provider", provider, "--model", model,
                "--max-ticks", str(max_ticks),
                "--tick-seconds", str(tick_seconds),
                "--spark-rate", str(spark),
                "--group-rate", str(group_rate),
                "--refusal-rate", str(refusal),
                "--conversations-per-tick", str(burst),
                "--parallel", str(parallel),
                "--state-every", "1", "--no-stream", "--quiet",
            ]
            if limit:
                cmd += ["--limit", str(limit)]
            if resume:
                cmd += ["--resume", str(run_dir)]
            elif body.get("reset", True):
                cmd += ["--reset"]
            log = (run_dir / "gui.log").open(
                "a" if resume else "w", encoding="utf-8")
            try:
                self.sim_proc = subprocess.Popen(
                    cmd, cwd=str(self.root), stdout=log,
                    stderr=subprocess.STDOUT)
            except OSError as exc:
                log.close()
                raise ValueError(f"lancement impossible : {exc}") from None
            self.sim_log_file = log
            self.sim_run_id = safe
            self.sim_run_dir = run_dir
            self.sim_error = None
            self.sim_config = {
                "provider": provider, "model": model, "max_ticks": max_ticks,
                "tick_seconds": tick_seconds, "spark_rate": spark,
                "group_rate": group_rate, "refusal_rate": refusal,
                "conversations_per_tick": burst, "parallel": parallel,
                "limit": limit, "run_id": safe,
                "resumed_from": safe if resume else None,
            }
            self.affinity_stats = {}
            self.social_tick = 0
            self.conversation_no = 0
            self.affinity = {}
            self.groups = {}
            self.opinions = {}
            self.affinity_source = None
            return self.sim_status()

    def sim_stop(self) -> dict:
        with self.sim_lock:
            proc = self.sim_proc
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            return self.sim_status()

    def social_reset(self) -> dict:
        """Efface l'état social affiché (affinités, groupes, opinions).

        L'état est vidé en mémoire ; les fichiers du run restent sur disque et
        peuvent être rechargés en relançant la GUI.
        """
        with self.sim_lock:
            proc = self.sim_proc
            if proc is not None and proc.poll() is None:
                raise ValueError("arrête la simulation avant de réinitialiser")
            self.affinity = {}
            self.groups = {}
            self.opinions = {}
            self.affinity_stats = {}
            self.affinity_source = None
            self.social_tick = 0
            self.conversation_no = 0
            self.sim_run_id = None
            self.sim_run_dir = None
            self.sim_config = {}
            self.sim_error = None
            return {
                "affinity": {},
                "groups": {},
                "opinions": {},
                "affinity_stats": {},
                "affinity_source": None,
                "social_tick": 0,
                "conversation_no": 0,
            }

    def _run_dir(self, run_id: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "-"
                       for c in str(run_id))[:80]
        if not safe:
            raise ValueError("identifiant de run invalide")
        return self.root / "social" / "runs" / safe

    def run_list(self) -> dict:
        runs_dir = self.root / "social" / "runs"
        runs = []
        for path in runs_dir.glob("*/state.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                mtime = path.stat().st_mtime
            except (OSError, ValueError):
                continue
            stats = data.get("stats") or {}
            runs.append({
                "run_id": path.parent.name,
                "tick": int(data.get("tick") or 0),
                "conversations": int(stats.get("conversations") or 0),
                "groups": len(data.get("groups") or {}),
                "opinions": len(data.get("opinions") or {}),
                "mtime": round(mtime, 3),
            })
        runs.sort(key=lambda r: r["mtime"], reverse=True)
        return {"runs": runs}

    def load_run(self, run_id: str) -> dict:
        with self.sim_lock:
            if self.sim_proc is not None and self.sim_proc.poll() is None:
                raise ValueError("arrête la simulation avant de charger un run")
            run_dir = self._run_dir(run_id)
            state_path = run_dir / "state.json"
            data = self._read_json(state_path)
            if not data:
                raise ValueError(f"aucun state.json dans {run_dir.name}")
            self.affinity = self._normalise_affinity(data.get("affinity"))
            self.groups = self._normalise_groups(data.get("groups"))
            self.opinions = self._normalise_opinions(data.get("opinions"))
            self.affinity_stats = data.get("stats") or {}
            self.affinity_source = str(state_path)
            self.social_tick = int(data.get("tick") or 0)
            self.conversation_no = int(data.get("conversation_no") or 0)
            self.sim_run_id = run_dir.name
            self.sim_run_dir = run_dir
            self.sim_config = {}
            self.sim_error = None
            return self.sim_status()

    def delete_run(self, run_id: str) -> dict:
        with self.sim_lock:
            if self.sim_proc is not None and self.sim_proc.poll() is None:
                raise ValueError("arrête la simulation avant de supprimer un run")
            run_dir = self._run_dir(run_id)
            if not run_dir.is_dir():
                raise ValueError(f"run introuvable : {run_dir.name}")
            shutil.rmtree(run_dir)
            if self.sim_run_dir == run_dir:
                self.social_reset()
            return self.run_list()

    def export_state(self) -> Path | None:
        if self.sim_run_dir is not None:
            path = self.sim_run_dir / "state.json"
            if path.exists():
                return path
        if self.affinity_source:
            path = Path(self.affinity_source)
            if path.exists():
                return path
        return None

    def agent_list(self) -> dict:
        return {"agents": [{
            "id": ag["id"],
            "identity_key": ag["identity_key"],
            "name": f"{ag['prenom']} {ag['nom']}",
            "cluster_id": ag.get("cluster_id"),
            "copy_index": ag.get("copy_index", 1),
            "copies": ag.get("copies", 1),
            "attributes": {a: ag["attributes"][a] for a in self.attrs},
            "mult": self.mult[ag["id"]],
        } for ag in self.agents]}

    # ----- sondage -------------------------------------------------------

    def list_surveys(self) -> list[dict]:
        out = []
        if self.surveys_dir.exists():
            for p in sorted(self.surveys_dir.glob("*.json")):
                if p.name.endswith(".report.json") or p.name.endswith(".answers.jsonl"):
                    continue
                try:
                    s = survey_mod.load_survey(p)
                except (SystemExit, ValueError, KeyError):
                    continue
                out.append({"name": p.stem, "title": s["title"],
                            "n_questions": len(s["questions"]),
                            "questions": [{"id": q["id"], "text": q["text"],
                                           "choices": q["choices"]} for q in s["questions"]]})
        return out

    def _survey_path(self, name: str) -> Path:
        safe = Path(name).name
        path = self.surveys_dir / f"{safe}.json"
        if not path.exists():
            raise ValueError(f"questionnaire inconnu : {name}")
        return path

    def start_survey(self, body: dict) -> str:
        path = self._survey_path(body.get("survey") or "")
        survey = survey_mod.load_survey(path)
        where = [w for w in (body.get("where") or []) if w]
        ids = body.get("ids")
        selected = survey_mod.select_agents(
            self.agents, where=where, ids=ids,
            one_per_cluster=bool(body.get("one_per_cluster")),
            limit=body.get("limit"), seed=int(body.get("seed") or 0))
        if not selected:
            raise ValueError("aucun agent sélectionné")
        provider = body.get("provider") or "mock"
        model = (body.get("model") or "").strip() or {
            "openai": "gpt-4o-mini", "ollama": "llama3.2:1b"}.get(provider, "mock")
        concurrency = max(1, min(4, int(body.get("concurrency") or 2)))
        crosstab = [a for a in (body.get("crosstab") or []) if a]
        safe_model = "".join(c if c.isalnum() or c in "._-" else "_" for c in model)
        out = self.surveys_dir / f"{path.stem}.{provider}.{safe_model}.answers.jsonl"
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "running": True, "done": 0, "total": 0,
               "error": None, "report": None, "n_agents": len(selected),
               "title": survey["title"], "provider": provider, "model": model,
               "out": str(out)}
        with self.jobs_lock:
            self.jobs[job_id] = job

        def progress(done, total):
            job["done"], job["total"] = done, total

        def worker():
            try:
                answers = survey_mod.run_survey(
                    selected, survey, provider=provider, model=model, out=out,
                    concurrency=concurrency, temperature=float(body.get("temperature") or 0.8),
                    shuffle=bool(body.get("shuffle")), progress=False, on_progress=progress,
                    force=True)
                job["report"] = survey_mod.aggregate(survey, answers, selected, crosstab)
            except Exception as e:  # noqa: BLE001
                job["error"] = f"{type(e).__name__}: {e}"
            finally:
                job["running"] = False

        threading.Thread(target=worker, daemon=True).start()
        return job_id

    def job_status(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job


def ollama_models() -> list[dict]:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5) as r:
            data = json.load(r)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return []
    out = []
    for m in data.get("models", []):
        name = m.get("name", "?")
        # Les modèles d'embedding ne servent pas à discuter.
        if any(h in name.lower() for h in ("embed", "minilm", "bge", "nomic", "e5")):
            continue
        out.append({"name": name, "size_gb": round(m.get("size", 0) / 1e9, 2)})
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "SyntheticPopGUI/1.0"

    def log_message(self, fmt, *args):  # silence les logs par requête
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def _error(self, msg: str, status: int = 400):
        self._json({"error": msg}, status)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(INDEX.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/agents":
            self._json(self.server.state.agent_list())
        elif self.path == "/api/meta":
            self._json(self.server.state.meta())
        elif self.path == "/api/sim/status":
            self._json(self.server.state.sim_status())
        elif self.path == "/api/runs":
            self._json(self.server.state.run_list())
        elif self.path == "/api/sim/export":
            path = self.server.state.export_state()
            if path is None:
                return self._error("aucun état à exporter", 404)
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Disposition",
                             f'attachment; filename="{path.parent.name}-state.json"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/surveys":
            self._json({"surveys": self.server.state.list_surveys()})
        elif self.path.startswith("/api/survey/status"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            job_id = (q.get("id") or [""])[0]
            try:
                job = self.server.state.job_status(job_id)
            except KeyError:
                return self._error("sonde inconnue", 404)
            self._json(job)
        elif self.path == "/api/models":
            models = ollama_models()
            self._json({"ollama": models, "suggestions": MODEL_SUGGESTIONS,
                        "ollama_up": bool(models)})
        elif self.path == "/api/health":
            import os
            self._json({"ok": True, "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
                        "ollama_up": bool(ollama_models())})
        elif self.path == "/favicon.ico":
            self._send(b"", "image/x-icon", 204)
        else:
            self._error("not found", 404)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("payload trop gros")
        return json.loads(self.rfile.read(length) or b"{}")

    def _chat_request(self, body):
        """Valide la requête de chat et renvoie (agent, history, provider, model)."""
        agent = self.server.state.by_id.get(body.get("agent_id"))
        if agent is None:
            raise LookupError("agent inconnu")
        message = (body.get("message") or "").strip()
        if not message:
            raise ValueError("message vide")
        provider = body.get("provider") or "mock"
        if provider not in ("mock", "openai", "ollama"):
            raise ValueError("fournisseur invalide")
        model = (body.get("model") or "").strip() or (
            {"openai": "gpt-4o-mini", "ollama": "llama3.2:1b"}.get(provider, "mock"))
        history = []
        for m in (body.get("history") or [])[-40:]:
            if (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content"), str)):
                history.append({"role": m["role"], "content": m["content"][:4000]})
        history.append({"role": "user", "content": message[:4000]})
        budget = int(body.get("budget_tokens") or 3000)
        history = chat_agent.trim_history(
            history, budget, system_tokens=chat_agent.estimate_tokens(agent["persona"]))
        return agent, history, provider, model, body

    def do_POST(self):
        if self.path in ("/api/sim/start", "/api/sim/stop", "/api/sim/reset",
                         "/api/sim/load", "/api/runs/delete"):
            try:
                body = self._read_body() if not self.path.endswith("stop") else {}
                if self.path.endswith("start"):
                    payload = self.server.state.sim_start(body)
                elif self.path.endswith("stop"):
                    payload = self.server.state.sim_stop()
                elif self.path.endswith("reset"):
                    payload = self.server.state.social_reset()
                elif self.path.endswith("load"):
                    payload = self.server.state.load_run(body.get("run_id", ""))
                else:
                    payload = self.server.state.delete_run(body.get("run_id", ""))
            except ValueError as exc:
                return self._error(str(exc))
            except OSError as exc:
                return self._error(str(exc), 500)
            return self._json(payload)
        if self.path == "/api/survey":
            return self._start_survey()
        if self.path == "/api/chat/stream":
            return self._chat_stream()
        if self.path != "/api/chat":
            return self._error("not found", 404)
        try:
            body = self._read_body()
        except (ValueError, OSError) as e:
            return self._error(str(e) if str(e) == "payload trop gros" else "JSON invalide",
                               413 if str(e) == "payload trop gros" else 400)
        try:
            agent, history, provider, model, body = self._chat_request(body)
        except LookupError as e:
            return self._error(str(e), 404)
        except ValueError as e:
            return self._error(str(e))

        t0 = time.perf_counter()
        try:
            answer = chat_agent.respond(
                agent, history, provider, model,
                temperature=float(body.get("temperature") or 0.8),
                max_tokens=body.get("max_tokens"), seed=body.get("seed"))
        except chat_agent.LLMError as e:
            return self._error(str(e), 502)
        except Exception as e:  # noqa: BLE001 — la GUI ne doit jamais tomber
            return self._error(f"{type(e).__name__}: {e}", 500)
        ms = round((time.perf_counter() - t0) * 1000)
        self._json({"answer": answer, "ms": ms, "provider": provider, "model": model})

    def _chat_stream(self):
        try:
            body = self._read_body()
        except (ValueError, OSError):
            return self._error("JSON invalide")
        try:
            agent, history, provider, model, body = self._chat_request(body)
        except LookupError as e:
            return self._error(str(e), 404)
        except ValueError as e:
            return self._error(str(e))

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def emit(obj):
            try:
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        t0 = time.perf_counter()
        try:
            if provider == "mock":
                answer = chat_agent.respond(agent, history, provider, model)
                emit({"type": "token", "t": answer})
            else:
                chat_agent.respond(
                    agent, history, provider, model,
                    temperature=float(body.get("temperature") or 0.8),
                    max_tokens=body.get("max_tokens"), seed=body.get("seed"),
                    stream=True, on_token=lambda piece: emit({"type": "token", "t": piece}))
            emit({"type": "done", "ms": round((time.perf_counter() - t0) * 1000),
                  "provider": provider, "model": model})
        except chat_agent.LLMError as e:
            emit({"type": "error", "error": str(e)})
        except Exception as e:  # noqa: BLE001
            emit({"type": "error", "error": f"{type(e).__name__}: {e}"})

    def _start_survey(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                return self._error("payload trop gros", 413)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            return self._error("JSON invalide")
        try:
            job_id = self.server.state.start_survey(body)
        except ValueError as e:
            return self._error(str(e))
        except Exception as e:  # noqa: BLE001
            return self._error(f"{type(e).__name__}: {e}", 500)
        self._json({"job": job_id})


def create_server(agents: Path, host: str = "127.0.0.1", port: int = 8765,
                  surveys: Path | None = None,
                  social_state: Path | None = None) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.state = State(agents, surveys, social_state)
    return httpd


INDEX = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Population synthétique — agents</title>
<style>
  :root{
    --paper:#f4f1ea; --ink:#1d2329; --muted:#6b7280; --line:#d9d3c7;
    --card:#fffdf8; --accent:#c2410c; --accent2:#0f766e; --shadow:0 1px 2px #0001,0 8px 24px #0000000a;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 ui-sans-serif,-apple-system,"Segoe UI",Inter,sans-serif}
  button,input,select,textarea{font:inherit;color:inherit}
  header{display:flex;align-items:center;gap:18px;padding:12px 18px;border-bottom:1px solid var(--line);background:#faf7f0;position:sticky;top:0;z-index:20;flex-wrap:wrap}
  .brand{font-weight:700;letter-spacing:.2px}
  .brand small{display:block;font-weight:400;color:var(--muted);letter-spacing:0}
  .chips{display:flex;gap:6px;flex-wrap:wrap}
  .chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12px;color:#374151}
  .chip b{color:var(--ink)}
  .spacer{flex:1}
  .ctrl{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
  .ctrl select#model{max-width:320px}
  select,input[type=text],input[type=search]{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 8px}
  .dot{width:8px;height:8px;border-radius:50%;background:#9ca3af;display:inline-block}
  .dot.ok{background:#16a34a}.dot.bad{background:#dc2626}
  main{display:grid;grid-template-columns:290px minmax(420px,1fr) 380px;gap:14px;padding:14px;height:calc(100vh - 65px)}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);min-height:0;display:flex;flex-direction:column;overflow:hidden}
  .panel h2{margin:0;padding:11px 14px;border-bottom:1px solid var(--line);font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
  .filters{padding:10px;display:grid;gap:8px;border-bottom:1px solid var(--line)}
  .filters .row{display:flex;gap:6px}
  .filters .row>*{flex:1;min-width:0}
  .list{overflow:auto;padding:6px}
  .agent{display:flex;align-items:center;gap:9px;padding:7px 9px;border-radius:9px;cursor:pointer;border:1px solid transparent}
  .agent:hover{background:#f6f1e7}
  .agent.sel{background:#fdeee5;border-color:#f2c3ac}
  .agent .av{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;background:#e8e2d6;font-size:10px;font-weight:700;color:#4b5563;flex:none}
  .agent.clone .av{background:#fde2d6;color:var(--accent)}
  .agent .who{min-width:0;flex:1}
  .agent .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:600;font-size:13px}
  .agent .sub{color:var(--muted);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .badge{font-size:10px;padding:1px 6px;border-radius:999px;background:#eee7da;color:#6b7280;flex:none}
  .badge.hot{background:#fde2d6;color:var(--accent);font-weight:700}
  .viz{display:flex;flex-direction:column}
  .tabs{display:flex;gap:4px;padding:8px 10px;border-bottom:1px solid var(--line)}
  .tabs button{border:1px solid transparent;background:transparent;padding:5px 11px;border-radius:8px;cursor:pointer;color:var(--muted);font-size:12px}
  .tabs button.on{background:#efe9dd;color:var(--ink);border-color:var(--line)}
  .view{display:none;flex:1;min-height:0;overflow:auto;padding:12px}
  .view.on{display:block}
  .scatterbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:8px}
  .legend{display:inline-flex;align-items:center;gap:4px;font-size:11px}
  .legend i{width:9px;height:9px;border-radius:50%;display:inline-block;box-shadow:inset -2px -2px 2px #0003}
  .legend i.unique{background:#a8b2bc}
  .legend i.clone{background:#c2410c}
  .legend i.ring,.legend i.selring{background:transparent;border:2px solid #c2410c;box-shadow:none;width:11px;height:11px}
  .legend i.selring{border-color:#0f766e}
  #scatter{width:100%;height:calc(100vh - 248px);min-height:320px;display:block;cursor:crosshair}
  .tip{position:fixed;pointer-events:none;background:#111827;color:#fff;padding:4px 8px;border-radius:6px;font-size:11px;opacity:0;transition:opacity .08s;z-index:50;white-space:nowrap}
  table.heat{border-collapse:separate;border-spacing:4px;width:100%;font-size:12px}
  table.heat th{font-weight:600;color:var(--muted);text-align:left;padding:2px 6px;font-size:11px}
  table.heat td{border-radius:7px;padding:7px 5px;text-align:center;background:#f1ece2;position:relative;vertical-align:middle}
  table.heat td .n{display:block;font-weight:700;font-size:13px}
  table.heat td .vl{display:block;font-size:9px;line-height:1.15;color:#6b7280;margin-top:2px;max-width:104px}
  table.heat th:first-child{width:118px}
  .duprow{display:flex;align-items:center;gap:10px;padding:9px 10px;border:1px solid var(--line);border-radius:10px;margin-bottom:7px;cursor:pointer;background:#fff}
  .duprow:hover{border-color:#e3b7a2}
  .duprow .n{font-weight:800;font-size:18px;color:var(--accent);min-width:38px;text-align:right}
  .duprow .a{font-size:11px;color:var(--muted)}
  .chat{display:flex;flex-direction:column}
  .chead{padding:12px 14px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}
  .chead .nm{font-weight:700}
  .chead .sub{font-size:11px;color:var(--muted)}
  .msgs{flex:1;overflow:auto;padding:14px;display:flex;flex-direction:column;gap:10px}
  .empty{margin:auto;text-align:center;color:var(--muted);font-size:13px;max-width:240px}
  .msg{max-width:88%;padding:8px 11px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word;font-size:13px}
  .msg.user{align-self:flex-end;background:#1f2937;color:#fff;border-bottom-right-radius:3px}
  .msg.assistant{align-self:flex-start;background:#f1ece2;border-bottom-left-radius:3px}
  .msg.error{background:#fee2e2;color:#991b1b}
  .composer{border-top:1px solid var(--line);padding:10px;display:grid;gap:7px}
  .composer textarea{resize:none;height:62px;border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:#fff}
  .composer .row{display:flex;gap:8px;align-items:center}
  .composer .row .spacer{flex:1}
  .btn{background:var(--ink);color:#fff;border:0;border-radius:9px;padding:7px 14px;cursor:pointer;font-weight:600}
  .btn:disabled{opacity:.45;cursor:default}
  .btn.ghost{background:transparent;color:var(--muted);border:1px solid var(--line);font-weight:500}
  .meta{font-size:11px;color:var(--muted)}
  .attrs{display:flex;flex-wrap:wrap;gap:4px;padding:9px 14px;border-bottom:1px solid var(--line);background:#fffaf2}
  .attrs span{font-size:10px;background:#f1ece2;padding:2px 6px;border-radius:5px;color:#4b5563}
  .copynote{padding:8px 14px;border-bottom:1px solid var(--line);background:#fdf3ec;font-size:11px;color:#9a3412;line-height:1.7}
  .copynote b{color:#7c2d12}
  .copychip{border:1px solid #f2c3ac;background:#fff;color:#9a3412;border-radius:999px;padding:1px 7px;margin:0 2px;font-size:10px;cursor:pointer}
  .copychip:hover{background:#fdeee5}
  .copychip.cur{background:#c2410c;border-color:#c2410c;color:#fff;font-weight:700}
  .copynote details{margin-top:3px}
  .copynote summary{cursor:pointer;color:#9a3412;font-weight:600;font-size:11px;outline:none}
  .copynote summary:hover{text-decoration:underline}
  .affin{padding:8px 14px;border-bottom:1px solid var(--line);background:#f7f7f2;font-size:11px;color:var(--muted);line-height:1.8;display:none}
  .affin b{color:var(--ink)}
  .affin .affrow{display:flex;align-items:center;gap:7px}
  .affin .affbar{height:5px;border-radius:99px;background:#e5e0d5;flex:1;overflow:hidden}
  .affin .affbar i{display:block;height:100%;border-radius:99px}
  .affin .affname{cursor:pointer;color:var(--ink);text-decoration:underline;text-underline-offset:2px}
  .affin .gmember{display:inline-block;margin:0 6px 3px 0;padding:1px 6px;border-radius:99px;background:#fff;border:1px solid var(--line);cursor:pointer}
  .affin .gmember.me{background:#e6f4f1;border-color:#0f766e}
  .simctl{border-top:1px solid var(--line);padding:8px 10px;background:#fbf8f1;font-size:11px}
  .simhead{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
  .simfields{display:grid;grid-template-columns:1fr 1fr;gap:4px 8px}
  .simfields label{display:flex;flex-direction:column;gap:2px;color:var(--muted)}
  .simfields label.wide{grid-column:1/-1}
  .simfields input{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:3px 5px;font-size:11px;width:100%}
  .simfields label.check{flex-direction:row;align-items:center;gap:5px;color:var(--ink)}
  .simfields label.check input{width:auto}
  .simrow{display:flex;gap:6px;margin:7px 0 5px;flex-wrap:wrap}
  .simrow .btn{padding:4px 9px;font-size:11px}
  .simevents{max-height:120px;overflow:auto;font-size:10px;line-height:1.5;color:var(--muted)}
  .simevents div{padding:2px 0;border-bottom:1px dotted var(--line)}
  .simevents b{color:var(--ink);font-weight:600}
  .dupcopies{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
  .surveyctl{display:flex;flex-wrap:wrap;gap:12px;align-items:center;font-size:12px;color:var(--muted);margin-bottom:8px}
  .surveyctl select,.surveyctl input{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:5px 7px}
  .progwrap{display:flex;align-items:center;gap:10px;margin:6px 0 14px}
  .progbar{flex:1;height:9px;background:#e8e2d6;border-radius:999px;overflow:hidden}
  .progfill{height:100%;width:0;background:var(--accent);transition:width .2s}
  .svq{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:12px;background:#fff}
  .svq h3{margin:0 0 8px;font-size:13px}
  .svq .err{color:#b91c1c;font-size:11px}
  .svbar{display:grid;grid-template-columns:190px 1fr 74px;gap:8px;align-items:center;margin:3px 0;font-size:12px}
  .svbar .track{background:#efe9dd;border-radius:6px;height:15px;overflow:hidden}
  .svbar .fill{height:100%;background:var(--accent2)}
  .svbar .fill.lead{background:var(--accent)}
  .svbar .pct{text-align:right;color:var(--muted)}
  .svcross{margin-top:8px;font-size:11px;color:var(--muted)}
  .svcross table{border-collapse:collapse;margin-top:4px}
  .svcross th,.svcross td{border:1px solid var(--line);padding:2px 7px;text-align:left}
  @media (max-width:1100px){
    main{grid-template-columns:260px 1fr;height:auto}
    .chat{grid-column:1/-1;height:70vh}
    #scatter{height:420px}
  }
  @media (max-width:760px){
    main{grid-template-columns:1fr}
    .panel{max-height:70vh}
  }
</style>
</head>
<body>
<header>
  <div class="brand">Population synthétique<small>500 agents · marges exactes</small></div>
  <div class="chips" id="stats"></div>
  <div class="spacer"></div>
  <div class="ctrl"><span class="dot" id="status"></span>
    <select id="provider" title="fournisseur">
      <option value="mock">mock</option>
      <option value="ollama">ollama</option>
      <option value="openai">openai</option>
    </select>
    <select id="model" title="modèle"></select>
  </div>
</header>

<main>
  <aside class="panel">
    <h2>Agents <span id="shown" class="badge"></span></h2>
    <div class="filters">
      <input type="search" id="q" placeholder="Nom, clé, id…">
      <div class="row">
        <select id="fattr"><option value="">Tous</option></select>
        <select id="fval"><option value="">—</option></select>
      </div>
    </div>
    <div class="list" id="list"></div>
    <div class="simctl">
      <div class="simhead"><b>Simulation</b><span class="badge" id="simstate">arrêtée</span></div>
      <div class="simfields">
        <label>Ticks <input id="simticks" type="number" value="200" min="1" max="5000"></label>
        <label>Vitesse <select id="simspeed" title="intervalle entre deux ticks">
          <option value="0">max</option>
          <option value="0.03">rapide</option>
          <option value="0.1" selected>normale</option>
          <option value="0.3">lente</option>
          <option value="1">très lente</option>
        </select></label>
        <label>Spark <input id="simspark" type="number" value="0.5" min="0" max="1" step="0.05"></label>
        <label>Groupes <input id="simgroup" type="number" value="0.35" min="0" max="1" step="0.05"></label>
        <label>Refus <input id="simrefusal" type="number" value="0.1" min="0" max="1" step="0.05"></label>
        <label>Conv./tick <input id="simburst" type="number" value="4" min="1" max="64"></label>
        <label>Parallèle <input id="simpar" type="number" value="4" min="1" max="32"></label>
        <label>Agents <input id="simlimit" type="number" value="" min="2" max="500" placeholder="tous"></label>
        <label class="wide">Run <input id="simrun" type="text" placeholder="auto (gui-horodatage)"></label>
        <label class="wide">Runs <select id="simruns"><option value="">— choisir un run —</option></select></label>
        <label class="wide check"><input id="simfresh" type="checkbox" checked> repartir de zéro (efface le run)</label>
      </div>
      <div class="simrow">
        <button class="btn" id="simstart">Démarrer</button>
        <button class="btn ghost" id="simstop" disabled>Arrêter</button>
        <button class="btn ghost" id="simreset" title="efface l'état affiché (affinités, groupes, opinions)">Réinitialiser</button>
      </div>
      <div class="simrow">
        <button class="btn ghost" id="simload">Charger</button>
        <button class="btn ghost" id="simresume">Reprendre</button>
        <button class="btn ghost" id="simdelete">Supprimer</button>
        <button class="btn ghost" id="simexport">Exporter</button>
      </div>
      <div class="meta" id="siminfo">Aucune simulation lancée.</div>
      <div class="simevents" id="simevents"></div>
    </div>
  </aside>

  <section class="panel viz">
    <div class="tabs">
      <button class="on" data-view="atlas">Atlas</button>
      <button data-view="attrs">Attributs</button>
      <button data-view="dups">Copies exactes</button>
      <button data-view="survey">Sonder</button>
    </div>
    <div class="view on" id="view-atlas">
      <div class="scatterbar">
        <label>X <select id="xa"></select></label>
        <label>Y <select id="ya"></select></label>
        <label><input type="checkbox" id="animate" checked> animer</label>
        <span class="legend"><i class="unique"></i>profil unique</span>
        <span class="legend"><i class="clone"></i>profil dupliqué</span>
        <span class="legend"><i class="ring"></i>anneau = groupe</span>
        <span class="legend"><i class="selring"></i>anneau teal = sélection</span>
        <span id="atlasinfo"></span>
        <span class="meta" id="affstat"></span>
      </div>
      <canvas id="scatter"></canvas>
    </div>
    <div class="view" id="view-attrs">
      <table class="heat" id="heat"></table>
      <p class="meta" style="margin-top:12px">Chaque cellule donne le nombre d'agents. Les marges imposent ces distributions : elles ne dépendent pas de la génération.</p>
    </div>
    <div class="view" id="view-dups">
      <p class="meta" style="margin-top:0">Les copies exactes sont conservées : chaque ligne est un agent distinct (prénom + nom uniques), mais plusieurs agents peuvent partager le même profil sur les 12 attributs contraints. Ils sont regroupés par <code>cluster_id</code> et répondront de façon quasi identique à modèle et température égaux.</p>
      <div id="dups"></div>
    </div>
    <div class="view" id="view-survey">
      <p class="meta" style="margin-top:0">Pose le même questionnaire à tout ou partie des agents, puis agrège les réponses. Le fournisseur et le modèle sont ceux du bandeau en haut. La sonde tourne en arrière-plan : on peut suivre la progression et la relancer.</p>
      <div class="surveyctl">
        <label>Questionnaire <select id="svSurvey"></select></label>
        <label>Portée <select id="svScope">
          <option value="all">tous les agents (500)</option>
          <option value="one_per_cluster">un par profil (302)</option>
          <option value="filtered">filtre courant</option>
          <option value="sample">échantillon</option>
        </select></label>
        <label id="svLimitWrap" style="display:none">taille <input type="number" id="svLimit" value="30" min="1" max="500" style="width:70px"></label>
        <label>croiser par <select id="svCross"><option value="">—</option></select></label>
        <label><input type="checkbox" id="svShuffle"> ordre aléatoire</label>
        <button class="btn" id="svRun">Lancer la sonde</button>
      </div>
      <div id="svScopeInfo" class="meta"></div>
      <div class="progwrap" id="svProgWrap" style="display:none">
        <div class="progbar"><div class="progfill" id="svProgFill"></div></div>
        <span class="meta" id="svProgTxt"></span>
      </div>
      <div id="svResults"></div>
    </div>
  </section>

  <aside class="panel chat" id="chat">
    <div class="chead" id="chead">
      <div><div class="nm">Aucun agent</div><div class="sub">choisis un agent à gauche</div></div>
    </div>
    <div class="attrs" id="cattrs" style="display:none"></div>
    <div class="copynote" id="copynote" style="display:none"></div>
    <div class="affin" id="affinity"></div>
    <div class="affin" id="group"></div>
    <div class="affin" id="opinions"></div>
    <div class="msgs" id="msgs"><div class="empty">Sélectionne un agent pour discuter avec lui.<br><br>Le prompt système est construit côté serveur à partir de ses attributs.</div></div>
    <div class="composer">
      <textarea id="input" placeholder="Pose une question… (Entrée pour envoyer)" disabled></textarea>
      <div class="row">
        <span class="meta" id="lat"></span><span class="spacer"></span>
        <button class="btn ghost" id="reset" title="effacer l'historique">reset</button>
        <button class="btn" id="send" disabled>Envoyer</button>
      </div>
    </div>
  </aside>
</main>
<div class="tip" id="tip"></div>

<script>
const $ = s => document.querySelector(s);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const S = { agents: [], meta: null, sel: null, history: [], filter: { q: "", attr: "", val: "" } };

function provider(){ return $("#provider").value; }
function model(){ return $("#model").value.trim(); }
let AVAILABLE = { ollama: [], suggestions: [] };
const OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"];

function pct(x){ return (100*x).toFixed(1) + " %"; }
function fmt(x, d=2){ return Number(x).toFixed(d).replace(".", ","); }

function affinityKey(a, b){ return a <= b ? `${a}:${b}` : `${b}:${a}`; }
function affinityBetween(a, b){
  const tab = (S.meta && S.meta.affinity) || {};
  const v = tab[affinityKey(a, b)];
  return v == null ? ((S.meta && S.meta.affinity_default) ?? 0.5) : v;
}
function affinityLabel(v){
  if (v < 0.25) return "très faible";
  if (v < 0.45) return "faible";
  if (v <= 0.55) return "neutre";
  if (v < 0.75) return "bonne";
  return "forte";
}
function affinityColor(v){
  const hue = 8 + 126 * Math.max(0, Math.min(1, v));
  return `hsl(${hue.toFixed(0)} 62% 40%)`;
}
function affinitiesOf(id){
  const out = [];
  for (const [key, value] of Object.entries((S.meta && S.meta.affinity) || {})){
    const [a, b] = key.split(":").map(Number);
    if (a === id) out.push({ id: b, value });
    else if (b === id) out.push({ id: a, value });
  }
  return out.sort((x, y) => y.value - x.value);
}
function groupsOf(id){
  return Object.values((S.meta && S.meta.groups) || {})
    .filter(g => (g.members || []).includes(id));
}
function groupColor(gid){
  const keys = Object.keys((S.meta && S.meta.groups) || {});
  const index = Math.max(0, keys.indexOf(gid));
  return `hsl(${(index * 47) % 360} 62% 42%)`;
}
function AL(a){ return (S.meta && S.meta.labels && S.meta.labels[a]) || a; }
function VL(a, v){ return (S.meta && S.meta.labels && S.meta.labels[a + "=" + v]) || v; }
function attrName(a){ const l = AL(a); return l === a ? a : a + " — " + l; }

function renderAffinityStatus(){
  const box = $("#affstat"); if (!box) return;
  const n = Object.keys((S.meta && S.meta.affinity) || {}).length;
  if (S.meta && S.meta.affinity_source){
    const run = S.meta.affinity_source.split("/").slice(-2, -1)[0] || "state.json";
    const nbGroups = Object.keys(S.meta.groups || {}).length;
    box.textContent = `affinités · ${n} paire(s) · ${nbGroups} groupe(s) · ${run}`;
    box.title = S.meta.affinity_source;
  } else {
    box.textContent = "affinités · aucun run social";
  }
}

async function boot(){
  const [meta, data] = await Promise.all([
    fetch("/api/meta").then(r => r.json()),
    fetch("/api/agents").then(r => r.json())
  ]);
  S.meta = meta; S.agents = data.agents;
  renderStats(); renderFilters(); renderAttrSelects(); renderHeat(); renderDups();
  renderList(); renderSurveyCross(); loadSurveys(); renderAffinityStatus();
  if (REDUCED) $("#animate").checked = false; startAtlas();
  await loadModels(); health();
  const last = localStorage.getItem("selectedAgent");
  if (last && S.agents.some(a => a.id === +last)) selectAgent(+last);
}

function renderStats(){
  const s = S.meta.stats;
  const box = $("#stats"); box.innerHTML = "";
  const add = (label, value, title) => {
    const c = el("span", "chip"); c.title = title || "";
    c.append(label + " ");
    const b = el("b", null, value); c.append(b); box.append(c);
  };
  add("identités", s.identities, "identités uniques (prénom + nom)");
  add("profils", s.distinct_profiles, "profils distincts sur les 12 attributs contraints");
  add("agents en copies", s.duplicated_agents, "agents qui partagent leur profil avec au moins un autre");
  add("H", fmt(s.entropy, 3) + " nats", "entropie de Shannon");
  add("e^H", fmt(s.effective, 0), "nombre effectif de personas");
  add("max", "×" + s.max_mult, "multiplicité maximale d'un profil");
}

function renderFilters(){
  const sel = $("#fattr");
  for (const a of S.meta.attrs) sel.append(new Option(attrName(a), a));
  sel.onchange = () => {
    S.filter.attr = sel.value; S.filter.val = "";
    const vs = $("#fval"); vs.innerHTML = "";
    vs.append(new Option("—", ""));
    if (sel.value) for (const v of S.meta.domains[sel.value]) {
      const n = S.meta.distributions[sel.value][v] || 0;
      vs.append(new Option(`${VL(sel.value, v)} (${n})`, v));
    }
    renderList(); updateAtlasInfo(); updateSurveyScopeInfo();
  };
  $("#fval").onchange = e => { S.filter.val = e.target.value; renderList(); updateAtlasInfo(); updateSurveyScopeInfo(); };
  $("#q").oninput = e => { S.filter.q = e.target.value.toLowerCase().trim(); renderList(); updateAtlasInfo(); updateSurveyScopeInfo(); };
}

function renderAttrSelects(){
  const xs = $("#xa"), ys = $("#ya");
  S.meta.attrs.forEach((a, i) => {
    xs.append(new Option(attrName(a), a)); ys.append(new Option(attrName(a), a));
  });
  xs.value = S.meta.attrs[0]; ys.value = S.meta.attrs[1] || S.meta.attrs[0];
  xs.onchange = ys.onchange = () => { updateAtlasInfo(); if (!ATLAS_RAF) drawAtlasFrame(ATLAS_T); };
  $("#animate").onchange = () => {
    if ($("#animate").checked) startAtlas();
    else { stopAtlas(); drawAtlasFrame(ATLAS_T); }
  };
}

function visible(){
  const f = S.filter;
  return S.agents.filter(a => {
    if (f.attr && f.val && a.attributes[f.attr] !== f.val) return false;
    if (!f.q) return true;
    return (a.name + " " + a.identity_key + " " + a.cluster_id + " " + a.id).toLowerCase().includes(f.q);
  });
}

function renderList(){
  const list = $("#list"); list.innerHTML = "";
  const items = visible();
  $("#shown").textContent = items.length;
  const frag = document.createDocumentFragment();
  for (const a of items){
    const d = el("div", "agent" + (a.mult > 1 ? " clone" : "") + (S.sel === a.id ? " sel" : ""));
    d.dataset.id = a.id;
    const init = a.name.split(" ").map(x => x[0]).join("").slice(0, 2).toUpperCase();
    d.append(el("div", "av", init));
    const who = el("div", "who");
    who.append(el("div", "nm", a.name));
    const tag = a.copies > 1 ? `copie ${a.copy_index}/${a.copies}` : "profil unique";
    who.append(el("div", "sub", "#" + a.id + " · " + tag + " · " + Object.entries(a.attributes).slice(0, 2).map(([k, v]) => VL(k, v)).join(" · ")));
    d.append(who);
    const b = el("span", "badge" + (a.copies > 1 ? " hot" : ""), a.copies > 1 ? "×" + a.copies : "1");
    b.title = a.copies > 1
      ? `profil ${a.cluster_id} partagé par ${a.copies} agents (copies exactes)`
      : "profil unique sur les 12 attributs contraints";
    d.append(b);
    d.onclick = () => selectAgent(a.id);
    frag.append(d);
  }
  list.append(frag);
}

function renderAffinity(id){
  const box = $("#affinity"); box.innerHTML = "";
  const list = affinitiesOf(id);
  if (!list.length){
    if (S.meta && S.meta.affinity_source){
      box.style.display = "block";
      box.append(el("b", null, "Affinités"));
      box.append(document.createTextNode(
        " — aucune paire enregistrée pour cet agent dans ce run."));
    } else {
      box.style.display = "none";
    }
    return;
  }
  box.style.display = "block";
  box.append(el("b", null, `Affinités · ${list.length} lien(s)`));
  const source = (S.meta.affinity_source || "").split("/").slice(-2, -1)[0];
  if (source) box.append(document.createTextNode(` · run ${source}`));
  const strong = list.slice(0, 4);
  const weak = list.filter(x => !strong.includes(x)).slice(-2).reverse();
  for (const item of [...strong, ...weak]){
    const other = S.agents.find(a => a.id === item.id);
    if (!other) continue;
    const row = el("div", "affrow");
    const name = el("span", "affname", other.name);
    name.title = `affinité ${fmt(item.value)} · ${affinityLabel(item.value)}`;
    name.onclick = () => selectAgent(other.id);
    const track = el("span", "affbar");
    const fill = el("i");
    fill.style.width = (100 * item.value) + "%";
    fill.style.background = affinityColor(item.value);
    track.append(fill);
    row.append(name, track, el("span", null, fmt(item.value)));
    box.append(row);
  }
  box.append(el("div", "meta", "Survole un agent dans l'Atlas pour comparer."));
}

function renderGroup(id){
  const box = $("#group"); box.innerHTML = "";
  const groups = (S.meta && S.meta.groups) || {};
  if (!Object.keys(groups).length){ box.style.display = "none"; return; }
  box.style.display = "block";
  const mine = groupsOf(id);
  if (!mine.length){
    box.append(el("b", null, "Groupe"));
    box.append(document.createTextNode(" — cet agent est hors groupe."));
    return;
  }
  for (const group of mine){
    const color = groupColor(group.id);
    const title = el("b", null, `Groupe ${group.id}`);
    title.style.color = color;
    box.append(title);
    box.append(document.createTextNode(
      ` · ${group.members.length} membres · cohésion ${fmt(group.cohesion)}` +
      ` · ${group.conversations || 0} conversation(s)`));
    const row = el("div");
    for (const member of group.members){
      const agent = S.agents.find(a => a.id === member);
      if (!agent) continue;
      const chip = el("span", "gmember" + (member === id ? " me" : ""), agent.name);
      chip.style.borderColor = color;
      chip.onclick = () => selectAgent(agent.id);
      row.append(chip);
    }
    box.append(row);
  }
}

function topicLabel(tid){
  return (S.meta && S.meta.topic_labels && S.meta.topic_labels[tid]) || tid;
}
function opinionsOf(id){
  const table = (S.meta && S.meta.opinions) || {};
  const values = table[id] || table[String(id)] || {};
  return Object.entries(values)
    .map(([tid, value]) => ({ tid, value }))
    .sort((a, b) => Math.abs(b.value - 0.5) - Math.abs(a.value - 0.5));
}
function renderOpinions(id){
  const box = $("#opinions"); box.innerHTML = "";
  const list = opinionsOf(id);
  const all = Object.keys((S.meta && S.meta.opinions) || {});
  if (!list.length){
    if (all.length){
      box.style.display = "block";
      box.append(el("b", null, "Opinions"));
      box.append(document.createTextNode(
        " — aucune opinion enregistrée pour cet agent."));
    } else {
      box.style.display = "none";
    }
    return;
  }
  box.style.display = "block";
  box.append(el("b", null, `Opinions · ${list.length} sujet(s)`));
  for (const item of list.slice(0, 4)){
    const row = el("div", "affrow");
    const label = el("span", "affname", topicLabel(item.tid));
    label.title = `opinion ${fmt(item.value)} sur ${item.tid}`;
    const track = el("span", "affbar");
    const fill = el("i");
    fill.style.width = (100 * item.value) + "%";
    fill.style.background = affinityColor(item.value);
    track.append(fill);
    row.append(label, track, el("span", null, fmt(item.value)));
    box.append(row);
  }
}

function selectAgent(id){
  const a = S.agents.find(x => x.id === id); if (!a) return;
  S.sel = id; S.history = []; localStorage.setItem("selectedAgent", id);
  renderList();
  $("#msgs").innerHTML = "";
  const head = $("#chead"); head.innerHTML = "";
  const av = el("div", "av", a.name.split(" ").map(x => x[0]).join("").slice(0,2).toUpperCase());
  av.style.cssText = "width:34px;height:34px;border-radius:50%;display:grid;place-items:center;background:#fde2d6;color:#c2410c;font-weight:800;font-size:12px";
  const box = el("div");
  box.append(el("div", "nm", a.name));
  const tag = a.copies > 1 ? `copie ${a.copy_index}/${a.copies}` : "profil unique";
  box.append(el("div", "sub", "#" + a.id + " · " + a.identity_key + " · " + tag));
  head.append(av, box);
  const cn = $("#copynote"); cn.innerHTML = "";
  if (a.copies > 1) {
    const peers = S.agents.filter(x => x.cluster_id === a.cluster_id);
    cn.append(el("b", null, `Copie ${a.copy_index}/${a.copies}`));
    cn.append(document.createTextNode(
      ` — profil ${a.cluster_id} partagé par ${a.copies} agents aux 12 attributs identiques `));
    const inline = a.copies <= 8;
    const wrap = el(inline ? "span" : "details");
    if (!inline) wrap.append(el("summary", null, `voir les ${a.copies} copies`));
    for (const peer of peers) {
      const chip = el("button", "copychip" + (peer.id === a.id ? " cur" : ""), peer.name);
      chip.title = `#${peer.id} · ${peer.identity_key}`;
      chip.onclick = () => selectAgent(peer.id);
      wrap.append(chip);
    }
    cn.append(wrap);
  } else {
    cn.append(el("b", null, "Profil unique"));
    cn.append(document.createTextNode(
      " — aucun autre agent ne partage exactement ces 12 attributs."));
  }
  cn.style.display = "block";
  const at = $("#cattrs"); at.innerHTML = ""; at.style.display = "flex";
  for (const [k, v] of Object.entries(a.attributes)) {
    if (k === "prenom" || k === "nom") continue;
    at.append(el("span", null, AL(k) + " : " + VL(k, v)));
  }
  renderAffinity(a.id);
  renderGroup(a.id);
  renderOpinions(a.id);
  $("#input").disabled = false; $("#send").disabled = false;
  $("#lat").textContent = "";
  const empty = el("div", "empty", "À toi de jouer. Exemple : « Présente-toi en deux phrases. »");
  $("#msgs").append(empty);
  $("#input").focus();
  if (!ATLAS_RAF) drawAtlasFrame(ATLAS_T);
}

async function send(){
  if (!S.sel) return;
  const input = $("#input"); const text = input.value.trim(); if (!text) return;
  input.value = "";
  S.history.push({ role: "user", content: text }); renderChat();
  $("#send").disabled = true; $("#lat").textContent = "…";
  // nœud de streaming (jeton par jeton)
  const box = $("#msgs");
  const node = el("div", "msg assistant stream", "");
  box.append(node); box.scrollTop = box.scrollHeight;
  let full = "";
  try{
    const r = await fetch("/api/chat/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_id: S.sel, message: text, history: S.history.slice(0, -1).filter(m => !m.error),
        provider: provider(), model: model()
      })
    });
    if (!r.ok || !r.body){
      const data = await r.json().catch(() => ({}));
      throw new Error(data.error || ("HTTP " + r.status));
    }
    const reader = r.body.getReader(); const dec = new TextDecoder();
    let buf = "";
    for (;;){
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0){
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line) continue;
        let ev; try { ev = JSON.parse(line); } catch { continue; }
        if (ev.type === "token"){ full += ev.t; node.textContent = full; box.scrollTop = box.scrollHeight; }
        else if (ev.type === "error"){ throw new Error(ev.error); }
        else if (ev.type === "done"){ $("#lat").textContent = `${ev.provider}/${ev.model} · ${ev.ms} ms`; }
      }
    }
    S.history.push({ role: "assistant", content: full });
  }catch(e){
    node.remove();
    S.history.push({ role: "assistant", content: "⚠ " + e.message, error: true });
    $("#lat").textContent = "";
  }finally{
    $("#send").disabled = false; renderChat(); input.focus();
  }
}

function renderChat(){
  const box = $("#msgs"); box.innerHTML = "";
  if (!S.history.length){
    box.append(el("div", "empty", "À toi de jouer. Exemple : « Présente-toi en deux phrases. »"));
    return;
  }
  for (const m of S.history){
    box.append(el("div", "msg " + (m.error ? "error" : m.role), m.content));
  }
  box.scrollTop = box.scrollHeight;
}

function renderHeat(){
  const t = $("#heat"); t.innerHTML = "";
  const head = el("tr");
  head.append(el("th", null, "attribut"));
  for (const v of S.meta.domains[S.meta.attrs[0]]) head.append(el("th", null, v));
  head.append(el("th", null, "valeur dominante"));
  t.append(head);
  let max = 1;
  for (const a of S.meta.attrs) for (const n of Object.values(S.meta.distributions[a])) max = Math.max(max, n);
  for (const a of S.meta.attrs){
    const tr = el("tr"); tr.append(el("th", null, attrName(a)));
    let best = 0, bestV = "";
    for (const v of S.meta.domains[a]){
      const n = S.meta.distributions[a][v] || 0;
      if (n > best){ best = n; bestV = v; }
      const td = el("td");
      td.style.background = `rgba(194,65,12,${0.06 + 0.5 * n / max})`;
      td.title = `${AL(a)} : ${VL(a, v)} — ${n} agents (${pct(n/500)})`;
      td.append(el("span", "n", n));
      td.append(el("span", "vl", VL(a, v)));
      tr.append(td);
    }
    const tdMax = el("td");
    tdMax.style.background = "#efe9dd";
    tdMax.append(el("span", "n", pct(best / 500)));
    tdMax.append(el("span", "vl", VL(a, bestV)));
    tr.append(tdMax);
    t.append(tr);
  }
}

function renderDups(){
  const box = $("#dups"); box.innerHTML = "";
  if (!S.meta.duplicates.length){ box.append(el("p", "meta", "Aucun doublon.")); return; }
  for (const d of S.meta.duplicates){
    const row = el("div", "duprow");
    row.append(el("div", "n", "×" + d.count));
    const mid = el("div"); mid.style.flex = "1";
    mid.append(el("div", null, `profil ${d.cluster_id || d.key.slice(0, 8)} — ${d.count} copies exactes`));
    mid.append(el("div", "a", Object.entries(d.attrs).map(([k, v]) => AL(k) + " : " + VL(k, v)).join(" · ")));
    const chips = el("div", "dupcopies");
    for (const id of (d.agent_ids || [d.agent_id])) {
      const a = S.agents.find(x => x.id === id);
      if (!a) continue;
      const chip = el("button", "copychip", a.name);
      chip.title = `#${a.id} · ${a.identity_key}`;
      chip.onclick = ev => { ev.stopPropagation(); selectAgent(a.id); };
      chips.append(chip);
    }
    mid.append(chips);
    row.append(mid);
    row.onclick = () => selectAgent(d.agent_id);
    box.append(row);
  }
}

const MOTION = new Map();
const SPRITES = {};
let ATLAS_RAF = 0, ATLAS_LAST = 0, ATLAS_T = 0, ATLAS_HITS = [], ATLAS_ACTIVE = false;
let ATLAS_HOVER = null;
const REDUCED = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

function rnd(id){ let x = Math.sin(id * 999.91) * 10000; return x - Math.floor(x); }
function clamp(v, a, b){ return Math.max(a, Math.min(b, v)); }

function eyeOpen(t, ph){
  // Ouvert la plupart du temps, fermé ~0,14 s de temps en temps.
  const cyc = 2.4 + (ph % 1) * 3.2;
  const tt = (t + ph * 1.7) % cyc;
  return tt < 0.14 ? Math.abs(tt - 0.07) / 0.07 : 1;
}

function initMotion(){
  if (MOTION.size || !S.agents.length) return;
  for (const a of S.agents){
    MOTION.set(a.id, {
      x: 0.03 + rnd(a.id * 1.93) * 0.94,
      y: 0.03 + rnd(a.id * 7.31 + 4) * 0.94,
      tx: null, ty: null,
      vx: 0, vy: 0,
      state: "idle",
      timer: 1.2 + rnd(a.id * 6.13 + 2) * 6.0,   // surtout immobile au départ
      hopZ: 0, hopVz: 0,
      phase: rnd(a.id * 11.3) * Math.PI * 2,
      ox: Math.cos(rnd(a.id * 2.71 + 3) * Math.PI * 2) * (0.02 + rnd(a.id * 4.19 + 7) * 0.11),
      oy: Math.sin(rnd(a.id * 2.71 + 3) * Math.PI * 2) * (0.02 + rnd(a.id * 4.19 + 7) * 0.11) * 0.8
    });
  }
}

function sphereSprite(kind){
  if (SPRITES[kind]) return SPRITES[kind];
  const N = 64, cv = document.createElement("canvas");
  cv.width = cv.height = N;
  const q = cv.getContext("2d");
  const cols = kind === "clone"
    ? ["#f8c3a3", "#c2410c", "#6f2508"]
    : ["#f2f5f8", "#9aa6b2", "#4f5a66"];
  const grd = q.createRadialGradient(N*.34, N*.30, N*.04, N*.5, N*.5, N*.52);
  grd.addColorStop(0, cols[0]);
  grd.addColorStop(.48, cols[1]);
  grd.addColorStop(1, cols[2]);
  q.fillStyle = grd;
  q.beginPath(); q.arc(N/2, N/2, N/2 - 2, 0, Math.PI*2); q.fill();
  q.globalAlpha = .75; q.fillStyle = "#fff";
  q.beginPath(); q.ellipse(N*.37, N*.31, N*.10, N*.07, -.6, 0, Math.PI*2); q.fill();
  SPRITES[kind] = cv;
  return cv;
}

function atlasLayout(){
  const c = $("#scatter");
  const w = c.clientWidth, h = c.clientHeight;
  const xa = $("#xa").value, ya = $("#ya").value;
  const dx = S.meta.domains[xa], dy = S.meta.domains[ya];
  const pad = { l: 54, r: 20, t: 22, b: 50 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const px = i => pad.l + iw * (i + .5) / Math.max(1, dx.length);
  const py = i => pad.t + ih * (i + .5) / Math.max(1, dy.length);
  return { c, g: c.getContext("2d"), w, h, xa, ya, dx, dy, pad, iw, ih, px, py };
}

function filterActive(){
  return !!(S.filter.q || (S.filter.attr && S.filter.val));
}

function updateAtlasInfo(){
  if (!S.meta) return;
  const active = filterActive();
  const n = active ? visible().length : S.agents.length;
  $("#atlasinfo").textContent = active
    ? `filtre actif · ${n} agent(s) regroupés dans leurs cellules ${$("#xa").value} × ${$("#ya").value}`
    : `essaim · ${S.agents.length} agents — ils se déplacent par sauts ; un filtre les regroupe dans leurs cellules`;
  settleAtlas();
}

function stepAtlas(dt, t){
  const L = atlasLayout();
  if (!L.w || !L.h) return;
  const active = filterActive();
  const match = active ? new Set(visible().map(a => a.id)) : null;
  for (const a of S.agents){
    const m = MOTION.get(a.id); if (!m) continue;
    const i = L.dx.indexOf(a.attributes[L.xa]);
    const j = L.dy.indexOf(a.attributes[L.ya]);
    const target = active && match.has(a.id) && i >= 0 && j >= 0;
    if (target){
      // Filtre actif : l'agent rejoint sa cellule et s'y pose (nuage lisible).
      const tx = clamp((L.px(i) - L.pad.l) / L.iw + m.ox * 1.7, .02, .98);
      const ty = clamp((L.py(j) - L.pad.t) / L.ih + m.oy * 1.7, .02, .98);
      m.vx += (tx - m.x) * 10 * dt;
      m.vy += (ty - m.y) * 10 * dt;
      const damp = Math.exp(-3.2 * dt);
      m.vx *= damp; m.vy *= damp;
      m.state = "idle"; m.hopVz = 0; m.hopZ *= Math.exp(-6 * dt);
    } else if (m.tx != null){
      // Position issue du run : on rejoint la cible en douceur au lieu de
      // téléporter la bille à chaque rafraîchissement de l'état.
      const k = 1 - Math.exp(-dt / 0.35);
      const dx = m.tx - m.x, dy = m.ty - m.y;
      m.x += dx * k; m.y += dy * k;
      m.vx = dx * 0.25; m.vy = dy * 0.25;   // orientation des yeux
      m.state = "idle"; m.hopVz = 0; m.hopZ *= Math.exp(-6 * dt);
      m.timer = Math.max(m.timer, 1.5);      // pas de saut pendant le suivi
    } else if (m.state === "jump"){
      // En l'air : petite trajectoire balistique.
      m.hopVz -= 3.4 * dt;
      m.hopZ += m.hopVz * dt;
      if (m.hopZ <= 0){
        m.hopZ = 0; m.hopVz = 0; m.vx = 0; m.vy = 0;
        m.state = "idle";
        m.timer = 2.0 + rnd(a.id * 3.7 + Math.floor(t * 7)) * 6.5;  // longue pause
      }
    } else {
      // Au sol : immobile, sauf quand le minuteur déclenche un saut.
      m.hopZ *= Math.exp(-8 * dt);
      m.timer -= dt;
      if (m.timer <= 0){
        const ang = rnd(a.id * 7.1 + t * 41.3) * Math.PI * 2;
        const sp = 0.035 + rnd(a.id * 4.9 + t * 23.1) * 0.055;
        m.state = "jump";
        m.hopVz = 0.30 + rnd(a.id * 2.3 + t * 17.7) * 0.35;
        m.vx = Math.cos(ang) * sp; m.vy = Math.sin(ang) * sp;
      }
    }
    m.x += m.vx * dt; m.y += m.vy * dt;
    if (m.x < .02){ m.x = .02; m.vx = Math.abs(m.vx) * .5; }
    if (m.x > .98){ m.x = .98; m.vx = -Math.abs(m.vx) * .5; }
    if (m.y < .02){ m.y = .02; m.vy = Math.abs(m.vy) * .5; }
    if (m.y > .98){ m.y = .98; m.vy = -Math.abs(m.vy) * .5; }
  }
}

function settleAtlas(){
  if (ATLAS_RAF || !MOTION.size) return;   // animé : la boucle s'en occupe
  for (let k = 0; k < 100; k++) stepAtlas(1 / 30, ATLAS_T + k / 30);
  drawAtlasFrame(ATLAS_T);
}

function drawEyes(g, x, cy, r, m, open, alpha){
  const look = clamp(m.vx / 0.06, -1, 1);
  const ex = r * 0.30, ey = -r * 0.14;
  const ew = r * 0.15, eh = Math.max(0.5, r * 0.21 * open);
  g.globalAlpha = alpha;
  g.fillStyle = "#20252b";
  for (const s of [-1, 1]){
    g.beginPath();
    g.ellipse(x + s * ex + look * r * 0.05, cy + ey, ew, eh, 0, 0, Math.PI * 2);
    g.fill();
  }
  if (open > 0.5){
    g.fillStyle = "rgba(255,255,255,.85)";
    for (const s of [-1, 1]){
      g.beginPath();
      g.arc(x + s * ex + look * r * 0.05 - ew * 0.35, cy + ey - eh * 0.4,
            Math.max(0.5, ew * 0.34), 0, Math.PI * 2);
      g.fill();
    }
  }
}

function drawAtlasFrame(t){
  if (!S.meta) return;
  const L = atlasLayout();
  if (!L.w || !L.h) return;
  const { c, g, w, h, dx, dy, pad, iw, ih, px, py, xa, ya } = L;
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)){
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  // Sol : damier régulier, une case par couple de valeurs.
  for (let i = 0; i < dx.length; i++){
    for (let j = 0; j < dy.length; j++){
      g.fillStyle = (i + j) % 2 ? "#f4efe4" : "#f9f5ec";
      g.fillRect(pad.l + iw * i / dx.length, pad.t + ih * j / dy.length,
                 iw / dx.length + .5, ih / dy.length + .5);
    }
  }
  g.strokeStyle = "#e6dfd2"; g.lineWidth = 1;
  for (let i = 1; i < dx.length; i++){
    const x = pad.l + iw * i / dx.length;
    g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, pad.t + ih); g.stroke();
  }
  for (let j = 1; j < dy.length; j++){
    const y = pad.t + ih * j / dy.length;
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(pad.l + iw, y); g.stroke();
  }
  g.strokeStyle = "#d9d3c7"; g.strokeRect(pad.l, pad.t, iw, ih);

  g.fillStyle = "#6b7280"; g.font = "11px ui-sans-serif"; g.textAlign = "center";
  dx.forEach((v, i) => g.fillText(v, px(i), h - 28));
  g.fillText(AL(xa), pad.l + iw/2, h - 10);
  g.save(); g.translate(15, pad.t + ih/2); g.rotate(-Math.PI/2);
  g.fillText(AL(ya), 0, 0); g.restore();

  const active = filterActive();
  const match = active ? new Set(visible().map(a => a.id)) : null;
  const order = S.agents.slice().sort((p, q) => {
    const mp = MOTION.get(p.id), mq = MOTION.get(q.id);
    return (mp.y - mp.hopZ * 0.2) - (mq.y - mq.hopZ * 0.2);
  });
  const hits = [];
  const groupOf = {};
  for (const [gid, group] of Object.entries((S.meta && S.meta.groups) || {})){
    for (const member of group.members || []) groupOf[member] = gid;
  }
  for (const a of order){
    const m = MOTION.get(a.id); if (!m) continue;
    const x = pad.l + m.x * iw;
    const groundY = pad.t + m.y * ih;
    const dim = active && !match.has(a.id);
    const sel = a.id === S.sel;
    const r = (a.copies > 1 ? 7.6 : 6.4) * (0.86 + 0.26 * m.y);
    const lift = m.hopZ * ih * 1.15;               // hauteur du saut à l'écran
    const squash = 1 + Math.min(0.22, m.hopZ * 1.4); // étirement léger en l'air
    const cy = groundY - lift;
    const alpha = dim ? .16 : 1;

    g.globalAlpha = alpha * (.16 - Math.min(.10, m.hopZ * 1.6));
    g.beginPath(); g.ellipse(x, groundY + r * .82, r * (1.0 + m.hopZ), r * .34, 0, 0, Math.PI*2);
    g.fillStyle = "#3b2f22"; g.fill();

    const sprite = sphereSprite(a.copies > 1 ? "clone" : "unique");
    g.globalAlpha = alpha;
    g.drawImage(sprite, x - r, cy - r, r * 2, r * 2 * squash);
    drawEyes(g, x, cy, r, m, eyeOpen(t, m.phase), alpha);

    const gid = groupOf[a.id];
    if (gid && !dim){
      g.globalAlpha = alpha * .85;
      g.strokeStyle = groupColor(gid); g.lineWidth = 1.6;
      g.beginPath(); g.arc(x, cy, r + 1.6, 0, Math.PI*2); g.stroke();
    }

    if (sel){
      g.globalAlpha = 1; g.strokeStyle = "#0f766e"; g.lineWidth = 2.4;
      g.beginPath(); g.arc(x, cy, r + 3, 0, Math.PI*2); g.stroke();
    }
    g.globalAlpha = 1;
    hits.push({ x, y: cy, r: Math.max(10, r + 4), a });
  }

  // Lien d'affinité entre l'agent sélectionné et l'agent survolé.
  if (S.sel && ATLAS_HOVER && S.sel !== ATLAS_HOVER &&
      S.meta && S.meta.affinity_source){
    const m1 = MOTION.get(S.sel), m2 = MOTION.get(ATLAS_HOVER);
    if (m1 && m2){
      const x1 = pad.l + m1.x * iw, y1 = pad.t + m1.y * ih - m1.hopZ * ih * 1.15;
      const x2 = pad.l + m2.x * iw, y2 = pad.t + m2.y * ih - m2.hopZ * ih * 1.15;
      const value = affinityBetween(S.sel, ATLAS_HOVER);
      const color = affinityColor(value);
      g.save();
      g.strokeStyle = color; g.lineWidth = 2.2; g.setLineDash([6, 4]);
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
      g.setLineDash([]);
      const label = `${affinityLabel(value)} · ${fmt(value)}`;
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      g.font = "600 11px ui-sans-serif"; g.textAlign = "center";
      const tw = g.measureText(label).width;
      g.fillStyle = "rgba(255,255,255,.92)";
      g.fillRect(mx - tw / 2 - 5, my - 9, tw + 10, 18);
      g.fillStyle = color; g.fillText(label, mx, my + 4);
      g.restore();
    }
  }
  ATLAS_HITS = hits;
}

function atlasFrame(now){
  if (!ATLAS_ACTIVE) return;
  const t = now / 1000;
  const dt = ATLAS_LAST ? Math.min(.05, t - ATLAS_LAST) : .016;
  ATLAS_LAST = t; ATLAS_T = t;
  stepAtlas(dt, t);
  drawAtlasFrame(t);
  ATLAS_RAF = requestAnimationFrame(atlasFrame);
}

function startAtlas(){
  if (!S.meta) return;
  initMotion();
  updateAtlasInfo();
  if (!$("#animate").checked){
    stopAtlas(); drawAtlasFrame(ATLAS_T); return;
  }
  ATLAS_ACTIVE = true; ATLAS_LAST = 0;
  if (!ATLAS_RAF) ATLAS_RAF = requestAnimationFrame(atlasFrame);
}

function stopAtlas(){
  ATLAS_ACTIVE = false;
  if (ATLAS_RAF) cancelAnimationFrame(ATLAS_RAF);
  ATLAS_RAF = 0;
}

function nearest(ev){
  const c = $("#scatter"), r = c.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  let best = null, bd = Infinity;
  for (const h of ATLAS_HITS){
    const d = (h.x-x)**2 + (h.y-y)**2;
    const lim = Math.max(12, h.r) ** 2;
    if (d < lim && (!best || d < bd)){ bd = d; best = h; }
  }
  return best;
}

$("#scatter").addEventListener("mousemove", ev => {
  const hit = nearest(ev), tip = $("#tip");
  ATLAS_HOVER = hit ? hit.a.id : null;
  if (hit){
    const tag = hit.a.copies > 1 ? `copie ${hit.a.copy_index}/${hit.a.copies}` : "profil unique";
    const sel = S.sel ? S.agents.find(a => a.id === S.sel) : null;
    let extra = "";
    if (sel && hit.a.id !== sel.id && S.meta && S.meta.affinity_source){
      const value = affinityBetween(sel.id, hit.a.id);
      extra = ` · affinité avec ${sel.name} : ${fmt(value)} (${affinityLabel(value)})`;
    }
    tip.textContent = `${hit.a.name} · ${tag}${extra} · cliquer pour discuter`;
    tip.style.left = (ev.clientX + 12) + "px"; tip.style.top = (ev.clientY - 10) + "px";
    tip.style.opacity = 1; $("#scatter").style.cursor = "pointer";
  } else { tip.style.opacity = 0; $("#scatter").style.cursor = "crosshair"; }
  if (!ATLAS_RAF) drawAtlasFrame(ATLAS_T);
});
$("#scatter").addEventListener("click", ev => { const hit = nearest(ev); if (hit) selectAgent(hit.a.id); });
$("#scatter").addEventListener("mouseleave", () => {
  $("#tip").style.opacity = 0;
  ATLAS_HOVER = null;
  if (!ATLAS_RAF) drawAtlasFrame(ATLAS_T);
});

document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".view").forEach(v => v.classList.remove("on"));
  $("#view-" + b.dataset.view).classList.add("on");
  if (b.dataset.view === "atlas") startAtlas(); else stopAtlas();
  if (b.dataset.view === "survey") updateSurveyScopeInfo();
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopAtlas();
  else if ($("#view-atlas").classList.contains("on")) startAtlas();
});

$("#send").onclick = send;
$("#input").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); send(); }
});
$("#reset").onclick = () => { S.history = []; renderChat(); $("#lat").textContent = ""; };
$("#svRun").onclick = runSurvey;
$("#svScope").onchange = updateSurveyScopeInfo;
$("#svLimit").oninput = updateSurveyScopeInfo;

// ---- sondage ----
let SV_JOB = null, SV_TIMER = 0;

function svScope(){
  const mode = $("#svScope").value;
  const where = [];
  if (mode === "filtered" && S.filter.attr && S.filter.val)
    where.push(S.filter.attr + "=" + S.filter.val);
  return { mode, where };
}

async function loadSurveys(){
  try{
    const d = await fetch("/api/surveys").then(r => r.json());
    const sel = $("#svSurvey"); sel.innerHTML = "";
    if (!d.surveys.length){ sel.append(new Option("(aucun questionnaire dans surveys/)", "")); return; }
    for (const s of d.surveys) sel.append(new Option(`${s.title} · ${s.n_questions} q.`, s.name));
  }catch(e){ /* silencieux */ }
}

function renderSurveyCross(){
  const sel = $("#svCross"); if (!sel) return;
  const keep = sel.value; sel.innerHTML = "";
  sel.append(new Option("—", ""));
  for (const a of S.meta.attrs) sel.append(new Option(attrName(a), a));
  sel.value = keep || "";
}

function updateSurveyScopeInfo(){
  const info = $("#svScopeInfo"); if (!info) return;
  const { mode, where } = svScope();
  $("#svLimitWrap").style.display = mode === "sample" ? "" : "none";
  let text;
  if (mode === "all") text = "500 agents — les copies exactes (×2 à ×48) répondront de façon identique : préfère « un par profil » pour éviter de sur-pondérer un même profil.";
  else if (mode === "one_per_cluster") text = "302 profils distincts — une copie par profil.";
  else if (mode === "filtered") text = where.length
      ? `filtre : ${where.join(", ")} — ${visible().length} agent(s).`
      : "aucun filtre actif : choisis « attribut = valeur » à gauche, sinon tout le monde est interrogé.";
  else text = `échantillon aléatoire de ${$("#svLimit").value || 0} agents.`;
  info.textContent = text;
}

async function runSurvey(){
  const name = $("#svSurvey").value; if (!name) return;
  const { mode, where } = svScope();
  const body = {
    survey: name, provider: provider(), model: model(),
    where, one_per_cluster: mode === "one_per_cluster",
    limit: mode === "sample" ? (+$("#svLimit").value || 30) : null,
    shuffle: $("#svShuffle").checked,
    crosstab: $("#svCross").value ? [$("#svCross").value] : [],
    concurrency: 2,
  };
  $("#svResults").innerHTML = "";
  $("#svRun").disabled = true;
  $("#svProgWrap").style.display = "";
  try{
    const r = await fetch("/api/survey", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || ("HTTP " + r.status));
    SV_JOB = d.job;
    pollSurvey();
  }catch(e){
    $("#svResults").append(el("p", "err", "⚠ " + e.message));
    $("#svRun").disabled = false;
  }
}

async function pollSurvey(){
  if (!SV_JOB) return;
  try{
    const s = await fetch("/api/survey/status?id=" + SV_JOB).then(r => r.json());
    const total = s.total || 0, done = s.done || 0;
    $("#svProgFill").style.width = total ? (100 * done / total) + "%" : "0%";
    $("#svProgTxt").textContent = total
      ? `${done}/${total} requêtes · ${s.n_agents} agents · ${s.provider}/${s.model}`
      : "préparation…";
    if (s.running){ SV_TIMER = setTimeout(pollSurvey, 600); return; }
    $("#svRun").disabled = false;
    if (s.error){ $("#svResults").append(el("p", "err", "⚠ " + s.error)); return; }
    renderSurveyReport(s.report);
  }catch(e){
    $("#svResults").append(el("p", "err", "⚠ " + e.message));
    $("#svRun").disabled = false;
  }
}

function renderSurveyReport(rep){
  const box = $("#svResults"); box.innerHTML = "";
  if (!rep) return;
  box.append(el("p", "meta", `${rep.title} — ${rep.n_agents} agents, ${rep.n_answers} réponses`));
  for (const q of rep.questions){
    const card = el("div", "svq");
    card.append(el("h3", null, `[${q.id}] ${q.text}`));
    if (q.errors) card.append(el("div", "err", q.errors + " erreur(s)"));
    if (q.choices){
      const entries = Object.entries(q.counts || {});
      const max = Math.max(1, ...entries.map(([, n]) => n));
      for (const [choice, n] of entries){
        const row = el("div", "svbar");
        row.append(el("div", null, choice));
        const track = el("div", "track"), fill = el("div", "fill" + (n === max ? " lead" : ""));
        fill.style.width = (100 * n / max) + "%"; track.append(fill); row.append(track);
        row.append(el("div", "pct", (100 * (q.proportions[choice] || 0)).toFixed(1) + " %"));
        card.append(row);
      }
      for (const [attr, tab] of Object.entries(q.crosstab || {})){
        const cw = el("div", "svcross");
        cw.append(el("div", null, "par " + AL(attr) + " :"));
        const choices = Object.keys(q.counts || {});
        const t = el("table"), head = el("tr");
        head.append(el("th", null, AL(attr)));
        for (const c of choices) head.append(el("th", null, c));
        t.append(head);
        for (const [val, counts] of Object.entries(tab)){
          const tr = el("tr"); tr.append(el("td", null, VL(attr, val)));
          for (const c of choices) tr.append(el("td", null, counts[c] || 0));
          t.append(tr);
        }
        cw.append(t); card.append(cw);
      }
    } else {
      for (const s of (q.samples || [])) card.append(el("div", "meta", "· " + s));
    }
    box.append(card);
  }
}

function fillModels(preferred){
  const sel = $("#model");
  const p = provider();
  const keep = preferred || sel.value || localStorage.getItem("model") || "";
  sel.innerHTML = "";
  if (p === "ollama"){
    for (const m of AVAILABLE.ollama)
      sel.append(new Option(`${m.name} · ${m.size_gb} Go`, m.name));
    if (!sel.options.length) sel.append(new Option("llama3.2:1b", "llama3.2:1b"));
  } else if (p === "openai"){
    for (const m of OPENAI_MODELS) sel.append(new Option(m, m));
  } else {
    sel.append(new Option("mock", "mock"));
  }
  const names = Array.from(sel.options, o => o.value);
  const def = { mock: "mock", openai: "gpt-4o-mini", ollama: "llama3.2:1b" }[p];
  sel.value = names.includes(keep) ? keep : (names.includes(def) ? def : names[0]);
  localStorage.setItem("model", sel.value);
}
$("#provider").onchange = () => { fillModels(); health(); };
$("#model").onchange = () => localStorage.setItem("model", model());

async function loadModels(){
  try{
    const d = await fetch("/api/models").then(r => r.json());
    AVAILABLE = d;
    fillModels(localStorage.getItem("model"));
    $("#status").className = "dot " + (d.ollama_up ? "ok" : "");
    $("#status").title = d.ollama_up ? "Ollama joignable" : "Ollama non joignable (mode mock disponible)";
    if (!SAVED_PROVIDER){
      $("#provider").value = d.ollama_up ? "ollama" : "mock";
      fillModels();
      health();
    }
  }catch(e){ /* silencieux */ }
}

async function health(){
  try{
    const h = await fetch("/api/health").then(r => r.json());
    if (provider() === "openai") $("#status").className = "dot " + (h.openai_key ? "ok" : "bad");
    else if (provider() === "ollama") $("#status").className = "dot " + (h.ollama_up ? "ok" : "bad");
    else $("#status").className = "dot " + (h.ollama_up ? "ok" : "");
  }catch(e){}
}

let SIM_POLL = 0;

function simBody(){
  return {
    provider: $("#provider").value,
    model: $("#model").value,
    max_ticks: +$("#simticks").value || 200,
    tick_seconds: +$("#simspeed").value || 0,
    spark_rate: +$("#simspark").value,
    group_rate: +$("#simgroup").value,
    refusal_rate: +$("#simrefusal").value,
    conversations_per_tick: +$("#simburst").value || 1,
    parallel: +$("#simpar").value || 1,
    limit: +$("#simlimit").value || null,
    run_id: ($("#simrun").value || "").trim(),
    reset: $("#simfresh").checked,
  };
}

async function simPost(path, body){
  const r = await fetch(path, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {})
  });
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || "erreur serveur");
  return d;
}

function renderSim(p){
  const state = $("#simstate");
  state.textContent = p.running ? "en cours" : (p.run_id ? "terminée" : "arrêtée");
  state.className = "badge" + (p.running ? " hot" : "");
  $("#simstart").disabled = !!p.running;
  $("#simstop").disabled = !p.running;
  const s = p.stats || {};
  $("#siminfo").textContent = p.run_id
    ? `${p.run_id} · tick ${p.tick} · ${s.conversations || 0} conversation(s) · ` +
      `${p.memory_pairs || 0} mémoire(s)${p.error ? " · erreur" : ""}`
    : "Aucune simulation lancée.";
  $("#siminfo").title = p.error || "";
  const box = $("#simevents"); box.innerHTML = "";
  for (const ev of (p.events || []).slice(-8).reverse()){
    const row = el("div");
    const names = (ev.names || []).join(", ");
    if (ev.type === "refusal"){
      row.append(el("b", null, `t${ev.tick} refus `), document.createTextNode(names));
    } else {
      const tag = ev.group ? ` · ${ev.group}` : "";
      row.append(el("b", null, `t${ev.tick} ${ev.id}${tag} `),
                 document.createTextNode(`${names} — ${ev.topic || ""}`));
    }
    box.append(row);
  }
}

function applySimState(p){
  if (!p || !p.run_id || !S.meta) return;
  if (p.groups) S.meta.groups = p.groups;
  if (p.affinity) S.meta.affinity = p.affinity;
  if (p.opinions) S.meta.opinions = p.opinions;
  S.meta.social_tick = p.tick;
  S.meta.conversation_no = p.conversation_no;
  S.meta.affinity_stats = p.stats || {};
  if (p.run_dir) S.meta.affinity_source = p.run_dir.replace(/\/$/, "") + "/state.json";
  if (p.positions && Object.keys(p.positions).length){
    initMotion();
    for (const [id, xy] of Object.entries(p.positions)){
      const m = MOTION.get(Number(id));
      if (m && Array.isArray(xy)){
        m.tx = clamp(+xy[0], 0, 1); m.ty = clamp(+xy[1], 0, 1);
      }
    }
  }
  renderAffinityStatus();
  if (S.sel){ renderAffinity(S.sel); renderGroup(S.sel); renderOpinions(S.sel); }
  if (!ATLAS_RAF) drawAtlasFrame(ATLAS_T);
}

async function simPoll(){
  clearTimeout(SIM_POLL);
  try {
    const p = await fetch("/api/sim/status").then(r => r.json());
    applySimState(p); renderSim(p);
    if (!p.running) loadRuns();
    if (p.running) SIM_POLL = setTimeout(simPoll, 1000);
  } catch (err){
    $("#siminfo").textContent = "statut indisponible : " + err.message;
    SIM_POLL = setTimeout(simPoll, 2000);
  }
}

async function simStart(){
  $("#simstart").disabled = true;
  try {
    const p = await simPost("/api/sim/start", simBody());
    renderSim(p); applySimState(p); simPoll();
  } catch (err){
    $("#siminfo").textContent = "échec : " + err.message;
    $("#simstart").disabled = false;
  }
}

async function simStop(){
  $("#simstop").disabled = true;
  try {
    const p = await simPost("/api/sim/stop", {});
    renderSim(p); applySimState(p);
  } catch (err){
    $("#siminfo").textContent = "échec : " + err.message;
  }
  simPoll();
}

$("#simstart").onclick = simStart;
$("#simstop").onclick = simStop;

async function simReset(){
  const btn = $("#simreset");
  btn.disabled = true;
  try {
    const p = await simPost("/api/sim/reset", {});
    for (const m of MOTION.values()){ m.tx = null; m.ty = null; }
    if (S.meta){
      S.meta.affinity = p.affinity || {};
      S.meta.groups = p.groups || {};
      S.meta.opinions = p.opinions || {};
      S.meta.affinity_stats = p.affinity_stats || {};
      S.meta.affinity_source = null;
      S.meta.social_tick = 0;
      S.meta.conversation_no = 0;
    }
    renderAffinityStatus();
    if (S.sel){ renderAffinity(S.sel); renderGroup(S.sel); renderOpinions(S.sel); }
    renderSim({running: false, run_id: null, tick: 0, stats: {},
               memory_pairs: 0, events: [], error: null});
    if (!ATLAS_RAF) drawAtlasFrame(ATLAS_T);
  } catch (err){
    $("#siminfo").textContent = "échec : " + err.message;
  } finally {
    btn.disabled = false;
  }
}
$("#simreset").onclick = simReset;

async function loadRuns(){
  try {
    const d = await fetch("/api/runs").then(r => r.json());
    const sel = $("#simruns");
    const current = sel.value;
    sel.innerHTML = "";
    sel.append(new Option("— choisir un run —", ""));
    for (const run of (d.runs || [])){
      const label = `${run.run_id} · t${run.tick} · ${run.conversations} conv.`;
      sel.append(new Option(label, run.run_id));
    }
    if (current) sel.value = current;
  } catch (err){ /* silencieux */ }
}

async function simLoad(){
  const id = $("#simruns").value;
  if (!id){ $("#siminfo").textContent = "choisis un run à charger"; return; }
  try {
    const p = await simPost("/api/sim/load", {run_id: id});
    applySimState(p); renderSim(p);
    $("#simrun").value = id;
  } catch (err){
    $("#siminfo").textContent = "échec : " + err.message;
  }
}

async function simResume(){
  const id = $("#simruns").value || ($("#simrun").value || "").trim();
  if (!id){ $("#siminfo").textContent = "choisis un run à reprendre"; return; }
  const body = simBody();
  body.run_id = id; body.resume = id; body.reset = false;
  try {
    const p = await simPost("/api/sim/start", body);
    renderSim(p); applySimState(p); simPoll();
  } catch (err){
    $("#siminfo").textContent = "échec : " + err.message;
  }
  loadRuns();
}

async function simDelete(){
  const id = $("#simruns").value;
  if (!id){ $("#siminfo").textContent = "choisis un run à supprimer"; return; }
  if (!confirm(`Supprimer le run ${id} et tous ses fichiers ?`)) return;
  try {
    await simPost("/api/runs/delete", {run_id: id});
    $("#simruns").value = "";
    if (S.meta && S.meta.affinity_source &&
        S.meta.affinity_source.includes("/" + id + "/")){
      const p = await simPost("/api/sim/reset", {});
      applySimState({run_id: null});
      for (const m of MOTION.values()){ m.tx = null; m.ty = null; }
      renderSim({running: false, run_id: null, tick: 0, stats: {},
                 memory_pairs: 0, events: [], error: null});
      if (S.meta){
        S.meta.affinity = {}; S.meta.groups = {}; S.meta.opinions = {};
        S.meta.affinity_stats = {}; S.meta.affinity_source = null;
        S.meta.social_tick = 0; S.meta.conversation_no = 0;
      }
      renderAffinityStatus();
      if (S.sel){ renderAffinity(S.sel); renderGroup(S.sel); renderOpinions(S.sel); }
    }
    loadRuns();
  } catch (err){
    $("#siminfo").textContent = "échec : " + err.message;
  }
}

$("#simload").onclick = simLoad;
$("#simresume").onclick = simResume;
$("#simdelete").onclick = simDelete;
$("#simexport").onclick = () => {
  if (!S.meta || !S.meta.affinity_source){
    $("#siminfo").textContent = "aucun état à exporter";
    return;
  }
  window.location = "/api/sim/export";
};

const SAVED_PROVIDER = localStorage.getItem("provider");
$("#provider").value = SAVED_PROVIDER || "mock";
fillModels(localStorage.getItem("model"));
new ResizeObserver(() => { if (!ATLAS_RAF) drawAtlasFrame(ATLAS_T); }).observe($("#scatter"));
boot();
simPoll();
loadRuns();
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=Path, default=Path("agents/agents.jsonl"))
    p.add_argument("--surveys", type=Path, default=Path("surveys"),
                   help="dossier des questionnaires JSON (défaut : surveys)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--social-state", type=Path, default=None,
                   help="state.json d'un run social (défaut : le plus récent)")
    p.add_argument("--open", action="store_true", help="ouvre le navigateur")
    args = p.parse_args()
    if not args.agents.exists():
        print(f"Fichier absent : {args.agents}\n"
              "Lance d'abord : uv run python tools/build_agents.py constraint_cells.csv population.csv",
              file=sys.stderr)
        return 1
    httpd = None
    chosen = args.port
    if args.port == 0:                       # port libre choisi par l'OS
        httpd = create_server(args.agents, args.host, 0, args.surveys,
                              args.social_state)
        chosen = httpd.server_address[1]
    else:                                    # port occupé : on essaie les 10 suivants
        for p in range(args.port, args.port + 10):
            try:
                httpd = create_server(args.agents, args.host, p, args.surveys,
                                      args.social_state)
                chosen = p
                break
            except OSError as e:
                if e.errno != errno.EADDRINUSE:
                    print(f"Impossible d'ouvrir le port : {e}", file=sys.stderr)
                    return 1
        if httpd is None:
            print(f"Ports {args.port} à {args.port + 9} occupés. "
                  f"Choisis-en un autre : --port 8888", file=sys.stderr)
            return 1
        if chosen != args.port:
            print(f"Port {args.port} occupé (autre processus), j'utilise {chosen}.")
    url = f"http://{args.host}:{chosen}/"
    print(f"GUI : {url}  ({len(httpd.state.agents)} agents, Ctrl+C pour arrêter)")
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\narrêt.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
