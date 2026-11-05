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
    def __init__(self, agents_path: Path, surveys_dir: Path | None = None):
        self.agents = chat_agent.load_agents(agents_path)
        self.surveys_dir = surveys_dir or (agents_path.parent.parent / "surveys")
        self.jobs: dict[str, dict] = {}
        self.jobs_lock = threading.Lock()
        self.by_id = {a["id"]: a for a in self.agents}
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
        }

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
                  surveys: Path | None = None) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.state = State(agents, surveys)
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
        <span id="atlasinfo"></span>
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
function AL(a){ return (S.meta && S.meta.labels && S.meta.labels[a]) || a; }
function VL(a, v){ return (S.meta && S.meta.labels && S.meta.labels[a + "=" + v]) || v; }
function attrName(a){ const l = AL(a); return l === a ? a : a + " — " + l; }

async function boot(){
  const [meta, data] = await Promise.all([
    fetch("/api/meta").then(r => r.json()),
    fetch("/api/agents").then(r => r.json())
  ]);
  S.meta = meta; S.agents = data.agents;
  renderStats(); renderFilters(); renderAttrSelects(); renderHeat(); renderDups();
  renderList(); renderSurveyCross(); loadSurveys();
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

    if (sel){
      g.globalAlpha = 1; g.strokeStyle = "#0f766e"; g.lineWidth = 2.4;
      g.beginPath(); g.arc(x, cy, r + 3, 0, Math.PI*2); g.stroke();
    }
    g.globalAlpha = 1;
    hits.push({ x, y: cy, r: Math.max(10, r + 4), a });
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
  if (hit){
    const tag = hit.a.copies > 1 ? `copie ${hit.a.copy_index}/${hit.a.copies}` : "profil unique";
    tip.textContent = `${hit.a.name} · ${tag} · cliquer pour discuter`;
    tip.style.left = (ev.clientX + 12) + "px"; tip.style.top = (ev.clientY - 10) + "px";
    tip.style.opacity = 1; $("#scatter").style.cursor = "pointer";
  } else { tip.style.opacity = 0; $("#scatter").style.cursor = "crosshair"; }
});
$("#scatter").addEventListener("click", ev => { const hit = nearest(ev); if (hit) selectAgent(hit.a.id); });
$("#scatter").addEventListener("mouseleave", () => { $("#tip").style.opacity = 0; });

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

const SAVED_PROVIDER = localStorage.getItem("provider");
$("#provider").value = SAVED_PROVIDER || "mock";
fillModels(localStorage.getItem("model"));
new ResizeObserver(() => { if (!ATLAS_RAF) drawAtlasFrame(ATLAS_T); }).observe($("#scatter"));
boot();
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
        httpd = create_server(args.agents, args.host, 0, args.surveys)
        chosen = httpd.server_address[1]
    else:                                    # port occupé : on essaie les 10 suivants
        for p in range(args.port, args.port + 10):
            try:
                httpd = create_server(args.agents, args.host, p, args.surveys)
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
