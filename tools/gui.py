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
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_agent  # noqa: E402

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
    def __init__(self, agents_path: Path):
        self.agents = chat_agent.load_agents(agents_path)
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
        reps = {}
        for ag in self.agents:
            key = tuple(ag["attributes"][a] for a in self.attrs)
            reps.setdefault(key, ag["id"])
        self.duplicates = sorted(
            ({"key": "|".join(k), "count": c, "agent_id": reps[k],
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
                "distinct": self.distinct,
                "entropy": self.entropy,
                "effective": math.exp(self.entropy),
                "max_mult": self.max_mult,
                "extra_copies": len(self.agents) - self.distinct,
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
            "attributes": {a: ag["attributes"][a] for a in self.attrs},
            "mult": self.mult[ag["id"]],
        } for ag in self.agents]}


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

    def do_POST(self):
        if self.path != "/api/chat":
            return self._error("not found", 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                return self._error("payload trop gros", 413)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            return self._error("JSON invalide")

        agent = self.server.state.by_id.get(body.get("agent_id"))
        if agent is None:
            return self._error("agent inconnu", 404)
        message = (body.get("message") or "").strip()
        if not message:
            return self._error("message vide")
        provider = body.get("provider") or "mock"
        if provider not in ("mock", "openai", "ollama"):
            return self._error("fournisseur invalide")
        model = (body.get("model") or "").strip() or (
            {"openai": "gpt-4o-mini", "ollama": "llama3.2:1b"}.get(provider, "mock"))

        history = []
        for m in (body.get("history") or [])[-24:]:
            if (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content"), str)):
                history.append({"role": m["role"], "content": m["content"][:4000]})
        history.append({"role": "user", "content": message[:4000]})

        t0 = time.perf_counter()
        try:
            answer = chat_agent.respond(agent, history, provider, model)
        except chat_agent.LLMError as e:
            return self._error(str(e), 502)
        except Exception as e:  # noqa: BLE001 — la GUI ne doit jamais tomber
            return self._error(f"{type(e).__name__}: {e}", 500)
        ms = round((time.perf_counter() - t0) * 1000)
        self._json({"answer": answer, "ms": ms, "provider": provider, "model": model})


def create_server(agents: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.state = State(agents)
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
  #scatter{width:100%;height:calc(100vh - 210px);min-height:320px;display:block;cursor:crosshair}
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
      <button data-view="dups">Doublons</button>
    </div>
    <div class="view on" id="view-atlas">
      <div class="scatterbar">
        <label>X <select id="xa"></select></label>
        <label>Y <select id="ya"></select></label>
        <label><input type="checkbox" id="jitter" checked> bruit</label>
        <span id="atlasinfo"></span>
      </div>
      <canvas id="scatter"></canvas>
    </div>
    <div class="view" id="view-attrs">
      <table class="heat" id="heat"></table>
      <p class="meta" style="margin-top:12px">Chaque cellule donne le nombre d'agents. Les marges imposent ces distributions : elles ne dépendent pas de la génération.</p>
    </div>
    <div class="view" id="view-dups">
      <p class="meta" style="margin-top:0">Personas portées par plusieurs agents. Une identité unique vient de <code>prenom+nom</code>, pas de ces attributs contraints.</p>
      <div id="dups"></div>
    </div>
  </section>

  <aside class="panel chat" id="chat">
    <div class="chead" id="chead">
      <div><div class="nm">Aucun agent</div><div class="sub">choisis un agent à gauche</div></div>
    </div>
    <div class="attrs" id="cattrs" style="display:none"></div>
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
const COLOR = { unique: "#a8b2bc", clone: "#c2410c" };

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
  renderList(); requestAnimationFrame(renderScatter);
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
  add("N", s.n, "individus");
  add("distinctes", s.distinct, "personas distinctes");
  add("H", fmt(s.entropy, 3) + " nats", "entropie de Shannon");
  add("e^H", fmt(s.effective, 0), "nombre effectif de personas");
  add("max", "×" + s.max_mult, "multiplicité maximale");
  add("copies en trop", s.extra_copies, "N − distinctes");
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
    renderList();
  };
  $("#fval").onchange = e => { S.filter.val = e.target.value; renderList(); };
  $("#q").oninput = e => { S.filter.q = e.target.value.toLowerCase().trim(); renderList(); };
}

function renderAttrSelects(){
  const xs = $("#xa"), ys = $("#ya");
  S.meta.attrs.forEach((a, i) => {
    xs.append(new Option(attrName(a), a)); ys.append(new Option(attrName(a), a));
  });
  xs.value = S.meta.attrs[0]; ys.value = S.meta.attrs[1] || S.meta.attrs[0];
  xs.onchange = ys.onchange = () => renderScatter();
  $("#jitter").onchange = () => renderScatter();
}

function visible(){
  const f = S.filter;
  return S.agents.filter(a => {
    if (f.attr && f.val && a.attributes[f.attr] !== f.val) return false;
    if (!f.q) return true;
    return (a.name + " " + a.identity_key + " " + a.id).toLowerCase().includes(f.q);
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
    who.append(el("div", "sub", "#" + a.id + " · " + Object.entries(a.attributes).slice(0, 3).map(([k, v]) => VL(k, v)).join(" · ")));
    d.append(who);
    const b = el("span", "badge" + (a.mult > 1 ? " hot" : ""), a.mult > 1 ? "×" + a.mult : "1");
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
  box.append(el("div", "sub", "#" + a.id + " · " + a.identity_key + (a.mult > 1 ? " · persona ×" + a.mult : " · persona unique")));
  head.append(av, box);
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
}

async function send(){
  if (!S.sel) return;
  const input = $("#input"); const text = input.value.trim(); if (!text) return;
  input.value = "";
  S.history.push({ role: "user", content: text }); renderChat();
  $("#send").disabled = true; $("#lat").textContent = "…";
  const t0 = performance.now();
  try{
    const r = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_id: S.sel, message: text, history: S.history.slice(0, -1).filter(m => !m.error),
        provider: provider(), model: model()
      })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
    S.history.push({ role: "assistant", content: data.answer });
    $("#lat").textContent = `${data.provider}/${data.model} · ${data.ms} ms`;
  }catch(e){
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
    const a = S.agents.find(x => x.id === d.agent_id);
    const row = el("div", "duprow");
    row.append(el("div", "n", "×" + d.count));
    const mid = el("div"); mid.style.flex = "1";
    mid.append(el("div", null, a.name + "  #" + a.id));
    mid.append(el("div", "a", Object.entries(d.attrs).map(([k, v]) => AL(k) + " : " + VL(k, v)).join(" · ")));
    row.append(mid);
    row.onclick = () => selectAgent(d.agent_id);
    box.append(row);
  }
}

function renderScatter(){
  const c = $("#scatter"); if (!c || !S.meta) return;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (!w || !h) return;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const xa = $("#xa").value, ya = $("#ya").value;
  const dx = S.meta.domains[xa], dy = S.meta.domains[ya];
  const pad = { l: 52, r: 18, t: 18, b: 42 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const px = i => pad.l + (dx.length === 1 ? iw/2 : iw * i / (dx.length - 1));
  const py = i => pad.t + (dy.length === 1 ? ih/2 : ih * i / (dy.length - 1));

  g.strokeStyle = "#e6dfd2"; g.lineWidth = 1;
  for (let i = 0; i < dx.length; i++){ g.beginPath(); g.moveTo(px(i), pad.t); g.lineTo(px(i), pad.t + ih); g.stroke(); }
  for (let i = 0; i < dy.length; i++){ g.beginPath(); g.moveTo(pad.l, py(i)); g.lineTo(pad.l + iw, py(i)); g.stroke(); }
  g.fillStyle = "#6b7280"; g.font = "11px ui-sans-serif";
  g.textAlign = "center";
  dx.forEach((v, i) => g.fillText(v, px(i), h - 22));
  g.save(); g.translate(16, pad.t + ih/2); g.rotate(-Math.PI/2); g.fillText(ya, 0, 0); g.restore();
  g.fillText(xa, pad.l + iw/2, h - 4);

  const jitterOn = $("#jitter").checked;
  const rnd = id => { let x = Math.sin(id * 999.91) * 10000; return x - Math.floor(x); };
  const hits = [];
  for (const a of S.agents){
    const i = dx.indexOf(a.attributes[xa]), j = dy.indexOf(a.attributes[ya]);
    if (i < 0 || j < 0) continue;
    const jx = jitterOn ? (rnd(a.id) - .5) * (iw / Math.max(1, dx.length) * .55) : 0;
    const jy = jitterOn ? (rnd(a.id + 77) - .5) * (ih / Math.max(1, dy.length) * .55) : 0;
    const x = px(i) + jx, y = py(j) + jy;
    const sel = a.id === S.sel;
    g.beginPath(); g.arc(x, y, sel ? 7 : (a.mult > 1 ? 4.2 : 3), 0, 7);
    g.fillStyle = sel ? "#111827" : (a.mult > 1 ? COLOR.clone : COLOR.unique);
    g.globalAlpha = sel ? 1 : (a.mult > 1 ? .78 : .55);
    g.fill(); g.globalAlpha = 1;
    if (sel){ g.strokeStyle = "#fff"; g.lineWidth = 2; g.stroke(); }
    hits.push({ x, y, a });
  }
  $("#atlasinfo").textContent = `${S.agents.length} agents · coloration : orange = persona dupliquée`;
  c._hits = hits;
}

function nearest(ev){
  const c = $("#scatter"), r = c.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  let best = null, bd = 100;
  for (const h of (c._hits || [])){
    const d = (h.x-x)**2 + (h.y-y)**2;
    if (d < bd){ bd = d; best = h; }
  }
  return best;
}

$("#scatter").addEventListener("mousemove", ev => {
  const hit = nearest(ev), tip = $("#tip");
  if (hit){
    tip.textContent = `${hit.a.name} · ×${hit.a.mult}`;
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
  if (b.dataset.view === "atlas") renderScatter();
});

$("#send").onclick = send;
$("#input").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); send(); }
});
$("#reset").onclick = () => { S.history = []; renderChat(); $("#lat").textContent = ""; };

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
new ResizeObserver(() => renderScatter()).observe($("#scatter"));
boot();
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=Path, default=Path("agents/agents.jsonl"))
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
        httpd = create_server(args.agents, args.host, 0)
        chosen = httpd.server_address[1]
    else:                                    # port occupé : on essaie les 10 suivants
        for p in range(args.port, args.port + 10):
            try:
                httpd = create_server(args.agents, args.host, p)
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
