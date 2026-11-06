"""Durable local lifecycle gates. No network listener or automatic publication."""
import contextlib
import getpass
import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path

from .source import GitSource, KnowledgeError, canonical, digest, git, safe_path


def now():
    return datetime.now(timezone.utc).isoformat()


def serialized(method):
    """Serialize lifecycle writers across processes, outside SQLite transactions."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self.operation_lock():
            return method(self, *args, **kwargs)
    return guarded


def code_snapshot(repo):
    revision = git(repo, "rev-parse", "HEAD").decode().strip()
    changed = git(repo, "diff", "--binary", "HEAD", "--", ".",
                  ":(exclude).knowledge-local", ":(exclude).local")
    paths = git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    untracked = {}
    for raw in paths:
        if not raw:
            continue
        name = raw.decode("utf-8")
        path = Path(repo) / name
        if path.is_symlink():
            raise KnowledgeError("Untracked symlinks are not supported")
        untracked[name] = digest(path.read_bytes())
    return {"base_revision": revision, "tracked_diff_sha256": digest(changed),
            "changed_paths": git(repo, "diff", "--name-only", "HEAD").decode().splitlines(),
            "untracked": untracked}


def make_tree(repo, directory):
    """Build isolated Git objects, never touching the active checkout or index."""
    directory = Path(directory).resolve()
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    with tempfile.TemporaryDirectory() as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))

        def index(*args):
            import subprocess
            result = subprocess.run(["git", "-C", str(repo), *args], env=env,
                                    capture_output=True)
            if result.returncode:
                raise KnowledgeError(result.stderr.decode("utf-8", "replace"))
            return result.stdout.decode().strip()

        index("read-tree", "--empty")
        for path in files:
            if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != directory.parent):
                raise KnowledgeError("Candidate symlinks are not supported")
            relative = safe_path(path.relative_to(directory).as_posix())
            if relative.startswith((".local/", ".knowledge-local/")):
                continue
            blob = git(repo, "hash-object", "-w", "--stdin", data=path.read_bytes()).decode().strip()
            index("update-index", "--add", "--cacheinfo", f"100644,{blob},{relative}")
        return index("write-tree")


class Service:
    """Single OS-principal service boundary; metadata filters never grant access.

    The SQLite directory must be protected by OS permissions. It is not a remotely
    authenticated multi-user service. An approval is an explicit reviewer attestation,
    not proof that an automated process could not use the same OS account.
    """
    def __init__(self, repo, database, policy, *, actor=None):
        self.source = GitSource(repo)
        self.repo = self.source.repo
        self.policy = policy
        self.policy_digest = digest(policy)
        self.actor = actor or getpass.getuser()
        if self.actor not in policy["principals"]:
            raise KnowledgeError("Principal is not authorized")
        self.remote = git(self.repo, "remote", "get-url", "origin").decode().strip()
        self.repository = digest(self.remote.encode())
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.thread_lock = threading.RLock()
        self.lock_depth = 0
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records(kind TEXT, id TEXT, payload TEXT NOT NULL,
                    PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, at TEXT, actor TEXT,
                    event TEXT, payload TEXT);
            """)
            binding = db.execute("SELECT value FROM metadata WHERE key='repository'").fetchone()
            if binding and binding[0] != self.repository:
                raise KnowledgeError("State store belongs to another repository")
            db.execute("INSERT OR IGNORE INTO metadata VALUES('repository',?)", (self.repository,))

    @contextlib.contextmanager
    def operation_lock(self):
        with self.thread_lock:
            if self.lock_depth:
                self.lock_depth += 1
                try:
                    yield
                finally:
                    self.lock_depth -= 1
                return
            path = self.database.with_suffix(".lock")
            with path.open("a+b") as handle:
                if handle.seek(0, 2) == 0:
                    handle.write(b"0")
                    handle.flush()
                deadline = time.monotonic() + 15
                while True:
                    handle.seek(0)
                    try:
                        if os.name == "nt":
                            import msvcrt
                            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise KnowledgeError("Another lifecycle operation holds the state lock; retry by status")
                        time.sleep(0.05)
                self.lock_depth = 1
                try:
                    yield
                finally:
                    self.lock_depth = 0
                    handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextlib.contextmanager
    def connection(self):
        db = sqlite3.connect(self.database, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def load(self, kind, identity):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM records WHERE kind=? AND id=?",
                             (kind, identity)).fetchone()
        if not row:
            raise KnowledgeError(f"Unknown {kind}: {identity}")
        return json.loads(row[0])

    def save(self, kind, identity, payload):
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?)",
                       (kind, identity, canonical(payload).decode()))
            db.execute("INSERT INTO audit(at,actor,event,payload) VALUES(?,?,?,?)",
                       (now(), self.actor, kind, canonical({"id": identity, **payload}).decode()))
        return payload

    def remote_tip(self):
        if git(self.repo, "remote", "get-url", "origin").decode().strip() != self.remote:
            raise KnowledgeError("Repository remote changed; reopen the service")
        lines = git(self.repo, "ls-remote", "--heads", "origin", self.source.ref).decode().splitlines()
        return lines[0].split()[0] if lines else None

    @serialized
    def reconcile(self):
        tip = self.remote_tip()
        if tip:
            git(self.repo, "fetch", "--no-tags", "origin", self.source.ref)
            self.source.snapshot(tip)
        return self.save("observation", "latest", {"revision": tip, "observed_at": now(),
                                                   "remote": self.remote})

    @serialized
    def start(self, task_id, epoch, *, scope="project", query="", previous=None):
        if not task_id or not epoch:
            raise KnowledgeError("Task and session epoch are required")
        try:
            old = self.load("task", task_id)
        except KnowledgeError:
            old = None
        if old and old["epoch"] == epoch:
            existing = self.load("offer", old["offer"])
            if (existing.get("scope") != scope or existing.get("query") != query
                    or existing["code_snapshot"] != code_snapshot(self.repo)
                    or existing["knowledge_revision"] != self.source.revision()):
                raise KnowledgeError("Announce inputs changed; use a new session epoch")
            return existing
        snapshot = self.source.snapshot()
        _, relevant = self.source.search(query, revision=snapshot["revision"], scope=scope)
        selected = {d["id"]: d for d in snapshot["documents"] if d.get("mandatory")}
        selected.update({d["id"]: d for d in relevant})
        manifest = [{k: d[k] for k in ("id", "path", "section", "sha256", "bytes")}
                    for d in sorted(selected.values(), key=lambda d: d["id"])]
        offer = {"id": uuid.uuid4().hex, "task": task_id, "epoch": epoch,
                 "repository": self.repository, "knowledge_revision": snapshot["revision"],
                 "scope": scope, "query": query,
                 "previous_revision": previous, "code_snapshot": code_snapshot(self.repo),
                 "manifest": manifest, "manifest_digest": digest(manifest),
                 "context": [{"id": d["id"], "text": d["text"]} for d in sorted(selected.values(), key=lambda d: d["id"])],
                 "created_at": now()}
        self.save("offer", offer["id"], offer)
        state = old or {"id": task_id, "code": "pending", "knowledge": "unassessed", "invalid": False}
        state.update(epoch=epoch, offer=offer["id"], receipt=None, freshness={})
        state.pop("overall", None)
        self.save("task", task_id, state)
        return offer

    @serialized
    def record_request(self, task_id, epoch, offer_id, manifest_digest, request):
        task = self.load("task", task_id)
        offer = self.load("offer", offer_id)
        if (task["epoch"] != epoch or task["offer"] != offer_id
                or offer["repository"] != self.repository
                or offer["knowledge_revision"] != self.source.revision()
                or manifest_digest != offer["manifest_digest"] or task["invalid"]):
            raise KnowledgeError("Stale, invalidated or mismatched context acknowledgement")
        # Validate against Git objects again; never trust client-supplied context text.
        self.source.snapshot(offer["knowledge_revision"])
        messages = request.get("messages", [])
        supplied = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
        if any(d["text"] not in supplied for d in offer["context"]):
            raise KnowledgeError("Recorded model request is missing offered content")
        identity = digest({"offer": offer_id, "request": request})
        try:
            existing = self.load("receipt", identity)
        except KnowledgeError:
            existing = None
        if existing:
            task["receipt"] = identity
            self.save("task", task_id, task)
            return existing
        self.save("request", identity, {"request": request, "sha256": digest(request),
                                       "recorded_at": now()})
        receipt = {"id": identity, "task": task_id, "epoch": epoch, "offer": offer_id,
                   "repository": self.repository, "manifest_digest": manifest_digest,
                   "request_digest": digest(request), "acknowledged_at": now()}
        self.save("receipt", identity, receipt)
        task["receipt"] = identity
        self.save("task", task_id, task)
        return receipt

    @serialized
    def freshness(self, task_id, boundary, *, offline=False):
        if boundary not in ("start", "phase", "consequential", "completion"):
            raise KnowledgeError("Unknown freshness boundary")
        task = self.load("task", task_id)
        offer = self.load("offer", task["offer"])
        current = self.source.revision()
        remote = None if offline else self.reconcile()
        if current != offer["knowledge_revision"] or (remote and remote["revision"] not in (None, current)):
            raise KnowledgeError("Knowledge changed; reload context under a new session epoch")
        observation = {"boundary": boundary, "knowledge_revision": current,
                       "code_snapshot": code_snapshot(self.repo), "offline": offline,
                       "remote": remote, "observed_at": now()}
        task["freshness"][boundary] = observation
        self.save("task", task_id, task)
        return observation

    @serialized
    def gate(self, task_id, boundary="consequential", *, operation="read"):
        task = self.load("task", task_id)
        offer = self.load("offer", task["offer"])
        if task["invalid"] or not task["receipt"]:
            raise KnowledgeError("A current instrumented receipt is required")
        receipt = self.load("receipt", task["receipt"])
        if receipt["offer"] != task["offer"] or receipt["epoch"] != task["epoch"]:
            raise KnowledgeError("Receipt is not current")
        observation = task["freshness"].get(boundary)
        if (not observation or observation["knowledge_revision"] != self.source.revision()
                or observation["code_snapshot"] != code_snapshot(self.repo)):
            raise KnowledgeError("Freshness observation does not match this boundary and code snapshot")
        if observation["offline"] and operation not in self.policy["offline_operations"]:
            raise KnowledgeError("Operation is not allowed with cached offline context")
        if offer["manifest_digest"] != receipt["manifest_digest"]:
            raise KnowledgeError("Context manifest mismatch")
        return task

    @serialized
    def result(self, task_id, code_status, *, no_update_reason=None):
        if code_status not in ("complete", "rolled_back"):
            raise KnowledgeError("Invalid code delivery state")
        task = self.load("task", task_id)
        task["code"] = code_status
        if no_update_reason:
            if len(no_update_reason.strip()) < 15:
                raise KnowledgeError("Provide a concrete no-update reason")
            task.update(knowledge="not_required", no_update_reason=no_update_reason)
        else:
            task["knowledge"] = "pending"
        return self.save("task", task_id, task)

    @serialized
    def propose(self, task_id, directory, evidence, *, semantic_class="interpretation"):
        if semantic_class not in ("observation", "interpretation", "decision", "proposal"):
            raise KnowledgeError("Unsupported semantic class")
        self.gate(task_id, operation="propose")
        if not isinstance(evidence, dict) or not evidence:
            raise KnowledgeError("A non-empty evidence record is required")
        task = self.load("task", task_id)
        base = self.source.revision()
        tree = make_tree(self.repo, directory)
        candidate = git(self.repo, "commit-tree", tree, "-p", base,
                        data=b"docs: prepare knowledge proposal\n").decode().strip()
        self.source.snapshot(candidate)
        files = self.source.files(candidate)
        content = {p: digest(self.source.blob(candidate, p)) for p in files}
        proposal = {"id": uuid.uuid4().hex, "task": task_id, "base": base,
                    "tree": tree, "candidate": candidate, "content_digest": digest(content),
                    "evidence": evidence, "evidence_digest": digest(evidence),
                    "policy_digest": self.policy_digest, "class": semantic_class,
                    "implementation": code_snapshot(self.repo), "remote_base": self.remote_tip(),
                    "review": "pending", "authorization": None, "delivery": "pending",
                    "created_at": now()}
        proposal["fingerprint"] = digest({k: proposal[k] for k in (
            "id", "base", "tree", "candidate", "content_digest", "evidence_digest", "policy_digest",
            "implementation", "remote_base", "class")})
        self.save("proposal", proposal["id"], proposal)
        task.update(knowledge="pending", proposal=proposal["id"])
        self.save("task", task_id, task)
        return proposal

    @serialized
    def review(self, proposal_id, fingerprint, decision, attestation):
        proposal = self.load("proposal", proposal_id)
        if self.actor not in self.policy["reviewers"]:
            raise KnowledgeError("Principal is not an authorized reviewer")
        if fingerprint != proposal["fingerprint"] or proposal["policy_digest"] != self.policy_digest:
            raise KnowledgeError("Review does not match proposal or current policy")
        if decision not in ("approved", "rejected", "needs_evidence", "narrowing") or not attestation.strip():
            raise KnowledgeError("Explicit review decision and attestation are required")
        proposal.update(review=decision, reviewer=self.actor, review_reference=attestation,
                        reviewed_fingerprint=fingerprint)
        return self.save("proposal", proposal_id, proposal)

    @serialized
    def authorize(self, proposal_id, fingerprint, reference):
        proposal = self.load("proposal", proposal_id)
        if self.actor not in self.policy["publishers"] or not reference.strip():
            raise KnowledgeError("Explicit publication authorization is required")
        if fingerprint != proposal["fingerprint"]:
            raise KnowledgeError("Authorization digest mismatch")
        proposal["authorization"] = {"fingerprint": fingerprint, "reference": reference,
                                     "actor": self.actor, "at": now()}
        return self.save("proposal", proposal_id, proposal)

    @serialized
    def invalidate(self, task_id, reason):
        if not reason.strip():
            raise KnowledgeError("Invalidation requires a reason")
        task = self.load("task", task_id)
        task.update(invalid=True, invalidation_reason=reason, receipt=None)
        if task.get("proposal"):
            proposal = self.load("proposal", task["proposal"])
            proposal.update(review="needs_evidence", delivery="pending", authorization=None)
            self.save("proposal", proposal["id"], proposal)
        return self.save("task", task_id, task)

    def _verify_remote(self, proposal, commit):
        actual = self.remote_tip()
        if not actual:
            return False
        git(self.repo, "fetch", "--no-tags", "origin", self.source.ref)
        try:
            git(self.repo, "merge-base", "--is-ancestor", commit, actual)
        except KnowledgeError:
            return False
        desired = self.source.files(proposal["candidate"])
        removed = set(self.source.files(proposal["base"])) - set(desired)
        actual_files = set(self.source.files(actual))
        if removed & actual_files:
            return False
        for path in desired:
            if path not in actual_files or self.source.blob(actual, path) != self.source.blob(proposal["candidate"], path):
                return False
        proposal.update(delivery="remote_verified", published_commit=commit,
                        remote_verification={"revision": actual, "at": now(), "remote": self.remote})
        self.save("proposal", proposal["id"], proposal)
        return True

    @serialized
    def publish(self, proposal_id):
        proposal = self.load("proposal", proposal_id)
        if (proposal["review"] != "approved" or proposal.get("reviewed_fingerprint") != proposal["fingerprint"]
                or not proposal["authorization"] or proposal["authorization"]["fingerprint"] != proposal["fingerprint"]
                or proposal["policy_digest"] != self.policy_digest):
            raise KnowledgeError("Matching review, policy and publication authorization are required")
        if self.actor not in self.policy["publishers"]:
            raise KnowledgeError("Principal cannot publish")
        if self.load("task", proposal["task"])["invalid"]:
            raise KnowledgeError("Evidence has been invalidated")
        self.source.snapshot(proposal["candidate"])
        # Recover a crash/lost push response before creating or sending another update.
        try:
            attempt = self.load("attempt", proposal_id)
        except KnowledgeError:
            attempt = None
        if attempt and self._verify_remote(proposal, attempt["commit"]):
            current = self.source.revision()
            if current == proposal["base"]:
                git(self.repo, "update-ref", self.source.ref, attempt["commit"], current)
            return self.load("proposal", proposal_id)
        self.gate(proposal["task"], operation="publish")
        if proposal["implementation"] != code_snapshot(self.repo):
            raise KnowledgeError("Implementation changed after review")
        if self.source.revision() != proposal["base"] or self.remote_tip() != proposal["remote_base"]:
            raise KnowledgeError("Stale proposal: reconcile concurrent knowledge changes")
        if proposal["remote_base"] not in (None, proposal["base"]):
            raise KnowledgeError("Remote baseline is not the reviewed knowledge base")
        if not attempt:
            commit = git(self.repo, "commit-tree", proposal["tree"], "-p", proposal["base"],
                         data=f"docs: publish knowledge proposal {proposal_id}\n".encode()).decode().strip()
            attempt = self.save("attempt", proposal_id, {"commit": commit, "state": "prepared",
                                                        "idempotency_key": proposal_id, "at": now()})
        # No force push: normal Git ancestry rules reject contention.
        git(self.repo, "push", "origin", f"{attempt['commit']}:{self.source.ref}")
        self.save("attempt", proposal_id, {**attempt, "state": "push_returned"})
        if not self._verify_remote(proposal, attempt["commit"]):
            raise KnowledgeError("Push returned but remote content was not verified")
        git(self.repo, "update-ref", self.source.ref, attempt["commit"], proposal["base"])
        return self.load("proposal", proposal_id)

    @serialized
    def complete(self, task_id):
        task = self.gate(task_id, boundary="completion", operation="complete")
        if task["code"] not in ("complete", "rolled_back"):
            raise KnowledgeError("Code delivery is not complete")
        if task["knowledge"] != "not_required":
            if not task.get("proposal"):
                raise KnowledgeError("Required knowledge delivery has no proposal")
            proposal = self.load("proposal", task["proposal"])
            if (proposal["delivery"] != "remote_verified" or proposal["review"] != "approved"
                    or proposal["policy_digest"] != self.policy_digest
                    or proposal["implementation"] != code_snapshot(self.repo)
                    or not self._verify_remote(proposal, proposal["published_commit"])):
                raise KnowledgeError("Required knowledge delivery is pending")
        task.update(overall="complete", completed_at=now())
        return self.save("task", task_id, task)
