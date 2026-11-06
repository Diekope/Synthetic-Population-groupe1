"""Read immutable Git objects; indexes are disposable, documents are authoritative."""
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath


class KnowledgeError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def git(repo, *args, data=None):
    result = subprocess.run(["git", "-C", str(repo), *args], input=data,
                            capture_output=True)
    if result.returncode:
        raise KnowledgeError(result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


def safe_path(path):
    p = PurePosixPath(path)
    if (not path or p.is_absolute() or ".." in p.parts or "\\" in path
            or ":" in path or ".git" in p.parts or str(p) != path):
        raise KnowledgeError(f"Unsafe repository path: {path}")
    return path


class GitSource:
    def __init__(self, repo, ref="refs/heads/team-knowledge"):
        self.repo = Path(repo).resolve()
        self.ref = ref
        if ref != "refs/heads/team-knowledge":
            raise KnowledgeError("This source supports only the team-knowledge branch")

    def revision(self, ref=None):
        # Explicit end-of-options prevents revision arguments becoming Git options.
        return git(self.repo, "rev-parse", "--verify", "--end-of-options",
                   (ref or self.ref) + "^{commit}").decode().strip()

    def blob(self, revision, path):
        safe_path(path)
        mode = git(self.repo, "ls-tree", revision, "--", path).split(b" ", 1)[0]
        if mode != b"100644":
            raise KnowledgeError(f"Missing or non-regular document: {path}")
        return git(self.repo, "show", f"{revision}:{path}")

    def files(self, revision):
        return git(self.repo, "ls-tree", "-r", "--name-only", revision).decode().splitlines()

    def snapshot(self, revision=None):
        revision = self.revision(revision)
        manifest = json.loads(self.blob(revision, "knowledge/manifest.json"))
        if manifest.get("schema") != 1:
            raise KnowledgeError("Unsupported document schema")
        documents, ids = [], set()
        allowed = {"proposed", "implemented", "verified", "published", "superseded", "rejected"}
        for entry in manifest["documents"]:
            if entry["id"] in ids or entry["status"] not in allowed:
                raise KnowledgeError("Duplicate document ID or unsupported status")
            ids.add(entry["id"])
            raw = self.blob(revision, entry["path"])
            text = raw.decode("utf-8")
            if b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
                raise KnowledgeError("Documents must be UTF-8 without BOM, with LF newlines")
            if not text.startswith(f"<!-- id: {entry['id']} -->"):
                raise KnowledgeError(f"Missing stable document identifier: {entry['path']}")
            documents.append({**entry, "text": text, "sha256": digest(raw), "bytes": len(raw),
                              "section": "whole-document"})
        mandatory = [d for d in documents if d.get("mandatory")]
        if not mandatory or any(d["status"] in ("superseded", "rejected") for d in mandatory):
            raise KnowledgeError("Missing or invalid mandatory context")
        paths = set(self.files(revision))
        for path in paths:
            safe_path(path)
            if not (path.startswith("knowledge/") or path in ("AGENTS.md", "README.md", ".gitignore")):
                raise KnowledgeError(f"Implementation file on knowledge-only branch: {path}")
        for doc in documents:
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", doc["text"]):
                if "://" in target or target.startswith("#"):
                    continue
                import posixpath
                target = posixpath.normpath(posixpath.join(
                    str(PurePosixPath(doc["path"]).parent), target.split("#")[0]))
                if target not in paths:
                    raise KnowledgeError(f"Broken relative link in {doc['id']}: {target}")
        claims = json.loads(self.blob(revision, "knowledge/claims.json"))
        seen = set()
        for claim in claims:
            required = {"id", "class", "statement", "applicability", "repository", "branch",
                        "implementation_revision", "paths", "evidence", "limitations", "status",
                        "supersedes", "review", "validation_scope"}
            if required - claim.keys() or claim["id"] in seen:
                raise KnowledgeError("Incomplete or duplicate claim")
            seen.add(claim["id"])
            if claim["class"] not in ("observation", "interpretation", "decision", "proposal"):
                raise KnowledgeError("Unsupported claim class")
            if claim["status"] not in allowed or not re.fullmatch(r"[0-9a-f]{40}", claim["implementation_revision"]):
                raise KnowledgeError("Invalid claim status or immutable implementation revision")
            if claim["class"] != "proposal" and not claim["evidence"]:
                raise KnowledgeError("Non-proposal claims need evidence")
            for evidence in claim["evidence"]:
                if evidence not in paths:
                    raise KnowledgeError("Missing claim evidence")
        return {"revision": revision, "documents": documents, "claims": claims}

    def search(self, query="", *, revision=None, scope=None, status=None):
        snapshot = self.snapshot(revision)
        terms = query.casefold().split()
        return snapshot["revision"], [d for d in snapshot["documents"]
            if (not scope or scope in d.get("scope", []))
            and (not status or d["status"] == status)
            and d["status"] not in ("superseded", "rejected")
            and all(term in d["text"].casefold() for term in terms)]
