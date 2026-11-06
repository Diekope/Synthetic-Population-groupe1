"""Shared CLI: python -m knowledgekit --help."""
import argparse
import getpass
import json
import sys
from pathlib import Path

from .hosts import ProjectChatHost
from .service import Service, make_tree
from .source import GitSource, KnowledgeError, git


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("directory", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--revision")
    search = commands.add_parser("search")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--scope")
    search.add_argument("--status")
    search.add_argument("--revision")
    start = commands.add_parser("start")
    start.add_argument("task")
    start.add_argument("epoch")
    start.add_argument("--scope", default="project")
    start.add_argument("--query", default="")
    start.add_argument("--previous")
    host = commands.add_parser("host-request")
    host.add_argument("task")
    host.add_argument("question")
    host.add_argument("--agent", default="1")
    host.add_argument("--provider", choices=("mock", "openai", "ollama"), default="mock")
    host.add_argument("--model", default="mock")
    for name in ("freshness", "gate"):
        sub = commands.add_parser(name)
        sub.add_argument("task")
        sub.add_argument("--boundary", default="consequential")
        if name == "freshness":
            sub.add_argument("--offline", action="store_true")
        else:
            sub.add_argument("--operation", default="read")
    result = commands.add_parser("result")
    result.add_argument("task")
    result.add_argument("status", choices=("complete", "rolled_back"))
    result.add_argument("--no-update-reason")
    proposal = commands.add_parser("propose")
    proposal.add_argument("task")
    proposal.add_argument("directory", type=Path)
    proposal.add_argument("evidence", type=Path)
    proposal.add_argument("--class", dest="semantic_class", default="interpretation")
    review = commands.add_parser("review")
    review.add_argument("proposal")
    review.add_argument("fingerprint")
    review.add_argument("decision", choices=("approved", "rejected", "needs_evidence", "narrowing"))
    review.add_argument("reference")
    authorize = commands.add_parser("authorize")
    authorize.add_argument("proposal")
    authorize.add_argument("fingerprint")
    authorize.add_argument("reference")
    publish = commands.add_parser("publish")
    publish.add_argument("proposal")
    complete = commands.add_parser("complete")
    complete.add_argument("task")
    invalidate = commands.add_parser("invalidate")
    invalidate.add_argument("task")
    invalidate.add_argument("reason")
    status = commands.add_parser("status")
    status.add_argument("kind", choices=("task", "proposal", "attempt", "offer", "receipt", "observation"))
    status.add_argument("id")
    commands.add_parser("reconcile")
    args = parser.parse_args()
    repo = args.repo.resolve()
    local = repo / ".knowledge-local"
    config = local / "policy.json"
    if args.command == "init":
        local.mkdir(exist_ok=True)
        policy = {"version": "1", "principals": [getpass.getuser()],
                  "reviewers": [getpass.getuser()], "publishers": [getpass.getuser()],
                  "offline_operations": ["read", "test", "propose"],
                  "automatic_observation_publication": False}
        if config.exists():
            raise KnowledgeError("Policy already exists; inspect it instead of overwriting")
        config.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        return {"policy": str(config), "mode": "local OS-principal; review required; no automatic publication"}
    if args.command == "bootstrap":
        # Create an orphan tree through Git objects; the application checkout is untouched.
        if git(repo, "for-each-ref", "--format=%(refname)", "refs/heads/team-knowledge").strip():
            raise KnowledgeError("team-knowledge already exists")
        tree = make_tree(repo, args.directory)
        commit = git(repo, "commit-tree", tree,
                     data=b"docs: establish draft team knowledge contracts\n").decode().strip()
        GitSource(repo).snapshot(commit)
        git(repo, "update-ref", "refs/heads/team-knowledge", commit, "0" * 40)
        return {"revision": commit, "publication": "local_only", "semantic_review": "pending"}
    if args.command == "validate":
        snapshot = GitSource(repo).snapshot(args.revision)
        return {"revision": snapshot["revision"], "documents": len(snapshot["documents"]),
                "claims": len(snapshot["claims"]), "status": "mechanically_validated"}
    if not config.exists():
        raise KnowledgeError("Run init and inspect the local access policy first")
    service = Service(repo, local / "state.sqlite", json.loads(config.read_text(encoding="utf-8")))
    command = args.command
    if command == "search":
        revision, documents = service.source.search(args.query, revision=args.revision,
                                                   scope=args.scope, status=args.status)
        return {"revision": revision, "documents": documents}
    if command == "start":
        return service.start(args.task, args.epoch, scope=args.scope, query=args.query, previous=args.previous)
    if command == "host-request":
        return ProjectChatHost(service).run(args.task, args.agent, args.question,
                                            provider=args.provider, model=args.model)
    if command == "freshness":
        return service.freshness(args.task, args.boundary, offline=args.offline)
    if command == "gate":
        return service.gate(args.task, args.boundary, operation=args.operation)
    if command == "result":
        return service.result(args.task, args.status, no_update_reason=args.no_update_reason)
    if command == "propose":
        return service.propose(args.task, args.directory, json.loads(args.evidence.read_text(encoding="utf-8")),
                               semantic_class=args.semantic_class)
    if command == "review":
        return service.review(args.proposal, args.fingerprint, args.decision, args.reference)
    if command == "authorize":
        return service.authorize(args.proposal, args.fingerprint, args.reference)
    if command == "publish":
        return service.publish(args.proposal)
    if command == "invalidate":
        return service.invalidate(args.task, args.reason)
    if command == "complete":
        return service.complete(args.task)
    if command == "reconcile":
        return service.reconcile()
    return service.load(args.kind, args.id)


if __name__ == "__main__":
    try:
        print(json.dumps(main(), ensure_ascii=False, indent=2))
    except (KnowledgeError, ValueError, KeyError, OSError) as error:
        print(f"knowledgekit: {error}", file=sys.stderr)
        sys.exit(1)
