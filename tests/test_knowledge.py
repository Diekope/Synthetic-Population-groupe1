"""Integration tests use isolated Git repositories, synthetic documents and fake transport."""
import getpass
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledgekit.hosts import ProjectChatHost
from knowledgekit.service import Service, make_tree
from knowledgekit.source import GitSource, KnowledgeError, canonical, digest, git


ROOT = Path(__file__).resolve().parents[1]


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


class KnowledgeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "code"
        self.repo.mkdir()
        git(self.repo, "init", "-b", "main")
        git(self.repo, "config", "user.name", "Test reviewer")
        git(self.repo, "config", "user.email", "review@example.invalid")
        git(self.repo, "config", "core.autocrlf", "false")
        write(self.repo / ".gitignore", ".knowledge-local/\n.local/\n__pycache__/\n")
        write(self.repo / "src/app.txt", "synthetic application\n")
        shutil.copytree(ROOT / "tools", self.repo / "tools")
        agent = {"id": 1, "identity_key": "abc", "prenom": "Test", "nom": "Agent",
                 "persona": "Synthetic fixture persona.", "attributes": {"A0": "v0"}}
        write(self.repo / "agents/agents.jsonl", json.dumps(agent) + "\n")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-m", "test: create isolated application fixture")
        self.code_revision = git(self.repo, "rev-parse", "HEAD").decode().strip()
        self.candidate = self.root / "knowledge"
        entries = []
        for identity, name, mandatory in [("ENTRY", "entry", True), ("POLICY", "policy", True),
                                          ("PROTOCOL", "protocol", True), ("OPTIONAL", "optional", False)]:
            text = f"<!-- id: {identity} -->\n# {name}\n\nSynthetic {name} reference.\n"
            write(self.candidate / f"knowledge/{name}.md", text)
            entries.append({"id": identity, "path": f"knowledge/{name}.md", "mandatory": mandatory,
                            "status": "implemented", "scope": ["project"]})
        write(self.candidate / "knowledge/manifest.json", json.dumps({"schema": 1, "documents": entries}))
        write(self.candidate / "knowledge/claims.json", "[]\n")
        tree = make_tree(self.repo, self.candidate)
        self.base = git(self.repo, "commit-tree", tree, data=b"docs: create synthetic knowledge\n").decode().strip()
        git(self.repo, "update-ref", "refs/heads/team-knowledge", self.base)
        self.remote = self.root / "remote.git"
        self.remote.mkdir()
        git(self.remote, "init", "--bare")
        git(self.repo, "remote", "add", "origin", str(self.remote))
        git(self.repo, "push", "origin", "team-knowledge")
        user = getpass.getuser()
        self.policy = {"version": "1", "principals": [user], "reviewers": [user],
                       "publishers": [user], "offline_operations": ["read", "test", "propose"],
                       "automatic_observation_publication": False}
        self.database = self.repo / ".knowledge-local/state.sqlite"
        self.service = Service(self.repo, self.database, self.policy)

    def tearDown(self):
        self.temporary.cleanup()

    def load_context(self, epoch="one"):
        offer = self.service.start("task", epoch)
        payload = {"messages": [{"role": "system", "content": "\n".join(d["text"] for d in offer["context"])}]}
        receipt = self.service.record_request("task", epoch, offer["id"], offer["manifest_digest"], payload)
        self.service.freshness("task", "consequential")
        return offer, payload, receipt

    def proposal(self):
        self.load_context()
        write(self.candidate / "knowledge/optional.md", "<!-- id: OPTIONAL -->\n# optional\n\nReviewed candidate change.\n")
        return self.service.propose("task", self.candidate, {"id": "EV-TEST", "result": "synthetic pass"})

    def approve(self, proposal):
        self.service.review(proposal["id"], proposal["fingerprint"], "approved", "explicit fixture review")
        self.service.authorize(proposal["id"], proposal["fingerprint"], "explicit fixture publication permission")

    def move_source(self):
        tree = make_tree(self.repo, self.candidate)
        revision = git(self.repo, "commit-tree", tree, "-p", self.base,
                       data=b"docs: move knowledge tip\n").decode().strip()
        git(self.repo, "update-ref", "refs/heads/team-knowledge", revision)
        return revision

    def test_branch_contains_only_knowledge(self):
        paths = self.service.source.files(self.base)
        self.assertTrue(all(p.startswith("knowledge/") for p in paths))
        self.assertEqual(git(self.repo, "rev-parse", "HEAD").decode().strip(), self.code_revision)

    def test_snapshot_remains_pinned_after_ref_moves(self):
        original = self.service.source.snapshot()
        write(self.candidate / "knowledge/optional.md", "<!-- id: OPTIONAL -->\n# renamed text\n")
        self.move_source()
        pinned = self.service.source.snapshot(original["revision"])
        self.assertEqual(original, pinned)
        self.assertNotEqual(original, self.service.source.snapshot())

    def test_search_filters_and_mandatory_offer(self):
        _, docs = self.service.source.search("optional", scope="project", status="implemented")
        self.assertEqual([d["id"] for d in docs], ["OPTIONAL"])
        offer = self.service.start("task", "one", scope="absent", query="no matching text")
        self.assertEqual(len(offer["manifest"]), 3)

    def test_missing_and_corrupt_mandatory_context_fail(self):
        (self.candidate / "knowledge/policy.md").unlink()
        self.move_source()
        with self.assertRaises(KnowledgeError):
            self.service.start("task", "one")
        write(self.candidate / "knowledge/policy.md", "Wrong stable identifier\n")
        self.move_source()
        with self.assertRaises(KnowledgeError):
            self.service.source.snapshot()

    def test_wrong_session_manifest_offer_and_missing_request(self):
        offer, payload, _ = self.load_context()
        for epoch, offer_id, manifest, request in [
            ("wrong", offer["id"], offer["manifest_digest"], payload),
            ("one", offer["id"], "wrong", payload),
            ("one", offer["id"], offer["manifest_digest"], {"messages": []}),
        ]:
            with self.assertRaises(KnowledgeError):
                self.service.record_request("task", epoch, offer_id, manifest, request)
        self.service.start("task", "two")
        with self.assertRaises(KnowledgeError):
            self.service.record_request("task", "one", offer["id"], offer["manifest_digest"], payload)

    def test_duplicate_announce_ack_and_status_after_restart(self):
        offer, payload, receipt = self.load_context()
        self.assertEqual(self.service.start("task", "one")["id"], offer["id"])
        again = self.service.record_request("task", "one", offer["id"], offer["manifest_digest"], payload)
        self.assertEqual(receipt["id"], again["id"])
        self.assertEqual(receipt, again)
        reopened = Service(self.repo, self.database, self.policy)
        self.assertEqual(reopened.load("receipt", receipt["id"])["id"], receipt["id"])

    def test_restart_and_compaction_invalidate_receipt(self):
        self.load_context()
        self.service.gate("task")
        self.service.start("task", "after-compaction")
        with self.assertRaises(KnowledgeError):
            self.service.gate("task")

    def test_rename_delete_supersede_rebuild_search(self):
        manifest_path = self.candidate / "knowledge/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["documents"][-1]
        entry.update(path="knowledge/new.md", status="superseded")
        (self.candidate / "knowledge/optional.md").rename(self.candidate / "knowledge/new.md")
        write(manifest_path, json.dumps(manifest))
        self.move_source()
        _, docs = self.service.source.search("optional")
        self.assertEqual(docs, [])
        self.assertNotIn("knowledge/optional.md", self.service.source.files(self.service.source.revision()))

    def test_access_principal_and_repository_isolation(self):
        with self.assertRaises(KnowledgeError):
            Service(self.repo, self.database, self.policy, actor="unauthorized")
        git(self.repo, "remote", "set-url", "origin", str(self.root / "different.git"))
        with self.assertRaises(KnowledgeError):
            Service(self.repo, self.database, self.policy)
        with self.assertRaises(KnowledgeError):
            self.service.remote_tip()

    def test_missing_review_authorization_and_changed_policy(self):
        proposal = self.proposal()
        with self.assertRaises(KnowledgeError):
            self.service.publish(proposal["id"])
        self.service.review(proposal["id"], proposal["fingerprint"], "approved", "fixture review")
        with self.assertRaises(KnowledgeError):
            self.service.publish(proposal["id"])
        with self.assertRaises(KnowledgeError):
            self.service.authorize(proposal["id"], "wrong", "fixture permission")
        self.approve(proposal)
        changed = Service(self.repo, self.database, {**self.policy, "version": "2"})
        with self.assertRaises(KnowledgeError):
            changed.publish(proposal["id"])

    def test_rejected_and_narrowed_review_not_publication(self):
        proposal = self.proposal()
        for decision in ("rejected", "narrowing", "needs_evidence"):
            self.service.review(proposal["id"], proposal["fingerprint"], decision, "fixture review")
            self.service.authorize(proposal["id"], proposal["fingerprint"], "fixture permission")
            with self.assertRaises(KnowledgeError):
                self.service.publish(proposal["id"])

    def test_evidence_invalidation_blocks_receipts_and_publication(self):
        proposal = self.proposal()
        self.approve(proposal)
        self.service.invalidate("task", "Evidence no longer applies")
        with self.assertRaises(KnowledgeError):
            self.service.gate("task")
        with self.assertRaises(KnowledgeError):
            self.service.publish(proposal["id"])
        self.assertEqual(self.service.load("proposal", proposal["id"])["review"], "needs_evidence")

    def test_stale_knowledge_and_dirty_code_fail_gate(self):
        self.load_context()
        write(self.repo / "src/app.txt", "changed implementation\n")
        with self.assertRaises(KnowledgeError):
            self.service.gate("task")
        self.service.freshness("task", "consequential")
        self.service.gate("task")
        self.move_source()
        with self.assertRaises(KnowledgeError):
            self.service.freshness("task", "phase")

    def test_offline_policy_blocks_publication_and_completion(self):
        self.load_context()
        self.service.freshness("task", "consequential", offline=True)
        self.service.gate("task", operation="read")
        with self.assertRaises(KnowledgeError):
            self.service.gate("task", operation="publish")
        self.service.result("task", "complete", no_update_reason="Only a disposable local test was executed.")
        self.service.freshness("task", "completion", offline=True)
        with self.assertRaises(KnowledgeError):
            self.service.complete("task")

    def test_pending_knowledge_prevents_completion(self):
        self.load_context()
        self.service.result("task", "complete")
        self.service.freshness("task", "completion")
        with self.assertRaises(KnowledgeError):
            self.service.complete("task")

    def test_no_update_reason_and_rollback_completion(self):
        self.load_context()
        with self.assertRaises(KnowledgeError):
            self.service.result("task", "rolled_back", no_update_reason="none")
        self.service.result("task", "rolled_back", no_update_reason="Disposable fixture changes were fully reverted; no shared fact changed.")
        self.service.freshness("task", "completion")
        self.assertEqual(self.service.complete("task")["overall"], "complete")

    def test_full_publication_and_remote_verification(self):
        proposal = self.proposal()
        self.approve(proposal)
        published = self.service.publish(proposal["id"])
        self.assertEqual(published["delivery"], "remote_verified")
        self.assertEqual(self.service.remote_tip(), published["published_commit"])
        self.assertEqual(git(self.repo, "rev-parse", "HEAD").decode().strip(), self.code_revision)
        self.service.start("task", "after-publication")
        offer = self.service.load("offer", self.service.load("task", "task")["offer"])
        request = {"messages": [{"role": "system", "content": "\n".join(d["text"] for d in offer["context"])}]}
        self.service.record_request("task", "after-publication", offer["id"], offer["manifest_digest"], request)
        self.service.result("task", "complete")
        self.service.freshness("task", "completion")
        self.assertEqual(self.service.complete("task")["overall"], "complete")

    def test_concurrent_publication_rejected_without_force(self):
        proposal = self.proposal()
        self.approve(proposal)
        concurrent = git(self.repo, "commit-tree", proposal["tree"], "-p", self.base,
                         data=b"docs: concurrent publication\n").decode().strip()
        git(self.repo, "push", "origin", f"{concurrent}:refs/heads/team-knowledge")
        with self.assertRaises(KnowledgeError):
            self.service.publish(proposal["id"])
        self.assertEqual(self.service.remote_tip(), concurrent)

    def test_push_rejection_keeps_durable_attempt(self):
        proposal = self.proposal()
        self.approve(proposal)
        original = git

        def reject(repo, *args, **kwargs):
            if args[0] == "push":
                raise KnowledgeError("synthetic non-fast-forward rejection")
            return original(repo, *args, **kwargs)

        with patch("knowledgekit.service.git", side_effect=reject):
            with self.assertRaises(KnowledgeError):
                self.service.publish(proposal["id"])
        self.assertEqual(self.service.load("attempt", proposal["id"])["state"], "prepared")
        self.assertEqual(self.service.load("proposal", proposal["id"])["delivery"], "pending")

    def test_crash_after_push_recovers_without_duplicate_commit(self):
        proposal = self.proposal()
        self.approve(proposal)
        with patch.object(self.service, "_verify_remote", side_effect=RuntimeError("crash after push")):
            with self.assertRaises(RuntimeError):
                self.service.publish(proposal["id"])
        tip = self.service.remote_tip()
        recovered = Service(self.repo, self.database, self.policy)
        result = recovered.publish(proposal["id"])
        self.assertEqual(result["published_commit"], tip)
        self.assertEqual(recovered.remote_tip(), tip)
        self.assertEqual(recovered.source.revision(), tip)

    def test_changed_announce_and_stale_ack_fail_closed(self):
        offer, payload, _ = self.load_context()
        with self.assertRaises(KnowledgeError):
            self.service.start("task", "one", query="different request")
        self.move_source()
        with self.assertRaises(KnowledgeError):
            self.service.record_request("task", "one", offer["id"], offer["manifest_digest"], payload)

    def test_parallel_operations_keep_durable_task_state(self):
        from concurrent.futures import ThreadPoolExecutor
        self.load_context()
        second = Service(self.repo, self.database, self.policy)
        with ThreadPoolExecutor(max_workers=2) as pool:
            operations = [pool.submit(self.service.result, "task", "complete"),
                          pool.submit(second.freshness, "task", "phase")]
            for operation in operations:
                operation.result()
        task = self.service.load("task", "task")
        self.assertEqual(task["code"], "complete")
        self.assertIn("phase", task["freshness"])

    def test_project_host_mock_records_exact_supplied_context(self):
        offer = self.service.start("task", "one")
        answer = ProjectChatHost(self.service).run("task", 1, "Synthetic question")
        receipt = answer["receipt"]
        request = self.service.load("request", receipt["id"])["request"]
        self.assertEqual(receipt["manifest_digest"], offer["manifest_digest"])
        self.assertTrue(all(d["text"] in request["messages"][0]["content"] for d in offer["context"]))

    def test_project_host_real_transport_hook_uses_recorded_payload(self):
        self.service.start("task", "one")
        host = ProjectChatHost(self.service)
        payloads = []

        def fake_transport(request, timeout):
            payloads.append(json.loads(request.data.decode()))
            return io.BytesIO(b'{"message":{"content":"synthetic transport answer"}}')

        with patch.object(host.host, "_open", side_effect=fake_transport):
            result = host.run("task", 1, "Synthetic question", provider="ollama", model="fixture")
        recorded = self.service.load("request", result["receipt"]["id"])["request"]
        self.assertEqual(recorded, payloads[0])
        self.assertEqual(result["answer"], "synthetic transport answer")

    def test_invalid_manifest_schema_and_application_file_rejected(self):
        write(self.candidate / "src/not-knowledge.py", "print('not allowed')\n")
        self.move_source()
        with self.assertRaises(KnowledgeError):
            self.service.source.snapshot()


if __name__ == "__main__":
    unittest.main()
