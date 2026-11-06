"""Finite protocol safety exploration with explicit failure events and negative controls.

This checks a deliberately small abstraction, not Python correspondence or general CTL/LTL.
There is no eventual human-approval assumption. Fair retries, connectivity recovery and
stable inputs would be necessary for liveness; this checker makes no liveness claim.
"""
from collections import deque
from dataclasses import dataclass, replace
import json


@dataclass(frozen=True)
class State:
    offered: bool = False
    receipt: bool = False
    fresh: bool = False
    approved: bool = False
    authorized: bool = False
    invalid: bool = False
    attempt: bool = False
    pushed: bool = False
    verified: bool = False
    code_complete: bool = False
    done: bool = False


def successors(state, broken=None):
    yield "offer", replace(state, offered=True)
    if state.offered and not state.invalid:
        yield "record_matching_request", replace(state, receipt=True)
    yield "wrong_session_ack_rejected", state
    yield "duplicate_or_lost_ack", state
    yield "observe_boundary", replace(state, fresh=True)
    yield "restart_or_compact", replace(state, receipt=False, fresh=False, done=False)
    yield "knowledge_or_code_changed", replace(state, receipt=False, fresh=False, done=False)
    yield "evidence_invalidated", replace(state, invalid=True, receipt=False, approved=False,
                                           authorized=False, verified=False, done=False)
    yield "policy_changed", replace(state, approved=False, authorized=False, done=False)
    if not state.invalid:
        yield "explicit_human_review", replace(state, approved=True)
        yield "explicit_publication_authorization", replace(state, authorized=True)
    yield "review_rejected_or_pending", state
    yield "code_result", replace(state, code_complete=True)
    if (state.receipt and state.fresh and not state.invalid) or broken == "gate":
        yield "gated_action", state
    if ((state.receipt and state.fresh and state.approved and state.authorized and not state.invalid)
            or broken == "publish"):
        yield "durable_attempt", replace(state, attempt=True)
        if state.attempt:
            yield "push", replace(state, pushed=True)
    yield "push_rejected_or_disconnected", state
    if state.pushed and state.attempt and not state.invalid:
        yield "query_remote_and_verify_content", replace(state, verified=True)
        yield "crash_after_push_then_status", state
    if ((state.receipt and state.fresh and state.code_complete and state.verified
         and state.approved and state.authorized and not state.invalid) or broken == "completion"):
        yield "complete_required_update", replace(state, done=True)


def violation(state, event):
    if event == "gated_action" and not (state.receipt and state.fresh and not state.invalid):
        return "gated action lacks current context receipt"
    if event in ("durable_attempt", "push") and not (
            state.receipt and state.fresh and state.approved and state.authorized and not state.invalid):
        return "publication lacks matching approval, authorization or context"
    if event == "complete_required_update" and not (
            state.receipt and state.fresh and state.code_complete and state.verified
            and state.approved and state.authorized and not state.invalid):
        return "required knowledge completion lacks verified delivery"
    return None


def explore(broken=None):
    initial = State()
    queue = deque([(initial, [])])
    seen = {initial}
    transitions = 0
    while queue:
        state, trace = queue.popleft()
        for event, target in successors(state, broken):
            transitions += 1
            error = violation(state, event)
            if error:
                return {"valid": False, "states": len(seen), "transitions": transitions,
                        "error": error, "counterexample": trace + [event]}
            if target not in seen:
                seen.add(target)
                queue.append((target, trace + [event]))
    return {"valid": True, "states": len(seen), "transitions": transitions}


def check():
    result = {"model": explore(), "negative_controls": {name: explore(name)
              for name in ("gate", "publish", "completion")}}
    if not result["model"]["valid"] or any(r["valid"] for r in result["negative_controls"].values()):
        raise RuntimeError("Protocol invariant or negative control failed")
    return result


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
