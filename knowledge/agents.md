<!-- id: TK-AGENTS -->
# Agent integration

Date: 2026-11-06. Lifecycle: proposed. Review: pending. Publication: local draft.

Mandatory entry point: [entry](index.md), [policy](policy.md) and [protocol](protocol.md).
Retrieve relevant references after loading these. Instruction files are guidance, not lifecycle hooks.

| Mode | Guarantee | Limits |
|---|---|---|
| Guidance-only | Human/agent reads instructions and invokes explicit tools | No automatic load proof or universal interception |
| Instrumented project chat adapter | Records exact JSON model request at the existing chat host transport; binds receipt to offer/task/session/repository | Supplied request only, not understanding, retention, agreement or successful remote processing |

The implemented host slice targets the existing `tools/chat_agent.py` transport through the shared
client. Mock mode validates plumbing; real OpenAI-compatible/Ollama execution requires explicit model
configuration and normal network authorization. Tests substitute transport, not a live model.
This is one project LLM host integration, not a native lifecycle adapter for every coding-agent product.

Context compaction, restart, truncation or loss of instrumentation requires a new session epoch,
fresh mandatory context and a new recorded request. Do not accept a model's statement that it read a
document as evidence. Never place private notes or runtime personal data in shared knowledge by default.
Recorded model requests may contain user content; keep the state store private and apply retention rules
before deploying beyond this local prototype. There is no automatic secret scrubber.
