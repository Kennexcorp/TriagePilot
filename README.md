# TriagePilot 🎫→💬

**TriagePilot turns raw support tickets into tone-matched responses — routing by *how upset the customer is*, not just what category the ticket falls under.** Runs 100% locally on Ollama (no ticket content ever leaves the device), costs nothing per ticket, and is built on LangGraph so the routing logic is typed, testable, and ready to grow into tool-calling and human approval gates.

![CI](https://img.shields.io/badge/CI-pending-lightgrey)
![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

![Demo](docs/demo.gif)
<!-- ⚠️ Record once P1 CLI loop exists: a 15–30s GIF showing a frustrated ticket
     routed to the Care agent, then a technical ticket routed to Resolution. -->

## What it does

Support tickets get routed by category (billing, technical, shipping) or keyword tags today — never by tone. A furious customer disputing a charge gets the same clinical, templated response as someone asking a routine question, and by the time a human reads it, the moment to catch the frustration early has already passed. TriagePilot takes raw ticket text and produces:

1. **A tone classification** (`de-escalation` vs `resolution`) via a Pydantic-structured LLM call — no regex or keyword matching
2. **A tone-matched response** — the Care agent acknowledges frustration before anything else; the Resolution agent answers the technical ask directly
3. **A rationale string** attached to every classification, for debugging *why* a ticket was routed the way it was
4. **A local-only guarantee** — no ticket content, which may include account or payment details, is ever sent to a third-party API

Right now this runs as a CLI loop for local testing; the response text is what a human agent (or, later, a real ticketing system integration) would act on.

**Why local-only?** Ticket content routinely includes account IDs, order numbers, and payment details. Running inference entirely through Ollama on-device is an architectural guarantee that data doesn't leave the network — not a policy promise sitting on top of a hosted API call.

---

## Quickstart

### Path A — uv (recommended)

Prereqs: [uv](https://github.com/astral-sh/uv), [Ollama](https://ollama.com/) installed and running.

```bash
git clone https://github.com/your-username/triagepilot.git
cd triagepilot
uv venv && source .venv/bin/activate
uv sync
ollama pull llama3.2
uv run main.py
```

First run pulls `llama3.2:3b` if you haven't already (~2GB download) — after that, everything is local and offline.

### Path B — Docker (one command, fully reproducible)

Prereqs: Docker + Docker Compose.

```bash
docker compose up
```

This stands up two services: `app` (the TriagePilot CLI/graph) and `ollama` (model runtime, with a volume for the model cache so it isn't re-downloaded on every rebuild). The model itself is deliberately not baked into the image — it's pulled at container start so image size stays small and the model is swappable via `.env`.

> On Linux, you may need `OLLAMA_HOST=http://ollama:11434` set explicitly rather than relying on Docker Desktop's automatic service-name resolution.

### Configuration

Copy `.env.example` → `.env`:

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Swap to `qwen2.5` for higher-quality (slower) classification — see Results for the trade-off once benchmarked. |
| `OLLAMA_HOST` | `http://localhost:11434` | Change to `http://ollama:11434` when running via Docker Compose. |
| `REFUND_APPROVAL_THRESHOLD_USD` | `50` | Reserved — takes effect once tool-calling + human-in-the-loop approval ship (see Roadmap). No refund actions are taken in the current MVP. |
| `LOG_LEVEL` | `info` | Set to `debug` to see full classifier rationale output. |

---

## Results

<!-- ⚠️ Every row below needs a real measured number from the P2 milestone.
     Do not fill these in with estimates — run the eval set and record it. -->

| Metric | Result |
|---|---|
| Classification accuracy (50-ticket labeled eval set) | **⚠️ pending — measure in P2** |
| Baseline comparison (keyword/rule-based router, same eval set) | **⚠️ pending (N× more/less accurate)** |
| End-to-end latency, classify + respond (M-series / 8-core CPU, `llama3.2:3b`) | **⚠️ pending — target ≤ 4s** |
| Care agent: acknowledges sentiment before resolution content (manual review, N=20) | **⚠️ pending** |
| Test coverage (graph nodes + classifier logic) | **⚠️ pending — target ≥ 80%** |
| Model comparison: `llama3.2:3b` vs `qwen2.5:7b` (accuracy + latency) | see [docs/MODEL_EVAL.md](docs/MODEL_EVAL.md) *(to be created in P2)* |

---

## How it works

```
Ticket text → Classifier (ChatOllama + Pydantic schema) → Router → Care Agent / Resolution Agent (ChatOllama) → response
```

Full architecture diagram, requirements, and risk matrix: [docs/DESIGN.md](docs/DESIGN.md)

### Why these choices

| Choice | Over | Because |
|---|---|---|
| **LangGraph `StateGraph`** | Hand-rolled if/else router, plain LCEL chain | Typed state + conditional edges leave room for the planned checkpointer/HITL/cycles without a rewrite — at the cost of a framework dependency a simple script wouldn't need. |
| **Ollama, local-only inference** | Hosted API (OpenAI/Anthropic) | Ticket content can include account/payment details that can't leave the device; accepted trade-off is lower raw model quality than a frontier hosted model, offset by keeping classification narrow (binary) and validated against an eval set. |
| **Pydantic structured output** | Manual JSON parsing / regex on raw LLM text | Removes a whole class of brittle parsing failures, at the cost of coupling to LangChain's `with_structured_output` abstraction. |
| **Single classifier + two leaf agents (no supervisor)** | Supervisor/orchestrator pattern with mid-conversation re-routing | Proves the core routing mechanism first; can't yet handle a tone shift mid-ticket — explicitly deferred to the Roadmap, not silently dropped. |
| **CLI-first packaging** | Web UI / Slack bot | Fastest path to validating the routing loop before investing in a UI aimed at non-technical support leads. |

Full trade-off analysis, requirements, and risk matrix: [docs/DESIGN.md](docs/DESIGN.md)

---

## Engineering challenges

<!-- ⚠️ These are anticipated challenges from the design doc's risk matrix,
     written as predictions. Once P1–P3 are actually built, rewrite each
     one in past tense with the real mechanism used and a measured outcome. -->

**Keeping the resolution agent from implying it took an action it can't yet take.** Without tool-calling, the Resolution agent has no way to actually check an order or issue a refund — but a fluent LLM will happily write "I've processed your refund" if not constrained. This is a real safety risk, not just a UX one, since a customer could act on a false confirmation. Mitigation: the system prompt explicitly restricts MVP responses to "acknowledge and explain next steps," never "confirm an action occurred," enforced and checked in the classifier/response eval set. *(⚠️ add measured outcome — % of eval responses that incorrectly imply a completed action, target 0%.)*

**Routing ambiguous, mixed-tone tickets correctly.** A ticket like "this is the third time I've asked and I still don't have an answer" is both frustrated and technical — a naive binary classifier could go either way inconsistently. Mitigation: a documented default-to-`de-escalation` rule for ambiguous cases, plus deliberately including mixed-tone examples in the labeled eval set rather than only clean-cut ones. *(⚠️ add measured accuracy specifically on the ambiguous subset.)*

**Failing gracefully when the local model isn't available.** Since the whole point of the project is local-only inference, an unreachable Ollama daemon or an unpulled model is the single most likely first-run failure a new user hits — not an edge case. Mitigation: a startup health check, one automatic retry, and a specific actionable error message rather than a raw stack trace. *(⚠️ add measured outcome once implemented — e.g. % of failure-mode tests passing.)*

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ConnectionError` / `httpx.ConnectError` on startup | Ollama isn't running. Start it with `ollama serve`, or confirm `OLLAMA_HOST` in `.env` points to the right address. |
| `model 'llama3.2' not found` | Run `ollama pull llama3.2` before starting TriagePilot. |
| Classifier raises a validation error on the Pydantic schema | The model returned an unexpected structure — this triggers one automatic retry; if it persists, try `OLLAMA_MODEL=qwen2.5`, which tends to follow structured-output instructions more reliably. |
| Response feels slow (>10s) on CPU-only hardware | Expected on larger models; switch `OLLAMA_MODEL` to `llama3.2:3b` (the smallest variant) rather than a 7B+ model if latency matters more than nuance. |

---

## Roadmap

- [ ] **Persistent memory** — `MemorySaver`/`SqliteSaver` checkpointer keyed on `thread_id`, so a reopened ticket has context from prior turns.
- [ ] **Tool-calling** — order-status lookup, account lookup, and refund calculator attached to the Resolution agent via a `ToolNode`.
- [ ] **Human-in-the-loop approval** — an `interrupt()` gate before any refund/credit action above `REFUND_APPROVAL_THRESHOLD_USD`.
- [ ] **Cyclical/ReAct reasoning** — let the Resolution agent loop back after a tool call (e.g. order lookup → shipping estimate → final answer) instead of a single straight-line pass.
- [ ] **Multi-agent handoff** — a supervisor node that can re-route mid-conversation when a customer's tone shifts partway through a ticket.
- [ ] **Streaming** — `graph.stream()`/`astream_events` output instead of a single blocking response.
- [ ] **Observability** — LangSmith tracing to debug classification decisions using the `rationale` field already captured in the schema.
- [ ] **Real ticketing system integration** — a Zendesk or Intercom webhook instead of the CLI loop.

## Project structure

```
triagepilot/
├── main.py                  # CLI entrypoint / interactive loop
├── graph/
│   ├── state.py             # TypedDict state schema
│   ├── nodes.py              # classifier, care_agent, resolution_agent
│   └── build.py              # StateGraph wiring
├── tests/
│   ├── test_classifier.py
│   ├── test_graph.py
│   └── fixtures/eval_set.json
├── docs/
│   ├── DESIGN.md             # full design doc — requirements, architecture, risk matrix
│   └── demo.gif
├── .github/workflows/ci.yml
├── .env.example
└── pyproject.toml
```

## License

MIT — see [LICENSE](LICENSE).
