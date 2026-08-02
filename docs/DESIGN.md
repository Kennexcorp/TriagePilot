# TriagePilot — Adaptive Customer Support Triage Assistant

### A Local, Privacy-First Support Ticket Triage Agent (LangGraph + Ollama + Pydantic)

> **The Hook:** *"TriagePilot — routes inbound support tickets by emotional register, not just category, using a locally-run LangGraph agent so no ticket content ever leaves the device."*

---

## Phase 1: Inception & Problem Alignment

### The Problem

- Support tickets today are routed by category (billing, technical, shipping) or keyword tags, not by tone — a customer who's furious about a billing error gets the same templated, clinical response as someone asking a routine question.
- The cost of missing tone compounds: an unacknowledged frustrated customer escalates, disputes a charge, or churns; by the time a human agent reads it, the value of catching it early has already decayed.
- The obvious workarounds don't hold up: hiring more human triage staff doesn't scale with ticket volume, and keyword/rule-based routing ("contains 'refund'" → billing queue) misses tone entirely and breaks the moment a ticket mixes both frustration and a technical ask.
- Existing tools solve the wrong layer: Zendesk/Intercom-style routing sorts by category, not sentiment; commercial AI copilots that would catch tone typically pipe ticket content (often containing account/payment details) to a third-party hosted API, which is a non-starter for teams with data-residency or PII constraints, and adds per-ticket API cost at volume.

### Target Audience

- **Primary:** Small support teams (2–10 agents) at a SaaS company who need to triage inbound tickets by tone before a human ever reads them, and can't send ticket content to an external API.
- **Secondary:** Engineering teams evaluating LangGraph who want a reference implementation of structured-output classification + conditional routing they can extend.

### User Constraints

- **Hardware:** Must run on a single developer laptop or small on-prem server — no GPU assumed (Ollama models chosen must run acceptably on CPU).
- **Skill level:** Support leads adjusting behavior (e.g. refund approval thresholds, once tool-calling ships) should do so via config, not code.
- **Privacy/compliance:** Ticket content may include account IDs, order numbers, or payment details — none of it may leave the local device/network. This rules out hosted LLM APIs for the MVP.
- **Supported inputs:** Plain-text ticket body plus optional structured metadata (customer ID, order ID). No image/attachment parsing in v1.

### Scope Boundaries

| In-Scope (MVP)                                                               | Out-of-Scope (v1)                                                   |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Structured (Pydantic) classification into`de-escalation` vs `resolution` | Persistent multi-turn memory (checkpointer)                         |
| Conditional routing to two specialized agent nodes (Care, Resolution)        | Tool-calling (order lookup, refund calculator)                      |
| CLI loop for local interactive testing                                       | Human-in-the-loop approval gate for refund/credit actions           |
| Local-only inference via Ollama                                              | Multi-agent handoff mid-conversation (supervisor re-routing)        |
|                                                                              | Streaming responses                                                 |
|                                                                              | Integration with a real ticketing system (Zendesk/Intercom webhook) |

Cutting memory, tools, HITL, and handoff keeps the MVP shippable in ~2 weeks by proving the core mechanism — structured classification driving conditional routing — before adding anything whose correctness depends on that mechanism already working. Every deferred item reappears in the README Roadmap.

---

## Phase 2: Specifications & Guardrails

### Functional Requirements

| ID | Requirement                                                                                                                                                                                            |
| -- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| F1 | System must classify each incoming ticket into exactly one of`{de-escalation, resolution}` using a structured (Pydantic) schema — never free-text parsing.                                          |
| F2 | System must route a classified ticket to the corresponding agent node via a LangGraph conditional edge.                                                                                                |
| F3 | The de-escalation (Care) agent's response must acknowledge the customer's stated frustration before offering any next step.                                                                            |
| F4 | The resolution agent's response must directly address the technical request with concrete next steps (even without live tool access in the MVP, it must not fabricate an action as already completed). |
| F5 | System must expose a CLI loop where a user submits ticket text and receives the routed agent's response within the same session.                                                                       |

### Non-Functional Requirements

| Category                       | Guardrail                                                                                                                                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Performance**          | End-to-end classify + respond ≤ 4s on an 8-core CPU / Apple M-series laptop, no GPU, using`llama3.2:3b`.                                                                                                                |
| **Privacy/Security**     | No ticket content or model inference call may leave the local machine/network — verified by an automated check that no outbound network call occurs during a test run.                                                    |
| **Reliability**          | System degrades gracefully (returns a clear fallback message, never an unhandled crash) if Ollama is unreachable or returns malformed structured output; at least 1 automatic retry before falling back.                   |
| **Output quality**       | Classification accuracy ≥ 90% on a held-out, labeled eval set of N=50 tickets spanning clearly-emotional, clearly-technical, and ambiguous/mixed cases.                                                                   |
| **Safety / Action-risk** | MVP agents are advisory-only — no response may claim a refund, credit, or account change has been made. Once tool-calling ships, any action ≥ $50 must be flagged for human approval before execution, not auto-applied. |
| **Portability**          | Runs on macOS, Linux, and WSL with Python 3.13+; installable in ≤5 commands via`uv`.                                                                                                                                    |
| **Testability**          | ≥80% unit test coverage on graph nodes and classifier logic; the labeled eval set is a versioned fixture (`tests/fixtures/eval_set.json`), never regenerated ad hoc.                                                    |

---

## Phase 3: Architecture with Justification

### System Context & Data Flow

```
Ticket text (CLI input)
        │
        ▼
Classifier Node ── ChatOllama + Pydantic(TicketClassification) ──► ticket_type
        │
        ▼
Router (conditional edge on state["ticket_type"])
        │
   ┌────┴─────┐
   ▼          ▼
Care Agent   Resolution Agent    (each: ChatOllama + role-specific system prompt)
   │          │
   └────┬─────┘
        ▼
      END → response text → CLI output
```

### Tech Stack Evaluation Matrix

| Layer                              | Chosen                                                                                        | Alternatives Considered                                                                | Justification (the trade-off)                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core engine**              | LangGraph`StateGraph`                                                                       | Hand-rolled`if/else` router, plain LangChain LCEL chain                              | Typed state + conditional edges leave room for the planned checkpointer/HITL/cycles without a rewrite; accepted trade-off: an added framework dependency and learning curve versus a five-line if/else script that would work fine for the MVP alone.                                                                                                                            |
| **Model/inference**          | Ollama,`llama3.2:3b`                                                                        | Hosted API (OpenAI/Anthropic), larger local model (`qwen2.5:14b`)                    | Satisfies the no-PII-leaves-device constraint at zero marginal cost per ticket; accepted trade-off: lower raw classification/response quality than a frontier hosted model, mitigated by keeping the classification task narrow (binary) and validating against the eval set rather than assuming quality.                                                                       |
| **Structured output**        | LangChain`with_structured_output` + Pydantic                                                | Raw Ollama HTTP client with manual JSON parsing/regex                                  | Removes an entire class of brittle parsing bugs; accepted trade-off: tied to LangChain's abstraction versioning and its occasional breaking changes across releases.                                                                                                                                                                                                             |
| **Graph topology**           | Single classifier + two leaf agents, no supervisor                                            | Supervisor/orchestrator pattern with re-routing; single mega-agent with tool access    | Simplest topology that proves structured classification + conditional routing end-to-end before adding complexity; accepted trade-off: cannot re-route mid-conversation if tone shifts — explicitly deferred to the Roadmap's supervisor item, not silently ignored.                                                                                                            |
| **UI**                       | CLI loop                                                                                      | Web UI (Streamlit/FastAPI+React), Slack bot                                            | Fastest way to validate the core routing loop before investing in a UI layer; accepted trade-off: not demo-friendly for non-technical stakeholders — flagged as a v2 candidate.                                                                                                                                                                                                 |
| **Configuration**            | `pydantic-settings` `BaseSettings` reading `.env`                                       | Raw`python-dotenv` + `os.getenv`, hand-rolled config module, TOML/YAML config file | Validates config at the boundary where it enters, which matters because support leads edit these values and`REFUND_APPROVAL_THRESHOLD_USD` is numeric, so a typo fails at startup naming the field rather than at first use with a `TypeError`; accepted trade-off: one more dependency than `os.getenv`, though it reuses Pydantic which the classifier already requires. |
| **Packaging & distribution** | `uv` + local Python entrypoint (primary); Docker Compose with an Ollama sidecar (secondary) | pip+venv, Poetry, PyPI package, hosted SaaS                                            | Matches the audience (a developer evaluating or extending the code), not an end-user product; accepted trade-off: no one-click install for a non-technical support lead yet.                                                                                                                                                                                                     |

### Classifier Design (core mechanism)

`TicketClassification` Pydantic schema, defined in `graph/schemas.py`:

- `ticket_type: TicketType` — required, where `TicketType = Literal["de-escalation", "resolution"]`.
- `rationale: str` — a short (≤20 word) justification the model must produce alongside the label, kept for tracing/debugging classification decisions (surfaced via LangSmith once observability ships).

`TicketType` is declared once in `graph/schemas.py` and imported by `graph/state.py`, so the set of valid labels has a single definition. Adding a third category is then a one-line change rather than an edit that has to be kept in sync across two files.

Note the deliberate separation between `graph/schemas.py` and `graph/state.py`: the Pydantic model is a runtime validation boundary that raises when the model returns something unexpected, while `TriageState`, the `TypedDict` in `state.py`, is a static typing artifact describing the channel LangGraph threads between nodes and is never validated at runtime. They are different kinds of contract and are kept in different files.

**Edge-case strategy:**

- Ambiguous tickets that contain both a frustration signal and a technical ask default to `de-escalation` — acknowledging tone first is the safer failure mode than jumping straight to resolution.
- Tickets under 3 words or empty input skip classification and return a clarification prompt rather than forcing a label on insufficient signal.
- A structured-output call that raises or returns an unexpected type is retried once, then falls back to `de-escalation` with a flag for human review (see Risk Matrix).

---

## Phase 4: Timeline & Risk Mitigation

### Milestone Tracker

| Phase                                        | Deliverable                                                                                                                                  | Target    |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| **P0 — Foundations**                  | Dependencies pinned and installed;`ruff`/`pytest` config; `.env.example`; `LICENSE`                                                  | 0.5 day   |
| **P1 — Core Engine**                  | Classifier + router + Care/Resolution agent nodes + CLI loop; unit tests on classifier logic (mocked LLM)                                    | 1 week    |
| **P1.5 — Eval set & model benchmark** | 50-ticket labeled eval set fixture;`llama3.2:3b` vs `qwen2.5:7b` accuracy + latency comparison; default model locked in `.env.example` | 2 days    |
| **P2 — Integration & Hardening**      | Measured accuracy against the guardrail; retry/fallback error handling; Docker Compose                                                       | 1 week    |
| **P3 — Hardening & Ship**             | GitHub Actions CI (lint + test + coverage gate); README case study with measured Results table; demo GIF; troubleshooting table              | 3–4 days |

### Risk Matrix

| Risk                                                                                             | Likelihood | Impact | Mitigation                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------ | ---------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classifier misroutes ambiguous (mixed-tone) tickets                                              | M          | M      | Explicit default-to-`de-escalation` fallback rule; eval set deliberately includes ambiguous examples, not just clean-cut ones.                                                               |
| Ollama unreachable or model not pulled at runtime                                                | M          | H      | Startup health check against the Ollama endpoint, clear actionable error message, 1 automatic retry with backoff before failing.                                                               |
| Resolution agent implies an action was taken (refund, account change) before tool-calling exists | H (in MVP) | H      | System prompt explicitly restricts MVP agents to "acknowledge and explain next steps," never "confirm an action occurred"; called out in the Roadmap as blocked on the tool-calling milestone. |
| Local model quality insufficient for realistic ticket tone/phrasing                              | M          | M      | Benchmark`llama3.2:3b` vs `qwen2.5:7b` on the eval set before locking the default model (see Open Questions).                                                                              |
| Structured output call returns malformed/unexpected type                                         | L          | M      | `try/except` with 1 retry, then fallback classification + human-review flag rather than a crash.                                                                                             |

---

## Phase 5: Execution Standards (the Evidence Layer)

- **Repo structure:**
  ```
  triagepilot/
  ├── main.py                  # CLI entrypoint / interactive loop
  ├── config.py                # pydantic-settings BaseSettings
  ├── graph/
  │   ├── __init__.py
  │   ├── schemas.py           # Pydantic contracts, validated at runtime
  │   ├── state.py             # TypedDict graph channel, not validated
  │   ├── prompts.py           # system prompts, tuned independently of node logic
  │   ├── nodes.py             # classifier, care_agent, resolution_agent
  │   └── build.py             # StateGraph wiring
  ├── tests/
  │   ├── test_classifier.py
  │   ├── test_graph.py
  │   └── fixtures/eval_set.json
  ├── docs/
  │   ├── DESIGN.md
  │   ├── TODO.md               # phased implementation checklist
  │   └── demo.gif
  ├── .github/workflows/ci.yml
  ├── .env.example
  ├── LICENSE
  └── pyproject.toml
  ```
- **Module layering:** imports run strictly one way, so the package has no cycles: `prompts` imports nothing; `schemas` imports only Pydantic; `state` imports `schemas` (for `TicketType`); `nodes` imports `config`, `prompts`, `schemas`, `state`; `build` imports `state` and `nodes`; `main` imports `config` and `build`. Prompts and schemas sit at the bottom precisely because they are the parts iterated on most during evaluation, and nothing depends on them changing.
- **Git hygiene:** Conventional commits — e.g. `feat: add ticket classifier node with Pydantic schema`, `fix: retry on malformed structured output`, `test: add 50-ticket labeled eval set`.
- **Testing:** Unit tests mock `ChatOllama` responses so CI never calls a live model. The 50-ticket eval set runs as a separate, marked-slow integration test executed manually before release (not on every CI run), so accuracy numbers stay meaningful without slowing normal development.
- **CI/CD:** GitHub Actions running `ruff` lint + `pytest` with an 80% coverage gate; coverage badge surfaced in the README. No deploy step — this ships as a local developer tool, not a hosted service.
- **Config:** `.env.example` with `OLLAMA_MODEL`, `OLLAMA_HOST`, `REFUND_APPROVAL_THRESHOLD_USD` (reserved for the tool-calling milestone), and `LOG_LEVEL`.
- **Containerization:** Docker Compose with two services — `app` (Python) and `ollama` (official image, volume-mounted for model cache); `app` depends on `ollama` and reads `OLLAMA_HOST=http://ollama:11434`. Model pulls are handled by an entrypoint script, not baked into the image (keeps image size down and model choice swappable).

---

## Phase 6: Final Case Study Narrative (STAR skeleton)

- **Situation:** Support tickets are routed by category or keyword, not tone, so a furious customer and a routine question get the same clinical response — and neither hiring more triage staff nor hand-tuned keyword rules scales or generalizes.
- **Task:** Build a locally-run, privacy-preserving triage agent that classifies ticket tone via structured LLM output and routes to a tone-matched response agent, shippable in roughly two weeks, without sending any customer data to a third-party API.
- **Action:** Chose LangGraph's `StateGraph` + conditional edges over a hand-rolled router specifically to leave room for the planned checkpointer/HITL/cycles without a rewrite; constrained the classifier to a Pydantic schema to remove brittle string-matching on LLM output; chose Ollama for fully local inference to satisfy the no-PII-leaves-device constraint, validated against a held-out eval set rather than assumed; anticipated the hardest problem — mixed-tone tickets — with a documented default-to-de-escalation rule instead of leaving it to model whim.
- **Result (targets to measure and report):** ⚠️ classification accuracy on the 50-ticket eval set; ⚠️ end-to-end latency on target hardware; ⚠️ accuracy delta vs. a naive keyword-based router baseline. These become the README's Results table once P1–P2 are built and measured — not estimated.
- **Developer Empathy Block:** 5-command `uv` quickstart, `.env.example`, a troubleshooting table covering Ollama-not-running / model-not-pulled / malformed-output failure modes, and a Roadmap listing memory, tool-calling, HITL, multi-agent handoff, streaming, and observability — the Out-of-Scope column above, carried forward.

---

## Appendix: Dependency Shortlist

Versions are the latest stable releases as of 2026-07-29, resolved by `uv add`. All of them support Python 3.10+, so the project's `>=3.13` floor is a deliberate choice rather than one forced by a dependency.

```
langgraph>=1.2.10          # StateGraph, conditional edges
langchain-ollama>=1.1.0    # ChatOllama
pydantic>=2.13.4           # classifier schema
pydantic-settings>=2.14.2  # typed .env config
# dev group
pytest>=9.1.1              # test framework
pytest-cov>=7.1.0          # coverage gate
ruff>=0.16.0               # linter + formatter
# system: Ollama daemon (https://ollama.com) must be installed and running locally
```

**A note on LangChain 1.x.** LangChain and LangGraph reached 1.0 after this document was first drafted. The three APIs this architecture depends on (`StateGraph` over a `TypedDict`, `add_conditional_edges`, and `with_structured_output` over a Pydantic model) all survive the major version unchanged, so no rework is needed. Two 1.x details do affect the implementation:

- Message classes import from `langchain_core.messages`. The `langchain.messages` re-export needs the `langchain` meta-package, which this project does not depend on: the pinned set is `langchain-core` 1.5.2 via `langchain-ollama` 1.1.0, and `import langchain` raises `ModuleNotFoundError` here (verified 2026-08-01).
- `with_structured_output` accepts `include_raw=True`, which returns `{"raw", "parsed", "parsing_error"}` instead of raising on a malformed response. The Reliability guardrail's retry-then-fallback path is built on that branch rather than on `try/except`, which keeps the raw response available for the human-review flag.

**Open questions resolved:** Default model choice is no longer deferred to P2. Benchmarking `llama3.2:3b` vs `qwen2.5:7b` on the eval set (accuracy + latency on target hardware) is an early milestone (see `docs/TODO.md`, P1.5), because the `.env.example` default, the ≤4s latency guardrail, and the README Results table all depend on its outcome. The comparison is recorded in `docs/MODEL_EVAL.md`.
