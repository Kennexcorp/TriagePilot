# TriagePilot — Implementation Checklist

Phases mirror the milestone tracker in [DESIGN.md](DESIGN.md). Rationale, requirement IDs (F1–F5), and the risk matrix live there.

---

## P0 — Foundations

- [X] Pin and install dependencies via `uv add` / `uv add --dev`: `langgraph`, `langchain-ollama`, `pydantic`, `pydantic-settings`; dev: `pytest`, `pytest-cov`, `ruff`
- [X] Add `[tool.ruff]`, `[tool.pytest.ini_options]`, and `[tool.coverage.*]` (80% gate) to `pyproject.toml`
- [X] Replace the placeholder `description` in `pyproject.toml`
- [X] Add `.env.example` with `OLLAMA_MODEL`, `OLLAMA_HOST`, `REFUND_APPROVAL_THRESHOLD_USD`, `LOG_LEVEL`
- [X] Add `LICENSE` (MIT) so the README link resolves
- [X] Add `.env`, `.pytest_cache/`, `.coverage`, `.ruff_cache/` to `.gitignore`
- [X] Correct the Python floor to 3.13+ in `README.md` and `DESIGN.md`
- [X] Comment out the `docs/demo.gif` embed until the GIF exists
- [X] Update the dependency shortlist in `DESIGN.md` to the pinned LangChain 1.x set

---

## P1 — Core Engine

**Done when:** `uv run main.py` classifies a typed ticket and prints a tone-matched response, and `uv run pytest` passes with `ChatOllama` mocked throughout.

- [X] `pyproject.toml`: add `pythonpath = ["."]` to `[tool.pytest.ini_options]`, without which `uv run pytest` cannot import `graph` or `config` (confirmed `ModuleNotFoundError`)
- [X] `config.py`: `pydantic-settings` `BaseSettings` reading `.env`, with `REFUND_APPROVAL_THRESHOLD_USD` typed numeric and `OLLAMA_HOST` validated as a URL. Done when an invalid value fails at import with a named field, not at first use.
- [X] `graph/__init__.py`: present, so `graph` is a regular package rather than a namespace package
- [X] `graph/schemas.py`: `TicketType = Literal["de-escalation", "resolution"]` declared once here, plus `TicketClassification` Pydantic model with `ticket_type: TicketType` and `rationale: str` (≤20 words), both carrying `Field(description=...)`
- [X] `graph/state.py`: `TriageState` `TypedDict` carrying `ticket_text`, `ticket_type`, `rationale`, `response`, importing `TicketType` from `schemas` rather than redeclaring the literal. Native `typing.TypedDict`; no `typing_extensions` on 3.13. Only `ticket_text` is required; the three fields written by later nodes are `NotRequired`.
- [X] `graph/prompts.py`: `CLASSIFIER_SYSTEM_PROMPT` (carries the default-to-`de-escalation` tie-break), `CARE_SYSTEM_PROMPT` (F3), `RESOLUTION_SYSTEM_PROMPT` (F4), plus `NO_COMPLETED_ACTION_RULE` shared by both agent prompts so one test substring protects F4 on both paths. Plain `str` module constants, no imports, so tests can assert on them without loading langchain.
- [X] `graph/nodes.py` → `classifier` node: `ChatOllama(...).with_structured_output(TicketClassification, include_raw=True)`, branching on `parsing_error` rather than `try/except` (F1)
- [X] `graph/nodes.py` → `classifier` guard: input under 3 words or empty skips classification and returns a clarification prompt
- [X] `graph/nodes.py` → `care_agent`: applies `CARE_SYSTEM_PROMPT`, acknowledging stated frustration before any next step (F3)
- [X] `graph/nodes.py` → `resolution_agent`: applies `RESOLUTION_SYSTEM_PROMPT`, restricted to "acknowledge and explain next steps" and forbidden from confirming any action occurred (F4, top risk in the matrix)
- [X] `graph/build.py`: `build_graph()` returning a compiled `StateGraph(TriageState)`, `add_edge(START, "classifier")`, `add_conditional_edges("classifier", route_by_tone, ["care_agent", "resolution_agent", END])`, both leaves to `END` (F2). `END` is in the path map because the guard path has already written a response and must not reach an agent.
- [X] `graph/build.py` → `route_by_tone`: returns node names, not labels. Only `resolution` reaches `resolution_agent`; ambiguous and fallback classifications default to `care_agent`, and an absent `ticket_type` (the guard path) goes straight to `END`.
- [X] `main.py`: CLI loop replacing the hello-world; reads ticket text, prints the routed response, exits cleanly on EOF/Ctrl-C (F5)
- [ ] `main.py`: Ollama startup health check with one retry and an actionable error naming the fix, never a raw traceback
- [ ] `main.py` → `read_ticket() -> str | None`: collect lines until a blank one so pasted multi-line tickets are not truncated at the first newline. `None` means EOF or interrupt, keeping exit handling in one place. Nothing under `graph/` changes: `split()` and `HumanMessage` already handle newlines. Accepted limitation: a ticket with an internal blank line submits only its first paragraph.
- [ ] `tests/test_classifier.py`: happy path for both labels, empty input, under-3-word input, malformed structured output, and the default-to-`de-escalation` fallback
- [ ] `tests/test_graph.py`: each label reaches the correct leaf node; state is populated end to end; unreachable Ollama degrades to the fallback message
- [ ] `tests/test_prompts.py`: assert `RESOLUTION_SYSTEM_PROMPT` still carries the no-completed-action constraint, so F4 is protected by a test rather than by memory
- [ ] Confirm no test makes a live model call, so CI never depends on a running Ollama

---

## P1.5 — Eval set & model benchmark

**Done when:** `.env.example` names a default model chosen on measured numbers, not assumption. Blocks the README Results table.

- [ ] `tests/fixtures/eval_set.json`: 50 labeled tickets spanning clearly-emotional, clearly-technical, and ambiguous/mixed, with the mixed subset separately identifiable so its accuracy can be reported on its own
- [ ] Eval runner marked `@pytest.mark.eval` so it stays out of normal CI runs
- [ ] `ollama pull llama3.2:3b` (not currently present locally)
- [ ] Measure accuracy and end-to-end latency for `llama3.2:3b` and `qwen2.5:7b` on the same eval set and hardware
- [ ] Measure the naive keyword/rule-based router baseline on the same eval set, for the README's comparison row
- [ ] Measure the F4 safety check: % of resolution responses that imply a completed action, target 0%
- [ ] `docs/MODEL_EVAL.md`: record both models' numbers and the hardware they were measured on
- [ ] Lock the winning model as the default in `.env.example`, `README.md`, and `DESIGN.md`; remove the ⚠️ provisional marker
- [ ] Reconcile the ≤4s latency guardrail in `DESIGN.md` against the measured result, adjusting the doc if the target proves wrong rather than quietly missing it

---

## P2 — Integration & Hardening

**Done when:** the reliability and privacy guardrails in DESIGN Phase 2 are enforced by a test, not by intention.

- [ ] Retry-once-then-fallback path fully wired: malformed structured output retries once, then falls back to `de-escalation` with a human-review flag on the state
- [ ] Automated check that no outbound network call leaves the machine during a test run (the Privacy guardrail)
- [ ] Structured logging honouring `LOG_LEVEL`, with the classifier `rationale` surfaced at debug
- [ ] `Dockerfile` for the `app` service
- [ ] `docker-compose.yml`: `app` + `ollama` sidecar, named volume for the model cache, `OLLAMA_HOST=http://ollama:11434`
- [ ] Entrypoint script pulls the model at container start rather than baking it into the image
- [ ] Verify `docker compose up` works from a clean state on a machine with no model cached

---

## P3 — Ship

**Done when:** every ⚠️ marker in the README is replaced by a measured number or an honest still-pending note. No estimates.

- [ ] `.github/workflows/ci.yml`: `ruff check`, `ruff format --check`, `pytest --cov` with the 80% gate, eval-marked tests excluded
- [ ] Confirm coverage ≥80% on `graph/` and `config.py`, adding tests rather than lowering the gate
- [ ] Replace the CI and coverage badges in `README.md` with live ones
- [ ] Record `docs/demo.gif`: 15–30s, a frustrated ticket routed to Care then a technical ticket routed to Resolution
- [ ] Restore the `docs/demo.gif` embed in `README.md`
- [ ] Fill the README Results table from the P1.5 and P3 measurements
- [ ] Rewrite the three README Engineering Challenges in past tense with the real mechanism and measured outcome
- [ ] Fill the Result bullet in DESIGN Phase 6 (STAR narrative) with the measured numbers
- [ ] Final pass: no ⚠️ marker left holding an unmeasured claim, no broken link or image in either doc
