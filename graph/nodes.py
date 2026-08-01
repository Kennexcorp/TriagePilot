"""LangGraph node callables.

Each node takes the whole TriageState and returns a partial update that
LangGraph merges in. No node decides routing; that lives in build.py.
"""

from langchain_core.messages import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_ollama import ChatOllama

from config import settings
from graph.prompts import (
    CARE_SYSTEM_PROMPT,
    CLASSIFIER_SYSTEM_PROMPT,
    RESOLUTION_SYSTEM_PROMPT,
)
from graph.schemas import TicketClassification
from graph.state import TriageState

MIN_TICKET_WORDS = 3
CLARIFICATION_RESPONSE = (
    "Could you tell me a bit more about what you need help with? "
    "A sentence or two about the problem is enough to get you to the "
    "right person."
)
FALLBACK_RATIONALE = "Classification failed; defaulted to de-escalation"


def _build_chat_model() -> ChatOllama:
    """Construct the Ollama client from validated settings."""

    return ChatOllama(
        model=settings.ollama_model,
        base_url=str(settings.ollama_host).rstrip("/"),
        temperature=0,
    )


def classifier(state: TriageState) -> TriageState:
    """Label the ticket, or ask for detail when there is too little to go on."""
    ticket_text = state["ticket_text"]
    if len(ticket_text.split()) < MIN_TICKET_WORDS:
        return {"response": CLARIFICATION_RESPONSE}

    model = _build_chat_model().with_structured_output(
        TicketClassification,
        include_raw=True,  # include_raw=True converts parse failures into data
    )
    result = model.invoke(
        [
            SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=ticket_text),
        ]
    )

    classification = result["parsed"]

    if result["parsing_error"] is not None or classification is None:
        # De-escalation, not resolution: acknowledging tone that was not
        # needed is a cheaper mistake than handing a furious customer a
        # checklist.

        return {"ticket_type": "de-escalation", "rationale": FALLBACK_RATIONALE}

    return {
        "ticket_type": classification.ticket_type,
        "rationale": classification.rationale,
    }


def care_agent(state: TriageState) -> TriageState:
    """Respond to a ticket routed for de-escalation."""
    return {"response": _respond(state, CARE_SYSTEM_PROMPT)}


def resolution_agent(state: TriageState) -> TriageState:
    """Respond to a ticket routed for resolution."""
    return {"response": _respond(state, RESOLUTION_SYSTEM_PROMPT)}


def _respond(state: TriageState, system_prompt: str) -> str:
    """Generate the customer-facing reply text under the given system prompt."""
    reply = _build_chat_model().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["ticket_text"]),
        ]
    )

    return reply.text
