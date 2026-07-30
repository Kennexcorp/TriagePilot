from typing import NotRequired, TypedDict

from graph.schemas import TicketType


class TriageState(TypedDict):
    """The channel LangGraph threads between nodes.

    Typing only, never validated at runtime; that boundary is TicketClassification
    in schemas.py. Only ticket_text exists at START. The classifier adds
    ticket_type and rationale, the leaf agents add response.
    """

    ticket_text: str
    ticket_type: NotRequired[TicketType]
    rationale: NotRequired[str]
    response: NotRequired[str]
