from typing import Literal

from pydantic import BaseModel, Field

TicketType = Literal["de-escalation", "resolution"]


class TicketClassification(BaseModel):
    """Classify an inbound support ticket by the customer's emotional register."""

    ticket_type: TicketType = Field(
        description="Classification of the ticket based on its content"
    )
    rationale: str = Field(
        description="Reasoning for the classification", max_length=250
    )
