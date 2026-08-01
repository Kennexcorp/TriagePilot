"""Graph construction and the routing decision.

This module owns which node runs next. Nodes never make that decision.
"""

from langgraph.constants import START
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from graph.nodes import care_agent, classifier, resolution_agent
from graph.state import TriageState


def route_by_tone(state: TriageState) -> str:
    """Pick the next node from the classification, or end if there is none."""
    ticket_type = state.get("ticket_type")
    if ticket_type is None:
        return END
    if ticket_type == "resolution":
        return "resolution_agent"
    return "care_agent"


def build_graph() -> CompiledStateGraph:
    """Wire the classifier to the two leaf agents and compile."""
    builder = StateGraph(TriageState)

    builder.add_node("classifier", classifier)
    builder.add_node("care_agent", care_agent)
    builder.add_node("resolution_agent", resolution_agent)

    builder.add_edge(START, "classifier")
    builder.add_conditional_edges(
        "classifier", route_by_tone, ["care_agent", "resolution_agent", END]
    )
    builder.add_edge("care_agent", END)
    builder.add_edge("resolution_agent", END)

    return builder.compile()
