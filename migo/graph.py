from langgraph.graph import END, StateGraph

from migo.nodes.evaluator import evaluator
from migo.nodes.knowledge_map_updater import knowledge_map_updater
from migo.nodes.question_generator import question_generator
from migo.nodes.router import router
from migo.state import GameState


def _should_end(state: GameState) -> str:
    if state.get("should_end", False):
        return "end"
    return "continue"


def build_graph():
    graph = StateGraph(GameState)

    graph.add_node("router", router)
    graph.add_node("question_generator", question_generator)
    graph.add_node("evaluator", evaluator)
    graph.add_node("knowledge_map_updater", knowledge_map_updater)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _should_end,
        {"end": END, "continue": "question_generator"},
    )
    graph.add_edge("question_generator", "evaluator")
    graph.add_edge("evaluator", "knowledge_map_updater")
    graph.add_edge("knowledge_map_updater", "router")

    return graph.compile()
