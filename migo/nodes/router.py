from loguru import logger

from migo.models.knowledge_map import KnowledgeMap
from migo.state import GameState


def router(state: GameState) -> dict:
    """Pick the next concept to quiz or end the round."""
    round_number = state["round_number"]
    max_rounds = state["max_rounds"]
    category = state["category"]

    if round_number >= max_rounds:
        logger.info("Round limit reached ({}/{}), ending", round_number, max_rounds)
        return {"should_end": True}

    km = KnowledgeMap(categories=state.get("knowledge_map", {}))
    topic = km.get_next_concept(category)

    if topic is None:
        logger.warning("No concepts available for category '{}', ending", category)
        return {"should_end": True}

    logger.info(
        "Round {}/{} — selected topic: '{}'", round_number + 1, max_rounds, topic
    )
    return {
        "current_topic": topic,
        "round_number": round_number + 1,
        "should_end": False,
    }
