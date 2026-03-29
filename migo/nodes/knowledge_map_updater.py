from loguru import logger

from migo.models.knowledge_map import KnowledgeMap
from migo.state import GameState


def knowledge_map_updater(state: GameState) -> dict:
    """Update the knowledge map and round score after an answer."""
    category = state["category"]
    topic = state["current_topic"]
    is_correct = state["is_correct"]
    round_score = state["round_score"]

    km = KnowledgeMap(categories=state.get("knowledge_map", {}))
    entry = km.update_confidence(category, topic, is_correct)

    if is_correct:
        round_score += 1

    logger.info(
        "Updated '{}' — correct={}, confidence={:.2f}, streak={}, score={}",
        topic, is_correct, entry.confidence, entry.streak, round_score,
    )

    return {
        "knowledge_map": km.categories,
        "round_score": round_score,
    }
