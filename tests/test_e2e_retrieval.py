"""End-to-end test: real Wikipedia retrieval → real Claude question generation.

Requires ANTHROPIC_API_KEY in .env and network access.
Run with: pytest tests/test_e2e_retrieval.py -v -s
"""

import os

import pytest

from migo.config import ANTHROPIC_API_KEY
from migo.nodes import question_generator as qg_module
from migo.nodes.question_generator import question_generator
from migo.nodes.router import router


def _make_state(topic: str) -> dict:
    category = "general_knowledge"
    concept = topic.lower().replace(" ", "_")
    return {
        "user_id": "e2e_test",
        "category": category,
        "current_topic": concept,
        "current_question": "",
        "current_options": [],
        "correct_answer": "",
        "user_answer": "",
        "is_correct": None,
        "explanation": "",
        "source_url": "",
        "retrieved_chunks": [],
        "knowledge_map": {
            category: {
                concept: {
                    "confidence": 0.0,
                    "attempts": 0,
                    "streak": 0,
                    "last_tested": None,
                },
            },
        },
        "round_number": 0,
        "max_rounds": 5,
        "round_score": 0,
        "round_history": [],
        "should_end": False,
    }


@pytest.fixture(autouse=True)
def _require_api_key_and_reset_client():
    if not ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not set")
    # Reset client singleton so each test gets a fresh one
    qg_module._client = None
    yield
    qg_module._client = None


TOPICS = ["Solar System", "French Revolution", "Python programming language"]


@pytest.mark.parametrize("topic", TOPICS)
def test_full_pipeline(topic):
    """Router picks topic → question_generator fetches Wikipedia → Claude generates question."""
    state = _make_state(topic)

    # Router picks the concept
    router_result = router(state)
    assert router_result["should_end"] is False
    state.update(router_result)

    # Question generator fetches real chunks + calls Claude
    qg_result = question_generator(state)

    assert qg_result["current_question"], f"No question generated for '{topic}'"
    assert len(qg_result["current_options"]) == 4, f"Expected 4 options for '{topic}'"
    assert qg_result["correct_answer"], f"No correct answer for '{topic}'"
    assert qg_result["source_url"], f"No source URL for '{topic}'"
    assert qg_result["retrieved_chunks"], f"No chunks retrieved for '{topic}'"

    print(f"\n--- {topic} ---")
    print(f"Q: {qg_result['current_question']}")
    for opt in qg_result["current_options"]:
        print(f"  {opt}")
    print(f"Answer: {qg_result['correct_answer']}")
    print(f"Source: {qg_result['source_url']}")
    print(f"Chunks: {len(qg_result['retrieved_chunks'])}")
