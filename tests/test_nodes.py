"""Unit tests for all LangGraph nodes. Claude API calls are mocked."""

import json
from unittest.mock import MagicMock, patch

import pytest

from migo.nodes.evaluator import evaluator
from migo.nodes.knowledge_map_updater import knowledge_map_updater
from migo.nodes.question_generator import question_generator
from migo.nodes.router import router


def _base_state(**overrides) -> dict:
    state = {
        "user_id": "test",
        "category": "general_knowledge",
        "current_topic": "world_war_2",
        "current_question": "",
        "current_options": [],
        "correct_answer": "",
        "user_answer": "",
        "is_correct": None,
        "explanation": "",
        "source_url": "https://en.wikipedia.org/wiki/World_War_II",
        "retrieved_chunks": ["Germany invaded Poland in 1939."],
        "knowledge_map": {
            "general_knowledge": {
                "world_war_2": {
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
    state.update(overrides)
    return state


# ── Router tests ──


class TestRouter:
    def test_picks_topic(self):
        result = router(_base_state())
        assert result["current_topic"] == "world_war_2"
        assert result["round_number"] == 1
        assert result["should_end"] is False

    def test_ends_at_max_rounds(self):
        result = router(_base_state(round_number=5))
        assert result["should_end"] is True

    def test_ends_when_no_concepts(self):
        result = router(_base_state(knowledge_map={}))
        assert result["should_end"] is True

    def test_prefers_fun_zone(self):
        km = {
            "general_knowledge": {
                "topic_a": {"confidence": 0.9, "attempts": 5, "streak": 3, "last_tested": None},
                "topic_b": {"confidence": 0.5, "attempts": 2, "streak": 1, "last_tested": None},
            },
        }
        result = router(_base_state(knowledge_map=km))
        assert result["current_topic"] == "topic_b"


# ── Question Generator tests ──


MOCK_QUESTION_JSON = json.dumps({
    "question": "When did Germany invade Poland?",
    "options": {"A": "1937", "B": "1938", "C": "1939", "D": "1940"},
    "correct_answer": "C",
    "difficulty": "easy",
})


def _mock_claude_response(text: str) -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    return response


class TestQuestionGenerator:
    @patch("migo.nodes.question_generator._get_client")
    def test_returns_valid_question(self, mock_get_client):
        client = MagicMock()
        client.messages.create.return_value = _mock_claude_response(MOCK_QUESTION_JSON)
        mock_get_client.return_value = client

        result = question_generator(_base_state())

        assert "When did Germany invade Poland?" in result["current_question"]
        assert len(result["current_options"]) == 4
        assert "C:" in result["correct_answer"]
        assert result["source_url"] != ""

    @patch("migo.nodes.question_generator._get_client")
    def test_retries_on_bad_json(self, mock_get_client):
        client = MagicMock()
        client.messages.create.side_effect = [
            _mock_claude_response("not json"),
            _mock_claude_response(MOCK_QUESTION_JSON),
        ]
        mock_get_client.return_value = client

        result = question_generator(_base_state())
        assert result["current_question"] == "When did Germany invade Poland?"
        assert client.messages.create.call_count == 2

    @patch("migo.nodes.question_generator._get_client")
    @patch("migo.nodes.question_generator.time.sleep")
    def test_fallback_after_all_retries(self, mock_sleep, mock_get_client):
        client = MagicMock()
        client.messages.create.return_value = _mock_claude_response("bad")
        mock_get_client.return_value = client

        result = question_generator(_base_state())
        assert "failed to generate" in result["current_question"].lower()
        assert result["current_options"] == []
        assert client.messages.create.call_count == 3


# ─�� Evaluator tests ──


MOCK_CORRECT_EVAL = json.dumps({
    "is_correct": True,
    "partial_credit": False,
    "explanation": "1939 is correct. Germany invaded Poland on September 1, 1939.",
})

MOCK_INCORRECT_EVAL = json.dumps({
    "is_correct": False,
    "partial_credit": False,
    "explanation": "1940 is incorrect. The invasion began in 1939.",
})


class TestEvaluator:
    @patch("migo.nodes.evaluator._get_client")
    def test_correct_answer(self, mock_get_client):
        client = MagicMock()
        client.messages.create.return_value = _mock_claude_response(MOCK_CORRECT_EVAL)
        mock_get_client.return_value = client

        state = _base_state(
            current_question="When did Germany invade Poland?",
            correct_answer="C: 1939",
            user_answer="C: 1939",
        )
        result = evaluator(state)

        assert result["is_correct"] is True
        assert "1939" in result["explanation"]

    @patch("migo.nodes.evaluator._get_client")
    def test_incorrect_answer(self, mock_get_client):
        client = MagicMock()
        client.messages.create.return_value = _mock_claude_response(MOCK_INCORRECT_EVAL)
        mock_get_client.return_value = client

        state = _base_state(
            current_question="When did Germany invade Poland?",
            correct_answer="C: 1939",
            user_answer="D: 1940",
        )
        result = evaluator(state)

        assert result["is_correct"] is False

    @patch("migo.nodes.evaluator._get_client")
    @patch("migo.nodes.evaluator.time.sleep")
    def test_fallback_on_failure(self, mock_sleep, mock_get_client):
        client = MagicMock()
        client.messages.create.return_value = _mock_claude_response("bad")
        mock_get_client.return_value = client

        state = _base_state(
            current_question="Q?",
            correct_answer="A",
            user_answer="B",
        )
        result = evaluator(state)

        assert result["is_correct"] is False
        assert "could not evaluate" in result["explanation"].lower()


# ── Knowledge Map Updater tests ──


class TestKnowledgeMapUpdater:
    def test_correct_answer_updates(self):
        state = _base_state(is_correct=True, round_score=2)
        result = knowledge_map_updater(state)

        assert result["round_score"] == 3
        entry = result["knowledge_map"]["general_knowledge"]["world_war_2"]
        assert entry.confidence == 0.15
        assert entry.attempts == 1
        assert entry.streak == 1

    def test_incorrect_answer_updates(self):
        state = _base_state(is_correct=False, round_score=2)
        # Start with some confidence so we can see it decrease
        state["knowledge_map"]["general_knowledge"]["world_war_2"]["confidence"] = 0.5
        result = knowledge_map_updater(state)

        assert result["round_score"] == 2
        entry = result["knowledge_map"]["general_knowledge"]["world_war_2"]
        assert entry.confidence == pytest.approx(0.4)
        assert entry.streak == 0

    def test_confidence_floor(self):
        state = _base_state(is_correct=False)
        result = knowledge_map_updater(state)

        entry = result["knowledge_map"]["general_knowledge"]["world_war_2"]
        assert entry.confidence == 0.0
