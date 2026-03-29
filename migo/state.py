from __future__ import annotations

from typing import TypedDict


class ConceptState(TypedDict):
    confidence: float
    attempts: int
    streak: int
    last_tested: str | None


class RoundResult(TypedDict):
    round: int
    score: int
    total: int
    timestamp: str


class GameState(TypedDict):
    user_id: str
    category: str
    current_topic: str
    current_question: str
    current_options: list[str]
    correct_answer: str
    user_answer: str
    is_correct: bool | None
    explanation: str
    source_url: str
    retrieved_chunks: list[str]
    knowledge_map: dict[str, dict[str, ConceptState]]
    round_number: int
    max_rounds: int
    round_score: int
    round_history: list[RoundResult]
    should_end: bool
