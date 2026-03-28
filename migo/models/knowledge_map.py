from __future__ import annotations

import random
from datetime import datetime

from pydantic import BaseModel, Field

from migo.config import (
    CONFIDENCE_BOOST,
    CONFIDENCE_PENALTY,
    FUN_ZONE_HIGH,
    FUN_ZONE_LOW,
)


class RetrievedChunk(BaseModel):
    text: str
    source_url: str
    score: float


class Question(BaseModel):
    question_text: str
    options: list[str]
    correct_answer: str
    difficulty: str = "medium"
    source_url: str = ""


class KnowledgeEntry(BaseModel):
    confidence: float = 0.0
    attempts: int = 0
    streak: int = 0
    last_tested: datetime | None = None


class KnowledgeMap(BaseModel):
    user_id: str = "default"
    categories: dict[str, dict[str, KnowledgeEntry]] = Field(default_factory=dict)
    overall_score: int = 0
    round_history: list[dict] = Field(default_factory=list)

    def get_entry(self, category: str, concept: str) -> KnowledgeEntry:
        return self.categories.setdefault(category, {}).setdefault(
            concept, KnowledgeEntry()
        )

    def update_confidence(
        self, category: str, concept: str, correct: bool
    ) -> KnowledgeEntry:
        """Update confidence for a concept after an answer.

        Correct: +0.15 (capped at 1.0), streak incremented.
        Incorrect: -0.1 (floored at 0.0), streak reset to 0.
        """
        entry = self.get_entry(category, concept)
        entry.attempts += 1
        entry.last_tested = datetime.now()
        if correct:
            entry.confidence = min(entry.confidence + CONFIDENCE_BOOST, 1.0)
            entry.streak += 1
        else:
            entry.confidence = max(entry.confidence - CONFIDENCE_PENALTY, 0.0)
            entry.streak = 0
        return entry

    def get_fun_zone_concepts(self, category: str) -> list[str]:
        """Return concepts with confidence in the fun zone (0.4–0.6)."""
        concepts = self.categories.get(category, {})
        return [
            name
            for name, entry in concepts.items()
            if FUN_ZONE_LOW <= entry.confidence <= FUN_ZONE_HIGH
        ]

    def get_unseen_concepts(self, category: str) -> list[str]:
        """Return concepts with confidence == 0.0 (never answered correctly)."""
        concepts = self.categories.get(category, {})
        return [
            name for name, entry in concepts.items() if entry.confidence == 0.0
        ]

    def get_next_concept(self, category: str) -> str | None:
        """Pick the next concept to quiz.

        Prefers fun-zone concepts (0.4–0.6), falls back to unseen (0.0).
        Returns None if the category has no concepts.
        """
        fun_zone = self.get_fun_zone_concepts(category)
        if fun_zone:
            return random.choice(fun_zone)

        unseen = self.get_unseen_concepts(category)
        if unseen:
            return random.choice(unseen)

        # All concepts are outside fun zone and seen — pick the one closest to fun zone
        concepts = self.categories.get(category, {})
        if not concepts:
            return None

        midpoint = (FUN_ZONE_LOW + FUN_ZONE_HIGH) / 2
        return min(concepts, key=lambda c: abs(concepts[c].confidence - midpoint))
