from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


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

    def update_entry(
        self, category: str, concept: str, is_correct: bool
    ) -> KnowledgeEntry:
        entry = self.get_entry(category, concept)
        entry.attempts += 1
        entry.last_tested = datetime.now()
        if is_correct:
            entry.confidence = min(entry.confidence + 0.15, 1.0)
            entry.streak += 1
        else:
            entry.confidence = max(entry.confidence - 0.1, 0.0)
            entry.streak = 0
        return entry
