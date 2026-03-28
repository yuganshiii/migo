"""Tests for KnowledgeMap model methods."""

from unittest.mock import patch

import pytest

from migo.models.knowledge_map import KnowledgeEntry, KnowledgeMap


@pytest.fixture
def km():
    """A KnowledgeMap with a variety of concepts at different confidence levels."""
    return KnowledgeMap(
        user_id="test_user",
        categories={
            "history": {
                "world_war_2": KnowledgeEntry(confidence=0.5, attempts=3, streak=1),
                "roman_empire": KnowledgeEntry(confidence=0.45, attempts=2, streak=0),
                "cold_war": KnowledgeEntry(confidence=0.0, attempts=0, streak=0),
                "french_revolution": KnowledgeEntry(confidence=0.0, attempts=0, streak=0),
                "ancient_egypt": KnowledgeEntry(confidence=0.8, attempts=5, streak=3),
                "viking_age": KnowledgeEntry(confidence=0.2, attempts=1, streak=0),
            }
        },
    )


class TestUpdateConfidence:
    def test_correct_increases_confidence(self, km):
        entry = km.update_confidence("history", "cold_war", correct=True)
        assert entry.confidence == pytest.approx(0.15)
        assert entry.attempts == 1
        assert entry.streak == 1

    def test_incorrect_decreases_confidence(self, km):
        entry = km.update_confidence("history", "world_war_2", correct=False)
        assert entry.confidence == pytest.approx(0.4)
        assert entry.attempts == 4
        assert entry.streak == 0

    def test_confidence_capped_at_1(self, km):
        entry = km.update_confidence("history", "ancient_egypt", correct=True)
        assert entry.confidence == pytest.approx(0.95)
        entry = km.update_confidence("history", "ancient_egypt", correct=True)
        assert entry.confidence == 1.0

    def test_confidence_floored_at_0(self, km):
        entry = km.update_confidence("history", "cold_war", correct=False)
        assert entry.confidence == 0.0

    def test_streak_increments_on_correct(self, km):
        km.update_confidence("history", "world_war_2", correct=True)
        entry = km.update_confidence("history", "world_war_2", correct=True)
        assert entry.streak == 3

    def test_streak_resets_on_incorrect(self, km):
        entry = km.update_confidence("history", "ancient_egypt", correct=False)
        assert entry.streak == 0

    def test_last_tested_is_set(self, km):
        entry = km.update_confidence("history", "cold_war", correct=True)
        assert entry.last_tested is not None

    def test_creates_new_concept_if_missing(self, km):
        entry = km.update_confidence("history", "new_topic", correct=True)
        assert entry.confidence == pytest.approx(0.15)
        assert entry.attempts == 1

    def test_creates_new_category_if_missing(self, km):
        entry = km.update_confidence("science", "physics", correct=False)
        assert entry.confidence == 0.0
        assert entry.attempts == 1


class TestGetFunZoneConcepts:
    def test_returns_concepts_in_fun_zone(self, km):
        result = km.get_fun_zone_concepts("history")
        assert "world_war_2" in result  # 0.5
        assert "roman_empire" in result  # 0.45
        assert len(result) == 2

    def test_excludes_outside_fun_zone(self, km):
        result = km.get_fun_zone_concepts("history")
        assert "ancient_egypt" not in result  # 0.8
        assert "cold_war" not in result  # 0.0
        assert "viking_age" not in result  # 0.2

    def test_empty_for_missing_category(self, km):
        assert km.get_fun_zone_concepts("nonexistent") == []

    def test_boundary_values_included(self):
        km = KnowledgeMap(
            categories={
                "test": {
                    "at_low": KnowledgeEntry(confidence=0.4),
                    "at_high": KnowledgeEntry(confidence=0.6),
                    "below_low": KnowledgeEntry(confidence=0.39),
                    "above_high": KnowledgeEntry(confidence=0.61),
                }
            }
        )
        result = km.get_fun_zone_concepts("test")
        assert "at_low" in result
        assert "at_high" in result
        assert "below_low" not in result
        assert "above_high" not in result


class TestGetUnseenConcepts:
    def test_returns_unseen_concepts(self, km):
        result = km.get_unseen_concepts("history")
        assert "cold_war" in result
        assert "french_revolution" in result
        assert len(result) == 2

    def test_excludes_seen_concepts(self, km):
        result = km.get_unseen_concepts("history")
        assert "world_war_2" not in result
        assert "ancient_egypt" not in result

    def test_empty_for_missing_category(self, km):
        assert km.get_unseen_concepts("nonexistent") == []


class TestGetNextConcept:
    def test_prefers_fun_zone(self, km):
        # Run multiple times to account for randomness
        results = {km.get_next_concept("history") for _ in range(50)}
        fun_zone = {"world_war_2", "roman_empire"}
        assert results.issubset(fun_zone)

    def test_falls_back_to_unseen(self):
        km = KnowledgeMap(
            categories={
                "science": {
                    "quantum": KnowledgeEntry(confidence=0.0),
                    "relativity": KnowledgeEntry(confidence=0.9),
                }
            }
        )
        results = {km.get_next_concept("science") for _ in range(20)}
        assert results == {"quantum"}

    def test_falls_back_to_closest_to_fun_zone(self):
        km = KnowledgeMap(
            categories={
                "science": {
                    "biology": KnowledgeEntry(confidence=0.3, attempts=2),
                    "chemistry": KnowledgeEntry(confidence=0.9, attempts=5),
                }
            }
        )
        # biology (0.3) is closer to midpoint 0.5 than chemistry (0.9)
        assert km.get_next_concept("science") == "biology"

    def test_returns_none_for_empty_category(self, km):
        assert km.get_next_concept("nonexistent") is None

    def test_returns_none_for_no_concepts(self):
        km = KnowledgeMap(categories={"empty": {}})
        assert km.get_next_concept("empty") is None
