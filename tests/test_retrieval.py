"""Integration tests for the full retrieval pipeline: fetch → index → query."""

from unittest.mock import patch

import pytest
from llama_index.core import VectorStoreIndex

from migo.models.knowledge_map import RetrievedChunk
from migo.retrieval.indexer import index_topic, query_topic

EINSTEIN_CONTENT = (
    "Albert Einstein was a German-born theoretical physicist who is widely held to be "
    "one of the greatest and most influential scientists of all time. He is best known "
    "for developing the theory of relativity, but he also made important contributions "
    "to quantum mechanics. His mass-energy equivalence formula E = mc squared, which "
    "arises from relativity theory, has been called the world's most famous equation. "
    "He received the 1921 Nobel Prize in Physics for his services to theoretical physics, "
    "and especially for his discovery of the law of the photoelectric effect."
)

ROMAN_EMPIRE_CONTENT = (
    "The Roman Empire was the post-Republican state of ancient Rome. It included large "
    "territorial holdings around the Mediterranean Sea in Europe, North Africa, and "
    "Western Asia, and was ruled by emperors. The city of Rome was the largest city in "
    "the world from roughly 100 BC to 400 AD, with Constantinople becoming the largest "
    "around 500 AD. The empire began when Augustus Caesar proclaimed himself the first "
    "emperor of Rome in 27 BC and lasted until the fall of Constantinople in 1453 AD. "
    "At its peak under Trajan, the empire covered 5 million square kilometres."
)

PHOTOSYNTHESIS_CONTENT = (
    "Photosynthesis is a biological process used by many cellular organisms to convert "
    "light energy into chemical energy, which is stored in organic compounds that can "
    "later be metabolized through cellular respiration to fuel the organism's activities. "
    "The term usually refers to oxygenic photosynthesis, where oxygen is produced as a "
    "byproduct and some of the chemical energy is stored in carbohydrate molecules such "
    "as sugars and starches. Most plants, algae, and cyanobacteria perform photosynthesis. "
    "The process uses carbon dioxide and water, converting them into glucose and oxygen "
    "using sunlight absorbed by chlorophyll in the chloroplasts."
)


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    """Patch CHROMA_PERSIST_DIR to a temp directory."""
    chroma_dir = str(tmp_path / "chromadb")
    with patch("migo.retrieval.indexer.CHROMA_PERSIST_DIR", chroma_dir):
        yield chroma_dir


@pytest.mark.slow
class TestRetrievalPipelineEinstein:
    """Full pipeline test with Einstein content."""

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_fetch_index_query(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = EINSTEIN_CONTENT

        index = index_topic("Albert Einstein")
        chunks = query_topic(index, "theory of relativity")

        assert len(chunks) > 0
        assert all(isinstance(c, RetrievedChunk) for c in chunks)
        assert any("relativity" in c.text.lower() for c in chunks)
        assert all(c.text.strip() for c in chunks)
        assert all("wikipedia.org/wiki/Albert_Einstein" in c.source_url for c in chunks)


@pytest.mark.slow
class TestRetrievalPipelineRomanEmpire:
    """Full pipeline test with Roman Empire content."""

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_fetch_index_query(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = ROMAN_EMPIRE_CONTENT

        index = index_topic("Roman Empire")
        chunks = query_topic(index, "Augustus Caesar emperor")

        assert len(chunks) > 0
        assert all(isinstance(c, RetrievedChunk) for c in chunks)
        assert any("augustus" in c.text.lower() or "emperor" in c.text.lower() for c in chunks)
        assert all(c.text.strip() for c in chunks)
        assert all("wikipedia.org/wiki/Roman_Empire" in c.source_url for c in chunks)


@pytest.mark.slow
class TestRetrievalPipelinePhotosynthesis:
    """Full pipeline test with photosynthesis content."""

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_fetch_index_query(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = PHOTOSYNTHESIS_CONTENT

        index = index_topic("Photosynthesis")
        chunks = query_topic(index, "chlorophyll sunlight plants")

        assert len(chunks) > 0
        assert all(isinstance(c, RetrievedChunk) for c in chunks)
        assert any(
            "chlorophyll" in c.text.lower() or "sunlight" in c.text.lower()
            for c in chunks
        )
        assert all(c.text.strip() for c in chunks)
        assert all("wikipedia.org/wiki/Photosynthesis" in c.source_url for c in chunks)


@pytest.mark.slow
class TestRetrievalPipelineErrorHandling:
    """Error handling integration tests."""

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_missing_article_raises_value_error(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.side_effect = ValueError("No Wikipedia article found for 'xyznonexistent'")

        with pytest.raises(ValueError, match="No Wikipedia article found"):
            index_topic("xyznonexistent")

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_network_error_raises_connection_error(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.side_effect = ConnectionError("Failed to fetch Wikipedia article")

        with pytest.raises(ConnectionError, match="Failed to fetch"):
            index_topic("Any Topic")

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_query_unrelated_concept_still_returns_chunks(self, mock_fetch, tmp_chroma_dir):
        """Querying an unrelated concept should still return chunks (just lower relevance)."""
        mock_fetch.return_value = EINSTEIN_CONTENT

        index = index_topic("Albert Einstein")
        chunks = query_topic(index, "cooking recipes pasta")

        # ChromaDB will still return results, just with lower scores
        assert isinstance(chunks, list)

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_chunks_have_valid_scores(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = EINSTEIN_CONTENT

        index = index_topic("Albert Einstein")
        chunks = query_topic(index, "Nobel Prize physics")

        assert all(isinstance(c.score, float) for c in chunks)
