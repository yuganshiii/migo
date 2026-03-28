"""Tests for indexer module."""

import shutil
import tempfile
from unittest.mock import patch

import pytest
from llama_index.core import VectorStoreIndex

from migo.models.knowledge_map import RetrievedChunk
from migo.retrieval.indexer import index_topic, query_topic


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    """Patch CHROMA_PERSIST_DIR to a temp directory."""
    chroma_dir = str(tmp_path / "chromadb")
    with patch("migo.retrieval.indexer.CHROMA_PERSIST_DIR", chroma_dir):
        yield chroma_dir


FAKE_CONTENT = (
    "Albert Einstein was a German-born theoretical physicist. "
    "He developed the theory of relativity, one of the two pillars of modern physics. "
    "His work is also known for its influence on the philosophy of science. "
    "Einstein is best known in popular culture for his mass-energy equivalence formula E = mc squared."
)


class TestIndexTopic:
    """Tests for index_topic."""

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_returns_vector_store_index(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = FAKE_CONTENT

        index = index_topic("Albert Einstein")

        assert isinstance(index, VectorStoreIndex)
        mock_fetch.assert_called_once_with("Albert Einstein")

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_wiki_fetch_error_propagates(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.side_effect = ValueError("No Wikipedia article found for 'xyz'")

        with pytest.raises(ValueError, match="No Wikipedia article found"):
            index_topic("xyz")

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_index_is_queryable(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = FAKE_CONTENT

        index = index_topic("Albert Einstein")
        retriever = index.as_retriever(similarity_top_k=2)
        results = retriever.retrieve("theory of relativity")

        assert len(results) > 0
        assert any("relativity" in r.text.lower() for r in results)

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_multiple_topics_same_collection(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = FAKE_CONTENT

        index1 = index_topic("Topic A")
        index2 = index_topic("Topic B")

        assert isinstance(index1, VectorStoreIndex)
        assert isinstance(index2, VectorStoreIndex)


class TestQueryTopic:
    """Tests for query_topic."""

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_returns_retrieved_chunks(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = FAKE_CONTENT
        index = index_topic("Albert Einstein")

        chunks = query_topic(index, "theory of relativity")

        assert len(chunks) > 0
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_chunk_has_text_and_source_url(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = FAKE_CONTENT
        index = index_topic("Albert Einstein")

        chunks = query_topic(index, "relativity")

        assert chunks[0].text
        assert "wikipedia.org/wiki/Albert_Einstein" in chunks[0].source_url

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_chunk_has_score(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = FAKE_CONTENT
        index = index_topic("Albert Einstein")

        chunks = query_topic(index, "relativity")

        assert isinstance(chunks[0].score, float)

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_respects_top_k(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = FAKE_CONTENT
        index = index_topic("Albert Einstein")

        chunks = query_topic(index, "physics", top_k=1)

        assert len(chunks) <= 1

    @patch("migo.retrieval.indexer.fetch_wikipedia_article")
    def test_default_top_k_is_3(self, mock_fetch, tmp_chroma_dir):
        mock_fetch.return_value = FAKE_CONTENT
        index = index_topic("Albert Einstein")

        chunks = query_topic(index, "physics")

        # With short content there may be fewer than 3 chunks total,
        # but we should never exceed 3
        assert len(chunks) <= 3
