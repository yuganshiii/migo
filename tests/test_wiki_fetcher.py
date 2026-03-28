"""Tests for wiki_fetcher module."""

from unittest.mock import MagicMock, patch

import pytest
import wikipedia

from migo.retrieval.wiki_fetcher import fetch_wikipedia_article


@pytest.fixture
def mock_page():
    page = MagicMock()
    page.title = "Python (programming language)"
    page.content = "Python is a high-level programming language."
    return page


class TestFetchWikipediaArticle:
    """Tests for fetch_wikipedia_article."""

    @patch("migo.retrieval.wiki_fetcher.wikipedia.page")
    def test_successful_fetch(self, mock_wiki_page, mock_page):
        mock_wiki_page.return_value = mock_page

        result = fetch_wikipedia_article("Python programming")

        assert result == "Python is a high-level programming language."
        mock_wiki_page.assert_called_once_with("Python programming", auto_suggest=True)

    @patch("migo.retrieval.wiki_fetcher.wikipedia.page")
    def test_disambiguation_picks_first_option(self, mock_wiki_page, mock_page):
        disambiguation_error = wikipedia.exceptions.DisambiguationError(
            "Python", ["Python (programming language)", "Python (snake)"]
        )
        mock_wiki_page.side_effect = [disambiguation_error, mock_page]

        result = fetch_wikipedia_article("Python")

        assert result == "Python is a high-level programming language."
        assert mock_wiki_page.call_count == 2
        mock_wiki_page.assert_called_with(
            "Python (programming language)", auto_suggest=False
        )

    @patch("migo.retrieval.wiki_fetcher.wikipedia.page")
    def test_disambiguation_then_page_error_raises_value_error(self, mock_wiki_page):
        disambiguation_error = wikipedia.exceptions.DisambiguationError(
            "Test", ["Option1"]
        )
        mock_wiki_page.side_effect = [
            disambiguation_error,
            wikipedia.exceptions.PageError("Option1"),
        ]

        with pytest.raises(ValueError, match="Could not find a Wikipedia article"):
            fetch_wikipedia_article("Test")

    @patch("migo.retrieval.wiki_fetcher.wikipedia.page")
    def test_page_error_raises_value_error(self, mock_wiki_page):
        mock_wiki_page.side_effect = wikipedia.exceptions.PageError("Nonexistent")

        with pytest.raises(ValueError, match="No Wikipedia article found"):
            fetch_wikipedia_article("Nonexistent topic xyz")

    @patch("migo.retrieval.wiki_fetcher.wikipedia.page")
    def test_network_error_raises_connection_error(self, mock_wiki_page):
        mock_wiki_page.side_effect = ConnectionError("Network unreachable")

        with pytest.raises(ConnectionError, match="Failed to fetch"):
            fetch_wikipedia_article("Python")

    @patch("migo.retrieval.wiki_fetcher.wikipedia.page")
    def test_returns_full_content(self, mock_wiki_page):
        long_content = "A" * 50000
        page = MagicMock()
        page.title = "Big Article"
        page.content = long_content
        mock_wiki_page.return_value = page

        result = fetch_wikipedia_article("Big Article")

        assert len(result) == 50000
