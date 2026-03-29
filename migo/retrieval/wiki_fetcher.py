"""Wikipedia article fetcher for the retrieval layer."""

import wikipedia
from loguru import logger


def fetch_wikipedia_article(topic: str) -> str:
    """Fetch the full plaintext content of a Wikipedia article for the given topic.

    Handles DisambiguationError by picking the first suggestion,
    PageError by raising a clean ValueError, and network errors.
    """
    logger.info("Fetching Wikipedia article for topic: {}", topic)

    try:
        page = wikipedia.page(topic, auto_suggest=True)
        logger.success("Fetched article: '{}' ({} chars)", page.title, len(page.content))
        return page.content

    except wikipedia.exceptions.DisambiguationError as e:
        first_option = e.options[0]
        logger.warning(
            "Disambiguation for '{}', trying first option: '{}'", topic, first_option
        )
        try:
            page = wikipedia.page(first_option, auto_suggest=False)
            logger.success(
                "Fetched article: '{}' ({} chars)", page.title, len(page.content)
            )
            return page.content
        except wikipedia.exceptions.PageError:
            raise ValueError(
                f"Could not find a Wikipedia article for '{topic}' "
                f"(tried disambiguation option '{first_option}')"
            )

    except wikipedia.exceptions.PageError:
        logger.warning("No direct page for '{}', falling back to search", topic)
        results = wikipedia.search(topic)
        if results:
            try:
                page = wikipedia.page(results[0], auto_suggest=False)
                logger.success(
                    "Fetched article via search: '{}' ({} chars)",
                    page.title, len(page.content),
                )
                return page.content
            except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
                pass
        raise ValueError(f"No Wikipedia article found for '{topic}'")

    except Exception as e:
        logger.error("Network or unexpected error fetching '{}': {}", topic, e)
        raise ConnectionError(
            f"Failed to fetch Wikipedia article for '{topic}': {e}"
        ) from e
