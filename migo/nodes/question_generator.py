import json
import time

import anthropic
from loguru import logger

from migo.config import CLAUDE_MODEL, MAX_RETRIES
from migo.prompts.question_gen import QUESTION_GEN_SYSTEM, QUESTION_GEN_USER
from migo.retrieval.indexer import index_topic, query_topic
from migo.state import GameState

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        from migo.config import ANTHROPIC_API_KEY
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from Claude's response."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _retrieve_chunks(topic: str) -> tuple[list[str], str]:
    """Fetch and index a Wikipedia article, return (chunk_texts, source_url)."""
    # Convert snake_case back to readable form for Wikipedia lookup
    readable_topic = topic.replace("_", " ")
    try:
        index = index_topic(readable_topic)
        chunks = query_topic(index, readable_topic)
        texts = [c.text for c in chunks]
        source_url = chunks[0].source_url if chunks else ""
        return texts, source_url
    except (ValueError, ConnectionError) as e:
        logger.warning("Retrieval failed for '{}': {}", readable_topic, e)
        return [], ""


def question_generator(state: GameState) -> dict:
    """Generate a multiple-choice question with real Wikipedia retrieval."""
    topic = state["current_topic"]

    # Fetch fresh chunks for the current topic
    chunks, source_url = _retrieve_chunks(topic)

    if not chunks:
        # Fall back to whatever is already in state
        chunks = state.get("retrieved_chunks", [])
        source_url = state.get("source_url", "")

    context = "\n\n".join(chunks) if chunks else f"General knowledge about {topic}"

    user_msg = QUESTION_GEN_USER.format(context=context, concept=topic)
    client = _get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                system=QUESTION_GEN_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )

            raw = _strip_code_fences(response.content[0].text)
            data = json.loads(raw)

            options = data["options"]
            correct_key = data["correct_answer"]

            return {
                "current_question": data["question"],
                "current_options": [
                    f"{k}: {v}" for k, v in options.items()
                ],
                "correct_answer": f"{correct_key}: {options[correct_key]}",
                "source_url": source_url,
                "retrieved_chunks": chunks,
            }

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "Attempt {}/{} — bad response for '{}': {}", attempt, MAX_RETRIES, topic, e
            )
        except anthropic.APIError as e:
            logger.warning(
                "Attempt {}/{} — API error for '{}': {}", attempt, MAX_RETRIES, topic, e
            )

        if attempt < MAX_RETRIES:
            backoff = 2 ** (attempt - 1)
            logger.info("Retrying in {}s...", backoff)
            time.sleep(backoff)

    logger.error("All {} attempts failed for topic '{}'", MAX_RETRIES, topic)
    return {
        "current_question": f"Sorry, failed to generate a question about {topic}.",
        "current_options": [],
        "correct_answer": "",
        "source_url": source_url,
        "retrieved_chunks": chunks,
    }
