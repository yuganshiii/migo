import json
import time

import anthropic
from loguru import logger

from migo.config import CLAUDE_MODEL, MAX_RETRIES
from migo.prompts.evaluation import EVAL_SYSTEM, EVAL_USER
from migo.state import GameState


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from Claude's response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        from migo.config import ANTHROPIC_API_KEY
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def evaluator(state: GameState) -> dict:
    """Evaluate the user's answer against the correct answer using Claude."""
    question = state["current_question"]
    correct_answer = state["correct_answer"]
    user_answer = state["user_answer"]
    source_url = state.get("source_url", "")

    user_msg = EVAL_USER.format(
        question=question,
        correct_answer=correct_answer,
        user_answer=user_answer,
        source_url=source_url,
    )
    client = _get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                system=EVAL_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )

            raw = _strip_code_fences(response.content[0].text)
            data = json.loads(raw)

            return {
                "is_correct": data["is_correct"],
                "explanation": data["explanation"],
                "source_url": source_url,
            }

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "Attempt {}/{} — bad response: {}", attempt, MAX_RETRIES, e
            )
        except anthropic.APIError as e:
            logger.warning(
                "Attempt {}/{} — API error: {}", attempt, MAX_RETRIES, e
            )

        if attempt < MAX_RETRIES:
            backoff = 2 ** (attempt - 1)
            logger.info("Retrying in {}s...", backoff)
            time.sleep(backoff)

    logger.error("All {} attempts failed for evaluation", MAX_RETRIES)
    return {
        "is_correct": False,
        "explanation": "Sorry, could not evaluate your answer. Marking as incorrect.",
        "source_url": source_url,
    }
