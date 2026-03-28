# Drift — Adaptive Trivia Engine

## Project Overview
Drift is an adaptive general knowledge trivia app that builds a knowledge map of each user, then generates questions targeting their ~50% confidence zone — the sweet spot where trivia is actually fun. Questions are grounded in Wikipedia content via RAG.

## Tech Stack
- **Agent orchestration:** LangGraph (stateful graph with typed state)
- **RAG / retrieval:** LlamaIndex with ChromaDB as vector store
- **LLM backbone:** Claude Sonnet via Anthropic API (`claude-sonnet-4-20250514`)
- **UI:** Streamlit
- **Language:** Python 3.11+

## Architecture

### LangGraph State Machine
The game loop is a single LangGraph `StateGraph` with these nodes:

1. **router** — Reads user's category choice + knowledge map. Decides which topic/concept to quiz next. Targets concepts with confidence near 0.4–0.6 (the "fun zone"). Falls back to unseen concepts (confidence 0.0) if no edge-zone concepts exist.
2. **question_generator** — Uses Claude + retrieved Wikipedia context to craft one multiple-choice or short-answer question. Always grounds the question in a specific retrieved passage.
3. **evaluator** — Takes the user's answer, grades it (correct/incorrect/partial), generates a short explanation, and includes the Wikipedia source URL.
4. **knowledge_map_updater** — Adjusts confidence scores for the tested concept. On correct: confidence += 0.15 (capped at 1.0). On incorrect: confidence -= 0.1 (floored at 0.0). Tracks attempt count and streak.
5. **difficulty_calibrator** — After each answer, decides: go broader (test sibling concept), go deeper (generate sub-concepts for this topic), or stay at current level. This controls the granularity — see below.

Edges: `router → question_generator → evaluator → knowledge_map_updater → difficulty_calibrator → router` (loop). Add a conditional edge from `router` that exits to `END` when the round is over (default: 5 questions per round).

### Granularity — Approach 3 (Hybrid: Start Flat, Deepen on Demand)
- Start with broad concepts only (e.g., "World War 2", "Space Exploration").
- When user gets a broad concept RIGHT, test a sibling broad concept next.
- When user gets a concept WRONG, the difficulty_calibrator generates 2-3 sub-concepts on the fly (e.g., "World War 2" → "Eastern Front", "D-Day", "Pacific Theater") and adds them to the knowledge map.
- Sub-concepts can themselves be deepened further on subsequent wrong answers.
- The knowledge map is a dynamic tree that grows as the user plays.

### Knowledge Map Schema
```python
{
    "user_id": str,
    "categories": {
        "general_knowledge": {
            "world_war_2": {"confidence": 0.7, "attempts": 5, "streak": 2, "last_tested": "2026-03-28T..."},
            "world_war_2/eastern_front": {"confidence": 0.2, "attempts": 1, "streak": 0, "last_tested": "..."},
            "space_exploration": {"confidence": 0.0, "attempts": 0, "streak": 0, "last_tested": null}
        }
    },
    "overall_score": int,
    "round_history": [{"round": 1, "score": 3, "total": 5, "timestamp": "..."}]
}
```
Store as JSON file locally at `data/knowledge_maps/{user_id}.json`.

### LlamaIndex + ChromaDB Retrieval Layer
- **Wikipedia retrieval:** Use `llama_index.readers.web.SimpleWebPageReader` or the `wikipedia` Python package to fetch article content.
- **Indexing:** Chunk Wikipedia articles (chunk_size=512, overlap=50) and store in ChromaDB via LlamaIndex's `ChromaVectorStore`.
- **Hybrid strategy:**
  - Pre-index a small seed corpus at setup (~20 popular Wikipedia articles across common GK topics: history, geography, science, sports, entertainment). Store in a persistent ChromaDB collection called `drift_core`.
  - For any topic outside the core set, fetch the Wikipedia article on-demand, index it into a session-scoped ChromaDB collection called `drift_session`, and generate questions from it.
- **Query:** When generating a question, retrieve top-3 chunks for the target concept. Pass them as context to Claude.

## Build Order (Incremental — V0 First)

### V0 — Minimal Playable Loop (Build This First)
One loop: user picks a topic → system fetches Wikipedia → generates one question from retrieved context → user answers → evaluator grades + explains with source → next question adapts based on a simple in-memory knowledge map → repeat for 5 questions → show round score.

No persistent storage, no leaderboard, no pre-indexed corpus, no difficulty calibration. Just:
- LangGraph graph with: router → question_generator → evaluator → knowledge_map_updater → router
- LlamaIndex fetching ONE Wikipedia article per topic, indexing into ChromaDB
- Claude generating and evaluating questions
- Streamlit UI: topic input, question display, answer input, result + explanation + source link, round score

### V1 — Persistence
- Save knowledge maps to JSON files
- Knowledge map loads on session start, so it remembers the user across sessions

### V2 — Pre-indexed Core Corpus
- Seed ChromaDB with ~20 Wikipedia articles at setup time
- Faster start for common topics

### V3 — Difficulty Calibrator + Granularity
- Add the difficulty_calibrator node
- Implement the "start flat, deepen on demand" logic
- Dynamic sub-concept generation on wrong answers

### V4 — Leaderboard + Scoring
- Score formula: base_points × difficulty_multiplier per question
- Leaderboard stored in `data/leaderboard.json`
- Display in Streamlit sidebar

### V5 — Polish
- Multiple categories
- User profiles
- Better UI (progress bars, streaks, animations)

## Directory Structure
```
drift/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── app.py                          # Streamlit entrypoint
├── drift/
│   ├── __init__.py
│   ├── graph.py                    # LangGraph state machine definition
│   ├── state.py                    # TypedDict for graph state
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── router.py               # Topic/concept selection logic
│   │   ├── question_generator.py   # Claude-powered question generation
│   │   ├── evaluator.py            # Answer grading + explanation
│   │   ├── knowledge_map_updater.py
│   │   └── difficulty_calibrator.py # V3+
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── wiki_fetcher.py         # Wikipedia article fetching
│   │   └── indexer.py              # LlamaIndex + ChromaDB indexing
│   ├── models/
│   │   ├── __init__.py
│   │   └── knowledge_map.py        # Knowledge map data model + persistence
│   └── prompts/
│       ├── question_gen.py          # Prompt templates for question generation
│       └── evaluation.py            # Prompt templates for answer evaluation
├── data/
│   ├── knowledge_maps/              # Per-user JSON files (V1+)
│   └── leaderboard.json             # Global leaderboard (V4+)
└── tests/
    └── ...
```

## Coding Conventions
- Use type hints everywhere. Use `TypedDict` for LangGraph state.
- Use `pydantic` for data models (knowledge map, question, round result).
- Keep each LangGraph node in its own file under `drift/nodes/`.
- Prompt templates go in `drift/prompts/` as Python string constants — not inline.
- Use `anthropic` Python SDK directly for Claude calls (not LangChain's ChatAnthropic wrapper).
- Handle API errors with retries (max 3 attempts with exponential backoff).
- Use `loguru` for logging.
- Format with `ruff`. Lint with `ruff check`.

## Key Constraints
- Do NOT over-engineer. Build V0 first, get the loop working end-to-end, then layer features.
- Each node should be testable in isolation — accept state dict in, return state dict out.
- Keep Streamlit UI minimal in V0: text input, markdown display, st.button. No custom components.
- Wikipedia fetching should be wrapped in a try/except — if an article doesn't exist, ask the user to pick another topic.
- ChromaDB should use persistent storage at `data/chromadb/` so indexes survive restarts.

## Environment Variables
```
ANTHROPIC_API_KEY=<key>
```
No other secrets needed. Everything else is local.

## Commands
- Install: `pip install -r requirements.txt`
- Run: `streamlit run app.py`
- Lint: `ruff check .`
- Format: `ruff format .`