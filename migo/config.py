import os

from dotenv import load_dotenv

load_dotenv()

# Claude
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3

# Game
DEFAULT_MAX_ROUNDS = 5

# Retrieval
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 3

# ChromaDB
CHROMA_PERSIST_DIR = "data/chromadb"
CORE_COLLECTION = "migo_core"
SESSION_COLLECTION = "migo_session"

# Knowledge map
KNOWLEDGE_MAPS_DIR = "data/knowledge_maps"
CONFIDENCE_BOOST = 0.15
CONFIDENCE_PENALTY = 0.1
FUN_ZONE_LOW = 0.4
FUN_ZONE_HIGH = 0.6
