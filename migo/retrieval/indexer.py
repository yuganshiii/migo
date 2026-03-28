"""LlamaIndex + ChromaDB indexing for Wikipedia content."""

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

from migo.config import CHROMA_PERSIST_DIR, CHUNK_OVERLAP, CHUNK_SIZE, SESSION_COLLECTION, TOP_K
from migo.models.knowledge_map import RetrievedChunk
from migo.retrieval.wiki_fetcher import fetch_wikipedia_article

_embed_model = None


def _get_embed_model() -> HuggingFaceEmbedding:
    """Lazily load and cache the embedding model."""
    global _embed_model
    if _embed_model is None:
        logger.info("Loading embedding model: BAAI/bge-small-en-v1.5")
        _embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embed_model


def index_topic(topic: str) -> VectorStoreIndex:
    """Fetch a Wikipedia article for `topic`, chunk it, embed, and store in ChromaDB.

    Returns a VectorStoreIndex ready for querying.
    """
    logger.info("Indexing topic: '{}'", topic)

    # 1. Fetch Wikipedia content
    content = fetch_wikipedia_article(topic)

    # 2. Create a LlamaIndex Document
    document = Document(text=content, metadata={"topic": topic})

    # 3. Chunk with SentenceSplitter
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents([document])
    logger.info("Split into {} chunks", len(nodes))

    # 4. Embed with HuggingFaceEmbedding + store in ChromaDB
    embed_model = _get_embed_model()

    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection(SESSION_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Build the index from the chunked nodes
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    logger.success("Indexed '{}' — {} chunks stored in '{}'", topic, len(nodes), SESSION_COLLECTION)
    return index


def query_topic(
    index: VectorStoreIndex, concept: str, top_k: int = TOP_K
) -> list[RetrievedChunk]:
    """Query the index for chunks relevant to `concept`.

    Returns up to `top_k` RetrievedChunk objects. Returns an empty list
    if no results are found.
    """
    logger.info("Querying index for concept: '{}' (top_k={})", concept, top_k)

    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(concept)

    if not results:
        logger.warning("No results found for concept: '{}'", concept)
        return []

    chunks = [
        RetrievedChunk(
            text=node.text,
            source_url=f"https://en.wikipedia.org/wiki/{node.metadata.get('topic', concept).replace(' ', '_')}",
            score=node.score if node.score is not None else 0.0,
        )
        for node in results
    ]

    logger.info("Retrieved {} chunks for '{}'", len(chunks), concept)
    return chunks
