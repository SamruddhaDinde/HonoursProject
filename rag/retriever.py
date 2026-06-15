"""
RAG Retriever — Medical knowledge retrieval from StatPearls index.

Provides a simple retrieve() function that takes a clinical case text
and returns the top-k most relevant medical textbook passages.

"""

import os
from pathlib import Path
from typing import Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


#  Configuration 

INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "/workspace/rag_index"))
COLLECTION_NAME = "statpearls_medical"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434").replace("/v1", "")

DEFAULT_TOP_K = 3
MAX_PASSAGE_LENGTH = 500  # characters per passage, to keep prompts manageable


class MedicalRetriever:
    """Retrieves relevant medical knowledge from StatPearls for RAG.

    Loads the pre-built ChromaDB index from disk. The index must be
    built first using build_rag_index.py.

    The retriever is designed to be instantiated once and reused across
    all cases in a run — the index stays in memory.
    """

    def __init__(self, index_dir: Optional[Path] = None, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        index_path = index_dir or INDEX_DIR

        if not (index_path / "chroma.sqlite3").exists():
            raise FileNotFoundError(
                f"RAG index not found at {index_path}. "
                "Run build_rag_index.py first."
            )

        # Set up embedding model (same as used during indexing)
        embed_model = OllamaEmbedding(
            model_name=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        Settings.embed_model = embed_model

        # Load ChromaDB index from disk
        chroma_client = chromadb.PersistentClient(path=str(index_path))
        chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        self.index = VectorStoreIndex.from_vector_store(vector_store)
        self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)

        print(f"  RAG retriever loaded from {index_path} "
              f"(top_k={self.top_k})")

    def retrieve(self, query_text: str, top_k: Optional[int] = None) -> list[dict]:
        """Retrieve relevant medical passages for a clinical case.

        Args:
            query_text: The clinical case text to search for.
            top_k: Override the default number of passages to retrieve.

        Returns:
            List of dicts with keys:
                text: the passage content (truncated to MAX_PASSAGE_LENGTH)
                title: source article title
                score: similarity score
        """
        k = top_k or self.top_k

        if k != self.top_k:
            retriever = self.index.as_retriever(similarity_top_k=k)
        else:
            retriever = self.retriever

        try:
            results = retriever.retrieve(query_text)
        except Exception as e:
            print(f"  RAG retrieval failed: {e}")
            return []

        passages = []
        for node in results:
            text = node.text.strip()
            if len(text) > MAX_PASSAGE_LENGTH:
                # Truncate at sentence boundary if possible
                cutoff = text[:MAX_PASSAGE_LENGTH].rfind(".")
                if cutoff > MAX_PASSAGE_LENGTH // 2:
                    text = text[:cutoff + 1]
                else:
                    text = text[:MAX_PASSAGE_LENGTH] + "..."

            passages.append({
                "text": text,
                "title": node.metadata.get("title", "Unknown"),
                "score": round(node.score, 4) if node.score else 0.0,
            })

        return passages

    def format_context(self, passages: list[dict]) -> str:
        """Format retrieved passages into a string for prompt injection.

        Returns a clean text block ready to insert into the agent prompt.
        Returns empty string if no passages retrieved.
        """
        if not passages:
            return ""

        lines = []
        for i, p in enumerate(passages, 1):
            lines.append(f"[{i}] {p['title']}")
            lines.append(f"    {p['text']}")

        return "\n".join(lines)