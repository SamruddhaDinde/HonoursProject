"""
RAG Index Builder — StatPearls Medical Knowledge Base (Local JSONL)

Reads pre-chunked StatPearls snippets from local JSONL files (produced by
MedRAG's statpearls.py chunking script), embeds them using Ollama's
nomic-embed-text, and stores in ChromaDB for retrieval.

Expected input: /workspace/medrag_tools/corpus/statpearls/chunk/*.jsonl
Expected output: /workspace/rag_index/ (ChromaDB persistent storage)

Run this ONCE. The index persists on disk.

Usage:
    python build_rag_index.py
"""

import os
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────

CHUNK_DIR = Path(os.getenv(
    "STATPEARLS_CHUNK_DIR",
    "/workspace/medrag_tools/corpus/statpearls/chunk"
))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "/workspace/rag_index"))
COLLECTION_NAME = "statpearls_medical"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434").replace("/v1", "")
BATCH_SIZE = 50  # documents per batch for embedding


def main():
    print("=" * 60)
    print("RAG Index Builder — StatPearls (Local JSONL)")
    print("=" * 60)

    # Check if index already exists
    if (INDEX_DIR / "chroma.sqlite3").exists():
        print(f"\nIndex already exists at {INDEX_DIR}")
        response = input("Rebuild? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping rebuild.")
            return
        import shutil
        shutil.rmtree(INDEX_DIR)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load pre-chunked snippets from JSONL files ───────────

    print(f"\n[1/3] Loading snippets from {CHUNK_DIR}...")

    if not CHUNK_DIR.exists():
        print(f"ERROR: Chunk directory not found at {CHUNK_DIR}")
        print("Run the MedRAG chunking script first:")
        print("  cd medrag_tools && python src/data/statpearls.py")
        sys.exit(1)

    jsonl_files = sorted(CHUNK_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {CHUNK_DIR}")
        sys.exit(1)

    print(f"  Found {len(jsonl_files)} JSONL files")

    rows = []
    for jf in jsonl_files:
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    content = row.get("content", "")
                    if len(content.strip()) >= 30:
                        rows.append(row)
                except json.JSONDecodeError:
                    continue

    print(f"  Loaded {len(rows)} snippets (skipped empty/short)")

    # ── Step 2: Set up embedding model and vector store ──────────────

    print("\n[2/3] Setting up embedding model and vector store...")

    from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.embed_model = embed_model

    # IMPORTANT: Disable LlamaIndex's built-in chunking.
    # StatPearls is ALREADY chunked at ~119 tokens per snippet.
    # We want one embedding per snippet, not sub-chunks of snippets.
    Settings.chunk_size = 2048    # large enough that no snippet gets split
    Settings.chunk_overlap = 0

    chroma_client = chromadb.PersistentClient(path=str(INDEX_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── Step 3: Build index in batches ───────────────────────────────
    # ── Step 3: Build index ──────────────────────────────────────────

    print(f"\n[3/3] Building index ({len(rows)} snippets)...")
    print("  Converting to documents...")

    # Subsample if too large
    # import random
    # random.seed(42)
    # MAX_SNIPPETS = 50000
    # if len(rows) > MAX_SNIPPETS:
    #     print(f"  Subsampling from {len(rows)} to {MAX_SNIPPETS} for practical indexing time...")
    #     random.shuffle(rows)
    #     rows = rows[:MAX_SNIPPETS]

    MAX_CHARS = 10000  # ~500 tokens, safely under nomic-embed-text's 8192 limit

    all_docs = [
        Document(
            text=row["content"][:MAX_CHARS],
            metadata={
                "title": row.get("title", "Unknown"),
                "snippet_id": row.get("id", ""),
                "source": "StatPearls/NCBI",
            },
        )
        for row in rows
    ]

    print(f"  Embedding and indexing {len(all_docs)} documents...")
    print("  This takes 1-2 hours for 50K snippets...")

    start = time.time()

    index = VectorStoreIndex.from_documents(
        all_docs,
        storage_context=storage_context,
        show_progress=True,
    )

    elapsed = time.time() - start
    print(f"\n  Index built in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    # ── Sanity check ─────────────────────────────────────────────────

    print("\n  Running sanity checks...")

    queries = [
        "renal transplant complications immunosuppression",
        "chest x-ray pneumonia findings",
        "skin rash differential diagnosis",
    ]

    retriever = index.as_retriever(similarity_top_k=3)

    for query in queries:
        print(f"\n  Query: '{query}'")
        results = retriever.retrieve(query)
        for i, node in enumerate(results):
            title = node.metadata.get("title", "Unknown")
            score = f"{node.score:.4f}" if node.score else "N/A"
            print(f"    [{i+1}] {title} (score: {score})")
            print(f"        {node.text[:100]}...")

    print(f"\n{'='*60}")
    print("Index build complete. Ready for RAG experiments.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()