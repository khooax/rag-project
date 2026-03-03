"""
Tests how chunk size effect on retrieval quality,
builds separate vector DBs for each chunk size from the raw documents, stored in existing Chroma DB, then measures precision
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Suppress Chroma telemetry before imports
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"]     = "false"

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR      = "./db"
CHUNK_SIZES = [256, 512, 1024]
TOP_K       = 5

RETRIEVAL_TEST_SET = [
    {
        "question":    "What is the minimum notice period if I resign?",
        "must_contain": ["notice", "26 weeks", "resign"],
    },
    {
        "question":    "How many days of annual leave am I entitled to?",
        "must_contain": ["annual leave", "7 days", "year"],
    },
    {
        "question":    "What is the overtime pay rate?",
        "must_contain": ["overtime", "1.5", "rate"],
    },
    {
        "question":    "How many days of sick leave do I get?",
        "must_contain": ["sick leave", "14 days", "outpatient"],
    },
    {
        "question":    "What are the CPF contribution rates?",
        "must_contain": ["cpf", "%", "employer"],
    },
    {
        "question":    "What is the maternity leave entitlement?",
        "must_contain": ["maternity", "16 weeks", "citizen"],
    },
    {
        "question":    "Can my employer deduct my salary?",
        "must_contain": ["deduct", "salary", "unauthorised"],
    },
    {
        "question":    "What is the maximum overtime hours per month?",
        "must_contain": ["72 hours", "overtime", "month"],
    },
    {
        "question":    "What is the Employment Pass minimum salary?",
        "must_contain": ["employment pass", "5,000", "salary"],
    },
    {
        "question":    "What happens if I am wrongfully dismissed?",
        "must_contain": ["wrongful", "dismissal", "claim"],
    },
]


def load_source_docs_from_chroma() -> list[Document]:
    """
    Extract the original (un-chunked or previously chunked) documents
    from existing Chroma DB, treat each stored chunk as a source
    document — re-chunking them at different sizes 
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    # get() returns all stored documents without a query
    raw = db.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    return docs


def precision_at_k(retrieved: list, must_contain: list[str], k: int = 5) -> float:
    """Fraction of top-k chunks containing at least one required keyword."""
    chunks = retrieved[:k]
    relevant = sum(
        1 for c in chunks
        if any(kw.lower() in c.page_content.lower() for kw in must_contain)
    )
    return relevant / len(chunks) if chunks else 0.0


def build_temp_db(
    source_docs: list[Document],
    chunk_size: int,
    embeddings: HuggingFaceEmbeddings,
) -> tuple[Chroma, int, str]:
    """Chunk source_docs at chunk_size and build a fresh temp Chroma DB."""
    overlap  = max(20, chunk_size // 10)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(source_docs)
    db_dir = f"./ablations/tmp_db_chunk_{chunk_size}"

    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
    )
    return db, len(chunks), db_dir


def run():
    print("=" * 60)
    print("Ablation: Chunk Size vs Retrieval Precision@5")

    if not os.path.exists(DB_DIR):
        print(f"ERROR: Vector DB not found at '{DB_DIR}'. Run ingest.py first.")
        sys.exit(1)

    source_docs = load_source_docs_from_chroma()
    print(f"  Loaded {len(source_docs)} stored passages as source material\n")

    if len(source_docs) == 0:
        print("ERROR: No documents found in DB. Check your ingest step.")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    results = {}

    for chunk_size in CHUNK_SIZES:
        print(f"\n── Chunk size: {chunk_size} (overlap={max(20, chunk_size // 10)}) ──")

        db, n_chunks, db_dir = build_temp_db(source_docs, chunk_size, embeddings)
        print(f"  Built DB with {n_chunks} chunks")

        retriever  = db.as_retriever(search_kwargs={"k": TOP_K})
        precisions = []
        per_q      = []

        for item in RETRIEVAL_TEST_SET:
            retrieved = retriever.invoke(item["question"])
            p         = precision_at_k(retrieved, item["must_contain"], TOP_K)
            precisions.append(p)
            per_q.append({"question": item["question"], "precision": round(p, 2)})

        avg = sum(precisions) / len(precisions)
        results[chunk_size] = {
            "avg_precision_at_5": round(avg, 3),
            "num_chunks":         n_chunks,
            "per_question":       per_q,
        }
        print(f"  Avg Precision@5: {avg:.3f}")

        # Clean up temp DB to save disk space
        shutil.rmtree(db_dir)

    # Results 
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Chunk Size':<12} {'# Chunks':<12} {'Precision@5':<14} Note")
    print("-" * 55)

    best_cs    = max(results, key=lambda k: results[k]["avg_precision_at_5"])
    best_score = results[best_cs]["avg_precision_at_5"]

    for cs in CHUNK_SIZES:
        r      = results[cs]
        gap    = best_score - r["avg_precision_at_5"]
        marker = " ← chosen" if cs == best_cs else (f" (-{gap:.3f})" if gap > 0 else "")
        print(f"{cs:<12} {r['num_chunks']:<12} {r['avg_precision_at_5']:<14.3f}{marker}")

    # Check for suspicious ties
    scores = [results[cs]["avg_precision_at_5"] for cs in CHUNK_SIZES]
    chunks = [results[cs]["num_chunks"]          for cs in CHUNK_SIZES]
    if len(set(chunks)) < len(CHUNK_SIZES):
        print("\n⚠️  WARNING: Two or more chunk sizes produced the same # chunks.")
        print("   This likely means source_docs were too short to split differently.")
        print("   Consider pointing load_source_docs_from_chroma() at your raw files.")

    # Per-question breakdown
    print("\ Breakdown ────────────────────────────────")
    print(f"{'Question':<52} " + "  ".join(f"k={cs}" for cs in CHUNK_SIZES))
    print("-" * 70)
    for i, item in enumerate(RETRIEVAL_TEST_SET):
        q = item["question"][:50] + ".."
        scores_row = "   ".join(
            f"{results[cs]['per_question'][i]['precision']:.2f}" for cs in CHUNK_SIZES
        )
        print(f"{q:<52} {scores_row}")

    # Save
    Path("ablations").mkdir(exist_ok=True)
    with open("ablations/results_chunk_size.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved to ablations/results_chunk_size.json")


if __name__ == "__main__":
    run()
