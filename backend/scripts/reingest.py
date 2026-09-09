"""One-off migration: re-chunk and re-embed every already-uploaded document.

Run once after a chunking algorithm change, so ChromaDB doesn't end up with
stale chunks left over at indices the new chunker no longer produces.

Usage (from anywhere):
    python3 backend/scripts/reingest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from dependencies import get_embedder, get_retriever
from services.chunker import chunk_document
from services.parser import parse_document


def reingest_all() -> None:
    embedder = get_embedder()
    retriever = get_retriever()

    filenames = sorted(os.listdir(settings.uploads_path))
    if not filenames:
        print("No files found in", settings.uploads_path)
        return

    for filename in filenames:
        filepath = os.path.join(settings.uploads_path, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            deleted = retriever.delete_source(filename)
        except ValueError:
            deleted = 0

        pages = parse_document(filepath)
        if not pages:
            print(f"[skip] {filename}: no extractable text")
            continue

        chunks = chunk_document(
            pages,
            filename,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        if not chunks:
            print(f"[skip] {filename}: produced no indexable chunks")
            continue

        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed(texts)
        retriever.add_chunks(chunks, embeddings)

        print(f"[ok] {filename}: {deleted} old chunks -> {len(chunks)} new chunks")


if __name__ == "__main__":
    reingest_all()
