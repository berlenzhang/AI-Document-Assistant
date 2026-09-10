from __future__ import annotations

import re

import chromadb
from rank_bm25 import BM25Okapi

_BM25_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _BM25_TOKEN_RE.findall(text.lower())


class Retriever:
    def __init__(self, persist_path: str):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_index: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_chunks: list[dict] = []
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        # BM25Okapi([]) raises ZeroDivisionError, so a fresh/empty install
        # must skip index construction entirely rather than build on empty data.
        if self.collection.count() == 0:
            self._bm25_index = None
            self._bm25_ids = []
            self._bm25_chunks = []
            return
        result = self.collection.get(include=["documents", "metadatas"])
        ids, docs, metas = result["ids"], result["documents"], result["metadatas"]
        tokenized = [_tokenize(d) for d in docs]
        new_index = BM25Okapi(tokenized)  # build fully before assigning
        self._bm25_index = new_index
        self._bm25_ids = ids
        self._bm25_chunks = [{"text": d, "metadata": m} for d, m in zip(docs, metas)]

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        ids = [f"{c['metadata']['source']}::{c['metadata']['chunk_index']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        # ChromaDB rejects None; replace with 0 for missing page numbers
        metadatas = [
            {k: (v if v is not None else 0) for k, v in c["metadata"].items()}
            for c in chunks
        ]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        self._rebuild_bm25_index()

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        n_results: int = 5,
        source_filter: str | None = None,
        distance_threshold: float = 1.5,
        hybrid_search_enabled: bool = True,
        rrf_k: int = 60,
    ) -> list[dict]:
        where = {"source": source_filter} if source_filter else None
        # Clamp n_results to collection size to avoid ChromaDB errors
        count = self.collection.count()
        if count == 0:
            return []
        actual_n = min(n_results, count)
        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)

        vector_ranked_ids: list[str] = []
        chunk_by_id: dict[str, dict] = {}
        for cid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist <= distance_threshold:
                vector_ranked_ids.append(cid)
                chunk_by_id[cid] = {"text": doc, "metadata": meta, "distance": dist}

        if not hybrid_search_enabled:
            return [chunk_by_id[cid] for cid in vector_ranked_ids]

        bm25_ranked_ids: list[str] = []
        if self._bm25_index is not None:
            tokens = _tokenize(query_text)
            if tokens:
                scores = self._bm25_index.get_scores(tokens)
                ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                for i in ranked:
                    if scores[i] <= 0:
                        continue
                    chunk = self._bm25_chunks[i]
                    if source_filter and chunk["metadata"].get("source") != source_filter:
                        continue
                    cid = self._bm25_ids[i]
                    bm25_ranked_ids.append(cid)
                    chunk_by_id.setdefault(cid, {**chunk, "distance": None})
                    if len(bm25_ranked_ids) >= n_results:
                        break

        rrf_scores: dict[str, float] = {}
        for rank, cid in enumerate(vector_ranked_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, cid in enumerate(bm25_ranked_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

        fused_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:n_results]
        return [{**chunk_by_id[cid], "rrf_score": rrf_scores[cid]} for cid in fused_ids]

    def list_sources(self) -> list[dict]:
        result = self.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in result["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"filename": src, "chunk_count": cnt} for src, cnt in counts.items()]

    def delete_source(self, filename: str) -> int:
        result = self.collection.get(where={"source": filename}, include=[])
        ids = result["ids"]
        if not ids:
            raise ValueError(f"No chunks found for source: {filename}")
        self.collection.delete(ids=ids)
        self._rebuild_bm25_index()
        return len(ids)
