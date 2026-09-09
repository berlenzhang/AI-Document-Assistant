from __future__ import annotations


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int,
        min_score: float | None = None,
    ) -> list[dict]:
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        scored = [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in zip(chunks, scores)
        ]
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)

        if min_score is not None:
            scored = [c for c in scored if c["rerank_score"] >= min_score]

        return scored[:top_n]
