"""Retrieval quality eval: recall@k against a hand-labeled test set.

Free, local, no API key needed. Mirrors the exact retrieval + rerank calls
`backend/routers/query.py` makes, using the real running config, so results
reflect the current hybrid-search retrieval baseline, not a hypothetical one.

Usage (from anywhere):
    python3 eval/retrieval_eval.py
"""
import json
import os
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "backend"))

from config import settings  # noqa: E402
from dependencies import get_embedder, get_reranker, get_retriever  # noqa: E402
from report import render_markdown_table  # noqa: E402

DATASET_PATH = os.path.join(_EVAL_DIR, "dataset.json")
REPORT_PATH = os.path.join(_EVAL_DIR, "retrieval_report.md")


def _chunk_key(metadata: dict) -> tuple:
    return (metadata["source"], metadata["chunk_index"])


def _resolve_expected_keys(retriever, source: str, substring: str) -> set:
    result = retriever.collection.get(where={"source": source}, include=["documents", "metadatas"])
    keys = set()
    for doc, meta in zip(result["documents"], result["metadatas"]):
        if substring in doc:
            keys.add(_chunk_key(meta))
    return keys


def run() -> None:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    embedder = get_embedder()
    retriever = get_retriever()
    reranker = get_reranker()

    rows = []
    positive_pool_hits = 0
    positive_rerank_hits = 0
    positive_count = 0
    negative_correct = 0
    negative_count = 0

    for entry in dataset:
        question = entry["question"]
        query_vector = embedder.embed_one(question)
        candidates = retriever.search(
            query_vector,
            question,
            n_results=settings.retrieval_candidate_pool,
            source_filter=None,
            distance_threshold=settings.similarity_threshold,
            hybrid_search_enabled=settings.hybrid_search_enabled,
            rrf_k=settings.hybrid_rrf_k,
        )
        reranked = reranker.rerank(
            question,
            candidates,
            top_n=settings.rerank_top_n,
            min_score=settings.rerank_min_score,
        )

        if entry["type"] == "positive":
            positive_count += 1
            expected_keys = _resolve_expected_keys(retriever, entry["source"], entry["expected_chunk_contains"])
            candidate_keys = {_chunk_key(c["metadata"]) for c in candidates}
            reranked_keys = {_chunk_key(c["metadata"]) for c in reranked}
            hit_pool = bool(expected_keys & candidate_keys)
            hit_rerank = bool(expected_keys & reranked_keys)
            positive_pool_hits += int(hit_pool)
            positive_rerank_hits += int(hit_rerank)
            rows.append([
                entry["id"],
                "positive",
                "PASS" if hit_pool else "FAIL",
                "PASS" if hit_rerank else "FAIL",
            ])
        else:
            negative_count += 1
            correctly_abstained = len(reranked) == 0
            negative_correct += int(correctly_abstained)
            rows.append([
                entry["id"],
                "negative",
                "-",
                "PASS" if correctly_abstained else "FAIL (retrieved chunks for an unanswerable question)",
            ])

    recall_at_pool = positive_pool_hits / positive_count if positive_count else float("nan")
    recall_at_rerank = positive_rerank_hits / positive_count if positive_count else float("nan")
    negative_accuracy = negative_correct / negative_count if negative_count else float("nan")

    summary = (
        f"recall@{settings.retrieval_candidate_pool} (retrieval pool): "
        f"{positive_pool_hits}/{positive_count} ({recall_at_pool:.0%})\n"
        f"recall@{settings.rerank_top_n} (post-rerank): "
        f"{positive_rerank_hits}/{positive_count} ({recall_at_rerank:.0%})\n"
        f"negative-control accuracy: {negative_correct}/{negative_count} ({negative_accuracy:.0%})"
    )
    table = render_markdown_table(
        ["id", "type", f"hit@{settings.retrieval_candidate_pool}", f"hit@{settings.rerank_top_n}"],
        rows,
    )
    report = f"# Retrieval Quality Report\n\n{summary}\n\n{table}\n"

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    run()
