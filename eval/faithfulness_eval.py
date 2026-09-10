"""Faithfulness eval: for each generated answer, check whether its claims are
actually supported by the cited chunks, using Claude as an independent judge.

Runs the same test set against BOTH generation providers (flan_t5, claude) so
the comparison between them is measured, not asserted.

Costs real API calls — run deliberately, not on every change. Requires
ANTHROPIC_API_KEY set in the project root .env (needed for the Claude
generation provider AND the judge, regardless of which provider is currently
configured as the default in .env).

Usage (from anywhere):
    python3 eval/faithfulness_eval.py
"""
import json
import os
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "backend"))

import anthropic  # noqa: E402

from config import settings  # noqa: E402
from dependencies import get_embedder, get_reranker, get_retriever  # noqa: E402
from services.llm import build_llm_service  # noqa: E402
from judge import judge_faithfulness  # noqa: E402
from report import render_markdown_table  # noqa: E402

DATASET_PATH = os.path.join(_EVAL_DIR, "dataset.json")
REPORT_PATH = os.path.join(_EVAL_DIR, "faithfulness_report.md")
PROVIDERS = ["flan_t5", "claude"]


def run() -> None:
    if not settings.anthropic_api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Add it to the project root .env "
            "(see .env.example) before running the faithfulness eval — it is "
            "required for both the Claude generation provider and the judge."
        )

    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    positive_entries = [e for e in dataset if e["type"] == "positive"]

    embedder = get_embedder()
    retriever = get_retriever()
    reranker = get_reranker()
    claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    all_rows = []
    provider_summaries = []

    for provider in PROVIDERS:
        llm = build_llm_service(provider)
        counts = {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0}
        na_count = 0

        for entry in positive_entries:
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
            chunks = reranker.rerank(
                question,
                candidates,
                top_n=settings.rerank_top_n,
                min_score=settings.rerank_min_score,
            )
            answer_text, citations = llm.answer(question, chunks)

            if not citations:
                na_count += 1
                all_rows.append([provider, entry["id"], "N/A", "no citations returned (system abstained)"])
                continue

            # Reconstruct the exact context this provider's generator saw,
            # reading the budget off this LLMService instance itself (None
            # for claude, numeric for flan_t5) — Citation.excerpt is only
            # 200 chars, shorter than either provider's real context.
            context_block, _ = llm._build_context_block(
                chunks[:llm._context_chunk_count],
                per_chunk_max_chars=llm._context_max_chars,
                total_max_chars=llm._context_total_max_chars,
            )

            verdict = judge_faithfulness(question, answer_text, context_block, settings.eval_judge_model, claude)
            counts[verdict.verdict] += 1
            all_rows.append([provider, entry["id"], verdict.verdict, verdict.explanation])

        total_graded = sum(counts.values())
        pass_rate = counts["SUPPORTED"] / total_graded if total_graded else float("nan")
        provider_summaries.append(
            f"### {provider}\n"
            f"SUPPORTED: {counts['SUPPORTED']}, PARTIALLY_SUPPORTED: {counts['PARTIALLY_SUPPORTED']}, "
            f"UNSUPPORTED: {counts['UNSUPPORTED']}, N/A (abstained): {na_count}\n"
            f"pass rate (SUPPORTED / graded): {counts['SUPPORTED']}/{total_graded} ({pass_rate:.0%})"
        )

    table = render_markdown_table(["provider", "id", "verdict", "explanation"], all_rows)
    report = (
        f"# Faithfulness Report\n\nJudge model: {settings.eval_judge_model}\n\n"
        + "\n\n".join(provider_summaries)
        + f"\n\n{table}\n"
    )

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    run()
