from fastapi import APIRouter, Depends

from config import settings
from dependencies import get_embedder, get_llm, get_reranker, get_retriever
from models.schemas import Citation, QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_document(
    body: QueryRequest,
    embedder=Depends(get_embedder),
    retriever=Depends(get_retriever),
    reranker=Depends(get_reranker),
    llm=Depends(get_llm),
):
    query_vector = embedder.embed_one(body.question)
    candidates = retriever.search(
        query_vector,
        n_results=settings.retrieval_candidate_pool,
        source_filter=body.filename,
        distance_threshold=settings.similarity_threshold,
    )
    chunks = reranker.rerank(
        body.question,
        candidates,
        top_n=settings.rerank_top_n,
        min_score=settings.rerank_min_score,
    )
    answer_text, citations_data = llm.answer(
        body.question,
        chunks,
        context_chunk_count=settings.answer_context_chunk_count,
        context_max_chars=settings.answer_context_max_chars,
        context_total_max_chars=settings.answer_context_total_max_chars,
        min_new_tokens=settings.answer_min_new_tokens,
        max_new_tokens=settings.answer_max_new_tokens,
        num_beams=settings.answer_num_beams,
    )
    citations = [Citation(**c) for c in citations_data]
    return QueryResponse(answer=answer_text, citations=citations)
