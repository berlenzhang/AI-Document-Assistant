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
        body.question,
        n_results=settings.retrieval_candidate_pool,
        source_filter=body.filename,
        distance_threshold=settings.similarity_threshold,
        hybrid_search_enabled=settings.hybrid_search_enabled,
        rrf_k=settings.hybrid_rrf_k,
    )
    chunks = reranker.rerank(
        body.question,
        candidates,
        top_n=settings.rerank_top_n,
        min_score=settings.rerank_min_score,
    )
    answer_text, citations_data = llm.answer(body.question, chunks)
    citations = [Citation(**c) for c in citations_data]
    return QueryResponse(answer=answer_text, citations=citations)
