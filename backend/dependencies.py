from functools import lru_cache
from services.embedder import Embedder
from services.retriever import Retriever
from services.llm import LLMService, build_llm_service
from services.reranker import Reranker
from config import settings


@lru_cache()
def get_embedder() -> Embedder:
    return Embedder(model_name=settings.embedding_model)


@lru_cache()
def get_retriever() -> Retriever:
    return Retriever(persist_path=settings.chroma_path)


@lru_cache()
def get_llm() -> LLMService:
    return build_llm_service(settings.llm_provider)


@lru_cache()
def get_reranker() -> Reranker:
    return Reranker(model_name=settings.reranker_model)
