from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings

# Project root is one level above this file (backend/)
_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    uploads_path: str = str(_PROJECT_ROOT / "data" / "uploads")
    chroma_path: str = str(_PROJECT_ROOT / "data" / "chroma_db")
    hf_model: str = "google/flan-t5-large"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 200
    chunk_overlap: int = 40
    retrieval_candidate_pool: int = 20
    similarity_threshold: float = 1.1
    hybrid_search_enabled: bool = True
    hybrid_rrf_k: int = 60
    anthropic_api_key: Optional[str] = None
    eval_judge_model: str = "claude-opus-4-8"
    llm_provider: Literal["flan_t5", "claude"] = "flan_t5"
    llm_claude_model: str = "claude-opus-4-8"
    llm_claude_max_tokens: int = 1024
    # Default False: interactive /query is latency-sensitive single-turn
    # extractive QA, not the kind of "remotely complicated" task adaptive
    # thinking is meant for. Configurable so the eval harness can A/B it
    # without a code change.
    llm_claude_thinking_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 5
    rerank_min_score: Optional[float] = -3.0
    answer_context_chunk_count: int = 5
    answer_context_max_chars: int = 300
    answer_context_total_max_chars: int = 1500
    answer_min_new_tokens: int = 64
    answer_max_new_tokens: int = 300
    answer_num_beams: int = 1

    class Config:
        env_file = str(_PROJECT_ROOT / ".env")


settings = Settings()
