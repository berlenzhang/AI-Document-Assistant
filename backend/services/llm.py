import re
from abc import ABC, abstractmethod
from typing import Optional

import anthropic
from transformers import pipeline as hf_pipeline

from config import settings

_NO_CONTEXT_FALLBACK = (
    "I don't know. The provided document does not contain enough information to answer this question."
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class GenerationBackend(ABC):
    """Turns (question, context_block) into a final answer string.

    Implementations own their own prompt template and any provider-specific
    quirks (token forcing, output post-processing, refusal handling) — the
    shared context-building logic in LLMService never sees those details.
    """

    @abstractmethod
    def generate(self, question: str, context_block: str) -> str:
        ...


# ---------------------------------------------------------------------------
# flan-t5 backend
# ---------------------------------------------------------------------------

_FLAN_T5_ANSWER_PROMPT = """\
Answer the question in 3 to 5 complete sentences, using only the information \
in the context below. Do not answer with a single word or a sentence fragment; \
write full, explanatory sentences. If the context does not contain enough \
information to answer, respond with exactly: I don't know.

Context:
{context_block}

Question: {question}
Answer:"""

_DONT_KNOW_PREFIX_RE = re.compile(r"^i don[’']t know\.?\s*", re.IGNORECASE)


class FlanT5Backend(GenerationBackend):
    def __init__(self, model_name: str, min_new_tokens: int, max_new_tokens: int, num_beams: int):
        self._pipeline = hf_pipeline(
            "text2text-generation",
            model=model_name,
            truncation=True,
        )
        self._min_new_tokens = min_new_tokens
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams

    def generate(self, question: str, context_block: str) -> str:
        prompt = _FLAN_T5_ANSWER_PROMPT.format(context_block=context_block, question=question)
        result = self._pipeline(
            prompt,
            max_new_tokens=self._max_new_tokens,
            min_new_tokens=self._min_new_tokens,
            num_beams=self._num_beams,
            no_repeat_ngram_size=3,
        )
        return self._strip_spurious_dont_know(result[0]["generated_text"])

    def _strip_spurious_dont_know(self, text: str) -> str:
        # min_new_tokens forces generation past a natural "I don't know.",
        # so the model sometimes says it anyway and then keeps going with
        # real content. If real content follows, the prefix is spurious.
        match = _DONT_KNOW_PREFIX_RE.match(text)
        if not match:
            return text
        remainder = text[match.end():].strip()
        if len(remainder) < 20:
            return text
        return remainder[0].upper() + remainder[1:]


# ---------------------------------------------------------------------------
# Claude backend
# ---------------------------------------------------------------------------

_CLAUDE_ANSWER_PROMPT = """\
Answer the question using only the information in the context below. If the \
context does not contain enough information to answer, respond with exactly: \
I don't know.

Context:
{context_block}

Question: {question}
Answer:"""
# Deliberately does not carry over flan-t5's rigid "3 to 5 sentences"
# instruction — that constraint exists to work around flan-t5's bare-fragment
# failure mode (see CLAUDE.md), not because it's a universal quality
# requirement. Claude doesn't need it.


class ClaudeBackend(GenerationBackend):
    def __init__(self, api_key: Optional[str], model: str, max_tokens: int, thinking_enabled: bool):
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to the project root .env "
                "before setting llm_provider=claude."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._thinking_enabled = thinking_enabled

    def generate(self, question: str, context_block: str) -> str:
        prompt = _CLAUDE_ANSWER_PROMPT.format(context_block=context_block, question=question)
        kwargs = {"thinking": {"type": "adaptive"}} if self._thinking_enabled else {}
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        # Never index response.content[0] unconditionally — a refusal can mean
        # empty (pre-output) or partial (mid-stream) content.
        if response.stop_reason == "refusal":
            return _NO_CONTEXT_FALLBACK
        return next((b.text for b in response.content if b.type == "text"), _NO_CONTEXT_FALLBACK)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class LLMService:
    def __init__(
        self,
        backend: GenerationBackend,
        context_chunk_count: int,
        context_max_chars: Optional[int],
        context_total_max_chars: Optional[int],
    ):
        self._backend = backend
        self._context_chunk_count = context_chunk_count
        self._context_max_chars = context_max_chars
        self._context_total_max_chars = context_total_max_chars

    def answer(self, question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
        if not chunks:
            return (_NO_CONTEXT_FALLBACK, [])
        candidate_chunks = chunks[:self._context_chunk_count]
        context_block, used_chunks = self._build_context_block(
            candidate_chunks,
            per_chunk_max_chars=self._context_max_chars,
            total_max_chars=self._context_total_max_chars,
        )
        if not used_chunks:
            return (_NO_CONTEXT_FALLBACK, [])
        answer_text = self._backend.generate(question, context_block)
        citations = [
            {
                "source": c["metadata"]["source"],
                "page": c["metadata"].get("page"),
                "chunk_index": c["metadata"]["chunk_index"],
                "excerpt": c["text"][:200],
            }
            for c in used_chunks
        ]
        return answer_text, citations

    def _build_context_block(
        self, chunks: list[dict], per_chunk_max_chars: Optional[int], total_max_chars: Optional[int]
    ) -> tuple[str, list[dict]]:
        if total_max_chars is None:
            # No cap (Claude path): rerank_top_n and chunk_size already bound
            # worst-case input to ~6000 chars (~1500 tokens) — negligible
            # against a 1M-token context window, so there's nothing to budget.
            used_chunks = [c for c in chunks if c["text"]]
            return "\n\n".join(c["text"] for c in used_chunks), used_chunks

        pieces = []
        used_chunks = []
        remaining = total_max_chars
        # per_chunk_max_chars assumes ~context_chunk_count chunks are present.
        # When fewer actually are (e.g. a whole short document is one chunk),
        # scale the per-chunk share up so the aggregate budget isn't wasted.
        effective_per_chunk_max = max(per_chunk_max_chars, total_max_chars // len(chunks))
        for chunk in chunks:
            if remaining <= 0:
                break
            budget = min(effective_per_chunk_max, remaining)
            text = self._sentence_truncate(chunk["text"], budget)
            if not text:
                continue
            pieces.append(text)
            used_chunks.append(chunk)
            remaining -= len(text) + 2  # account for the "\n\n" separator
        return "\n\n".join(pieces), used_chunks

    def _sentence_truncate(self, text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text

        sentences = _SENTENCE_SPLIT_RE.split(text)
        kept = ""
        for sentence in sentences:
            candidate = f"{kept} {sentence}".strip() if kept else sentence
            if len(candidate) > max_chars:
                break
            kept = candidate
        if kept:
            return kept

        # No whole sentence fits (e.g. one long run-on line) — fall back to
        # the last whitespace boundary at or before the budget. If even the
        # first word doesn't fit, there's nothing usable to return.
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        return truncated[:last_space] if last_space > 0 else ""


def build_llm_service(provider: str) -> LLMService:
    if provider == "claude":
        backend = ClaudeBackend(
            api_key=settings.anthropic_api_key,
            model=settings.llm_claude_model,
            max_tokens=settings.llm_claude_max_tokens,
            thinking_enabled=settings.llm_claude_thinking_enabled,
        )
        return LLMService(
            backend=backend,
            context_chunk_count=settings.answer_context_chunk_count,
            context_max_chars=None,
            context_total_max_chars=None,
        )
    if provider == "flan_t5":
        backend = FlanT5Backend(
            model_name=settings.hf_model,
            min_new_tokens=settings.answer_min_new_tokens,
            max_new_tokens=settings.answer_max_new_tokens,
            num_beams=settings.answer_num_beams,
        )
        return LLMService(
            backend=backend,
            context_chunk_count=settings.answer_context_chunk_count,
            context_max_chars=settings.answer_context_max_chars,
            context_total_max_chars=settings.answer_context_total_max_chars,
        )
    raise ValueError(f"Unknown llm_provider: {provider!r}")
