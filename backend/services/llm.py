import re

from transformers import pipeline as hf_pipeline

ANSWER_PROMPT = """\
Answer the question in 3 to 5 complete sentences, using only the information \
in the context below. Do not answer with a single word or a sentence fragment; \
write full, explanatory sentences. If the context does not contain enough \
information to answer, respond with exactly: I don't know.

Context:
{context_block}

Question: {question}
Answer:"""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DONT_KNOW_PREFIX_RE = re.compile(r"^i don[’']t know\.?\s*", re.IGNORECASE)


class LLMService:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self._pipeline = hf_pipeline(
            "text2text-generation",
            model=model_name,
            truncation=True,
        )

    def answer(
        self,
        question: str,
        chunks: list[dict],
        context_chunk_count: int,
        context_max_chars: int,
        context_total_max_chars: int,
        min_new_tokens: int,
        max_new_tokens: int,
        num_beams: int,
    ) -> tuple[str, list[dict]]:
        if not chunks:
            return (
                "I don't know. The provided document does not contain enough information to answer this question.",
                [],
            )
        candidate_chunks = chunks[:context_chunk_count]
        context_block, used_chunks = self._build_context_block(
            candidate_chunks,
            per_chunk_max_chars=context_max_chars,
            total_max_chars=context_total_max_chars,
        )
        if not used_chunks:
            return (
                "I don't know. The provided document does not contain enough information to answer this question.",
                [],
            )
        prompt = ANSWER_PROMPT.format(context_block=context_block, question=question)
        result = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
        )
        answer_text = self._strip_spurious_dont_know(result[0]["generated_text"])
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

    def _build_context_block(
        self, chunks: list[dict], per_chunk_max_chars: int, total_max_chars: int
    ) -> tuple[str, list[dict]]:
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
