# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git commits
Do not include `Co-Authored-By: Claude` in commit messages.

## Running the app

**Backend** (from `backend/`):
```bash
source ../venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Frontend** (from `frontend/`):
```bash
python3 -m http.server 3000
```

The backend auto-creates `data/uploads/` and `data/chroma_db/` on first startup. Interactive API docs at `http://localhost:8000/docs`.

## Architecture

This is a RAG (Retrieval-Augmented Generation) document assistant. The full pipeline:

**Upload flow:** `POST /upload` → `parser.py` extracts text by page → `chunker.py` accumulates whole sentences into ~200-word windows (40-word overlap, sentence-aligned — never splits a sentence) → `embedder.py` encodes with all-MiniLM-L6-v2 (384-dim) → `retriever.py` upserts into ChromaDB.

**Query flow:** `POST /query` → embed the question → ChromaDB cosine similarity search over a wide candidate pool (`retrieval_candidate_pool=20`, distance threshold `1.1`) → `reranker.py` cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores and filters candidates (`rerank_top_n=5`, `rerank_min_score=-3.0`) → `llm.py` builds a budget-aware, sentence-safe context block and calls flan-t5-large with `min_new_tokens` enforced to avoid bare-fragment answers → return answer + citations (only for chunks actually shown to the model).

**Key design constraints:**
- flan-t5 (base or large — same tokenizer family) has a 512-token input limit. The context budget is two-tiered in `config.py`: `answer_context_max_chars` (300, per-chunk ceiling) and `answer_context_total_max_chars` (1500, **aggregate** ceiling across all chunks combined — the number that's actually sized against the token limit, leaving room for the fixed prompt template and the question). Raising chunk count alone without an aggregate cap will blow this budget and risk clipping the `Question:`/`Answer:` cue itself, since the tokenizer truncates from the right.
- The per-chunk share scales to how many chunks actually made it through reranking (`llm.py:_build_context_block`): `effective_per_chunk_max = max(answer_context_max_chars, answer_context_total_max_chars // len(chunks))`. Without this, a short document that collapses into a single chunk (common for `.txt`/`.docx`) gets truncated to just the flat per-chunk cap even though most of the aggregate budget is unused.
- Context truncation is sentence-aware (`llm.py:_sentence_truncate`) — it keeps whole sentences up to the char budget rather than cutting mid-sentence, falling back to a whitespace boundary only if no single sentence fits.
- Even a well-chunked, exactly-on-topic chunk can still be up to ~1200 chars (200 words) — if the relevant sentence sits late in that chunk, it can still be truncated away. This is a structural tension between `chunk_size` (retrieval granularity) and the answer context budget (generation-side, capped by the model's token limit), not fully solved by budget math alone.
- Bare-fragment answers (e.g. "Motorcycle riding gear" instead of a full sentence) were caused by greedy decoding stopping at the shortest valid span — `answer_min_new_tokens` (64, in `config.py`) forces the model past that point. Prompt wording alone (including a one-shot example) was tested and found to have no effect on this; `min_new_tokens` is the lever that matters. Beam search was also tested and rejected — ~2x latency with no quality gain (this was measured on flan-t5-base; re-check if this ever changes for a larger model).
- Forcing `min_new_tokens` past a natural stop has a side effect: the model sometimes emits "I don't know." and then keeps generating real content anyway (self-contradicting). `llm.py:_strip_spurious_dont_know` strips the prefix when substantial content follows it.
- `rerank_min_score=-3.0` was calibrated empirically against real query/chunk pairs from the project's own test documents, not guessed — genuinely relevant matches scored as low as -0.42 (worse for chunks that mix multiple sub-topics, diluting relevance), while genuinely irrelevant matches topped out around -10.75. If retrieval quality regresses, re-run this kind of calibration rather than guessing a new number; don't assume 0.0 is a safe cutoff (it produces false negatives).
- Chunk `page` metadata is attributed to wherever the chunk's text **starts** (`chunker.py`, `window[0][1]`), not a word-count majority across the chunk. This matters beyond just accuracy: the frontend (`frontend/app.js`) highlights a citation by taking its `excerpt` (always the first 200 chars of the chunk, see `llm.py:answer`) and doing an exact substring search for it within the PDF page named in `citation.page`. If `page` doesn't match where the excerpt text actually is, the highlight silently fails (no error) — this happened in practice when page was attributed by word-count plurality instead.
- ChromaDB rejects `None` metadata values — `retriever.py:add_chunks` converts `None` page numbers to `0`. Page `0` means "no page info" (TXT/DOCX files).
- The similarity threshold is cosine *distance* (0–2 scale), not cosine similarity. `1.1` is the current default (tightened from a `1.5` permissive default; `0.8` was tried earlier and was too strict, filtering all chunks). This threshold only gates what reaches the reranker — the reranker's own `rerank_min_score` is the actual relevance gate now.
- The `lru_cache` singletons in `dependencies.py` ensure the embedding model, reranker model, LLM, and ChromaDB client are each loaded once per process, not per request.
- The backend venv runs Python 3.9, which does not support `X | None` union syntax evaluated at runtime. Pydantic `Settings` fields must use `Optional[X]` from `typing` (pydantic resolves annotations eagerly regardless of `from __future__ import annotations`); plain functions can use `X | None` in signatures only if the file has `from __future__ import annotations` at the top (defers evaluation, never resolved since nothing calls `get_type_hints` on them).
- Chunking changes require re-running `backend/scripts/reingest.py` once against everything already in `data/uploads/` — `retriever.add_chunks` upserts by `source::chunk_index`, so if a re-chunked document produces fewer chunks than before, old trailing-index chunks are never overwritten and become stale orphans in ChromaDB.

**Settings** (`backend/config.py`): all tunables live in `Settings` (Pydantic BaseSettings). Override via `.env` at the project root. Key fields: `hf_model`, `embedding_model`, `chunk_size`, `chunk_overlap`, `retrieval_candidate_pool`, `similarity_threshold`, `reranker_model`, `rerank_top_n`, `rerank_min_score`, `answer_context_chunk_count`, `answer_context_max_chars`, `answer_context_total_max_chars`, `answer_min_new_tokens`, `answer_max_new_tokens`, `answer_num_beams`.

**Module responsibilities:**
- `services/parser.py` — file-type dispatch; returns `[{text, page}]`
- `services/chunker.py` — sentence-accumulating chunker (sentence-aligned overlap, oversized-sentence fallback for unpunctuated text); returns `[{text, metadata}]`
- `services/embedder.py` — `Embedder` class wrapping sentence-transformers
- `services/retriever.py` — `Retriever` class wrapping ChromaDB
- `services/reranker.py` — `Reranker` class wrapping a `sentence_transformers.CrossEncoder`
- `services/llm.py` — `LLMService` with `answer()` (budget-aware context building, sentence-safe truncation, spurious-"I don't know" stripping)
- `backend/dependencies.py` — FastAPI `Depends` providers via `@lru_cache`
- `backend/scripts/reingest.py` — one-off migration: re-chunks and re-embeds every file already in `data/uploads/`; run after any chunking change


## When planning changes (plan mode)

- Read this whole file for context before proposing a plan.
- Reference existing module boundaries (parser/chunker/embedder/retriever/reranker/llm/dependencies) — extend them, don't restructure unless explicitly asked.
- Flag any new config fields needed in config.py, following existing naming/pattern conventions.
- Explicitly call out anywhere the plan deviates from a decision this file documents as deliberate (e.g. thresholds, budget math, calibrated constants) and explain why.
- Note any place `reingest.py` or the `dependencies.py` singletons need updating as a result of the change.
- Don't write code in plan mode — plan only, wait for review.
