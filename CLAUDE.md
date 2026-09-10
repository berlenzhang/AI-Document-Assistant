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

**Query flow:** `POST /query` → embed the question → ChromaDB cosine similarity search over a wide candidate pool (`retrieval_candidate_pool=20`, distance threshold `1.1`) → `reranker.py` cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores and filters candidates (`rerank_top_n=5`, `rerank_min_score=-3.0`) → `llm.py` builds a sentence-safe context block (budget-aware for flan-t5, uncapped for Claude — see below) and delegates generation to whichever `GenerationBackend` `llm_provider` selects (`flan_t5` default, or `claude`) → return answer + citations (only for chunks actually shown to the model).

**Key design constraints:**
- **Generation is pluggable** (`llm.py`): `GenerationBackend` is an ABC with one method, `generate(question, context_block) -> str`; `FlanT5Backend` and `ClaudeBackend` implement it, `build_llm_service(provider)` constructs the right one plus the context budget that goes with it, and `config.py`'s `llm_provider` (`"flan_t5"` default or `"claude"`) selects between them. Each backend owns its own prompt template and quirks — nothing provider-specific leaks into the shared `LLMService.answer()`/`_build_context_block()`/`_sentence_truncate()` logic.
- **flan-t5 path**: flan-t5 (base or large — same tokenizer family) has a 512-token input limit. The context budget is two-tiered in `config.py`: `answer_context_max_chars` (300, per-chunk ceiling) and `answer_context_total_max_chars` (1500, **aggregate** ceiling across all chunks combined — the number that's actually sized against the token limit, leaving room for the fixed prompt template and the question). Raising chunk count alone without an aggregate cap will blow this budget and risk clipping the `Question:`/`Answer:` cue itself, since the tokenizer truncates from the right.
- The per-chunk share scales to how many chunks actually made it through reranking (`llm.py:LLMService._build_context_block`): `effective_per_chunk_max = max(answer_context_max_chars, answer_context_total_max_chars // len(chunks))`. Without this, a short document that collapses into a single chunk (common for `.txt`/`.docx`) gets truncated to just the flat per-chunk cap even though most of the aggregate budget is unused.
- Context truncation is sentence-aware (`llm.py:LLMService._sentence_truncate`) — it keeps whole sentences up to the char budget rather than cutting mid-sentence, falling back to a whitespace boundary only if no single sentence fits. Shared by both providers, but only ever invoked with a real budget on the flan-t5 path (see below).
- Even a well-chunked, exactly-on-topic chunk can still be up to ~1200 chars (200 words) — if the relevant sentence sits late in that chunk, it can still be truncated away on the flan-t5 path. This is a structural tension between `chunk_size` (retrieval granularity) and the flan-t5 answer context budget (generation-side, capped by the model's token limit), not fully solved by budget math alone.
- **Claude path bypasses the char budget entirely** (`build_llm_service` passes `context_max_chars=None, context_total_max_chars=None`) rather than using a larger-but-still-arbitrary number. `_build_context_block` treats `total_max_chars is None` as a dedicated bypass branch (join full chunk text, skip the budget-accounting loop) — the alternative naive path would crash with `TypeError` on `total_max_chars // len(chunks)` the instant either budget argument is `None`. The bypass is justified with real numbers, not just "Claude has a big context window": `rerank_top_n=5` bounds chunk count and `chunk_size=200` words bounds each chunk's raw size by construction, so worst case is ~5 × 200 words × ~6 chars/word ≈ 6000 chars ≈ ~1500 tokens — negligible against a 1M-token context window even accounting for Opus's tokenizer.
- Bare-fragment flan-t5 answers (e.g. "Motorcycle riding gear" instead of a full sentence) were caused by greedy decoding stopping at the shortest valid span — `answer_min_new_tokens` (64, in `config.py`, passed only to `FlanT5Backend`) forces the model past that point. Prompt wording alone (including a one-shot example) was tested and found to have no effect on this; `min_new_tokens` is the lever that matters. Beam search was also tested and rejected — ~2x latency with no quality gain (measured on flan-t5-base). Claude doesn't have this failure mode, so `ClaudeBackend` has no equivalent forcing.
- Forcing `min_new_tokens` past a natural stop has a flan-t5-specific side effect: the model sometimes emits "I don't know." and then keeps generating real content anyway (self-contradicting). `FlanT5Backend._strip_spurious_dont_know` strips the prefix when substantial content follows it — scoped to that backend, not shared.
- `llm_claude_thinking_enabled` defaults to `False` even though adaptive thinking is generally the recommended default for "remotely complicated" tasks — grounded single-turn extractive QA over ≤6 reranked chunks is plain Q&A, not agentic reasoning, and `/query` is latency-sensitive. Left as a config toggle so the eval harness can A/B it without a code change.
- `ClaudeBackend` raises a plain `RuntimeError` (not `SystemExit`) when `anthropic_api_key` is missing, since construction happens during FastAPI dependency resolution — `SystemExit` is a `BaseException` and wouldn't be caught cleanly there, unlike in the standalone `eval/` scripts where `SystemExit` is the right choice.
- `rerank_min_score=-3.0` was calibrated empirically against real query/chunk pairs from the project's own test documents, not guessed — genuinely relevant matches scored as low as -0.42 (worse for chunks that mix multiple sub-topics, diluting relevance), while genuinely irrelevant matches topped out around -10.75. If retrieval quality regresses, re-run this kind of calibration rather than guessing a new number; don't assume 0.0 is a safe cutoff (it produces false negatives).
- Chunk `page` metadata is attributed to wherever the chunk's text **starts** (`chunker.py`, `window[0][1]`), not a word-count majority across the chunk. This matters beyond just accuracy: the frontend (`frontend/app.js`) highlights a citation by taking its `excerpt` (always the first 200 chars of the chunk, see `llm.py:answer`) and doing an exact substring search for it within the PDF page named in `citation.page`. If `page` doesn't match where the excerpt text actually is, the highlight silently fails (no error) — this happened in practice when page was attributed by word-count plurality instead.
- ChromaDB rejects `None` metadata values — `retriever.py:add_chunks` converts `None` page numbers to `0`. Page `0` means "no page info" (TXT/DOCX files).
- The similarity threshold is cosine *distance* (0–2 scale), not cosine similarity. `1.1` is the current default (tightened from a `1.5` permissive default; `0.8` was tried earlier and was too strict, filtering all chunks). This threshold only gates what reaches the reranker — the reranker's own `rerank_min_score` is the actual relevance gate now.
- The `lru_cache` singletons in `dependencies.py` ensure the embedding model, reranker model, LLM, and ChromaDB client are each loaded once per process, not per request.
- The backend venv runs Python 3.9, which does not support `X | None` union syntax evaluated at runtime. Pydantic `Settings` fields must use `Optional[X]` from `typing` (pydantic resolves annotations eagerly regardless of `from __future__ import annotations`); plain functions can use `X | None` in signatures only if the file has `from __future__ import annotations` at the top (defers evaluation, never resolved since nothing calls `get_type_hints` on them).
- Chunking changes require re-running `backend/scripts/reingest.py` once against everything already in `data/uploads/` — `retriever.add_chunks` upserts by `source::chunk_index`, so if a re-chunked document produces fewer chunks than before, old trailing-index chunks are never overwritten and become stale orphans in ChromaDB.

**Settings** (`backend/config.py`): all tunables live in `Settings` (Pydantic BaseSettings). Override via `.env` at the project root. Key fields: `hf_model`, `embedding_model`, `chunk_size`, `chunk_overlap`, `retrieval_candidate_pool`, `similarity_threshold`, `reranker_model`, `rerank_top_n`, `rerank_min_score`, `answer_context_chunk_count`, `answer_context_max_chars`, `answer_context_total_max_chars`, `answer_min_new_tokens`, `answer_max_new_tokens`, `answer_num_beams`, `anthropic_api_key`, `eval_judge_model`, `llm_provider`, `llm_claude_model`, `llm_claude_max_tokens`, `llm_claude_thinking_enabled`.

**Module responsibilities:**
- `services/parser.py` — file-type dispatch; returns `[{text, page}]`
- `services/chunker.py` — sentence-accumulating chunker (sentence-aligned overlap, oversized-sentence fallback for unpunctuated text); returns `[{text, metadata}]`
- `services/embedder.py` — `Embedder` class wrapping sentence-transformers
- `services/retriever.py` — `Retriever` class wrapping ChromaDB
- `services/reranker.py` — `Reranker` class wrapping a `sentence_transformers.CrossEncoder`
- `services/llm.py` — `GenerationBackend` (ABC) with `FlanT5Backend`/`ClaudeBackend` implementations; `LLMService` (shared context building, sentence-safe truncation, citation building) delegates generation to whichever backend it's constructed with; `build_llm_service(provider)` is the factory both `dependencies.py` and `eval/faithfulness_eval.py` use to wire one up
- `backend/dependencies.py` — FastAPI `Depends` providers via `@lru_cache`
- `backend/scripts/reingest.py` — one-off migration: re-chunks and re-embeds every file already in `data/uploads/`; run after any chunking change

## Evaluation (`eval/`)

Script-based harness, not part of the running app. Two independent checks, both reusing the real `backend/` singletons via `sys.path.insert` (the same pattern `backend/scripts/reingest.py` uses), so results always reflect the current config rather than a hardcoded snapshot.

- `eval/retrieval_eval.py` — recall@k against `eval/dataset.json` (hand-labeled question → expected-chunk-substring pairs). Expected chunks are resolved by substring match at eval time, not hardcoded `chunk_index`, since those have already drifted twice this project (chunk-size changes, page-attribution fixes). Free, local, no API key needed. Reports recall at both the retrieval-pool stage and the post-rerank stage separately — they differ in practice (e.g. a bare-acronym query can survive hybrid retrieval but still get filtered out by the reranker).
- `eval/faithfulness_eval.py` — for each positive case, runs the full pipeline through **both** generation providers (`flan_t5` and `claude`, via `services.llm.build_llm_service`, independent of whatever `llm_provider` is currently set to) and has Claude (`claude-opus-4-8`, via `eval/judge.py`) grade whether each generated answer's claims are supported by the exact context that generator saw — this is what makes "Claude is more faithful than flan-t5" a measured number instead of an assumption. Costs real API calls (30 per run: 10 judge calls for flan-t5's answers + 10 Claude generation calls + 10 judge calls for Claude's answers) — run deliberately, not on every change. Requires `ANTHROPIC_API_KEY` in the project root `.env` (used by this script and, if `llm_provider=claude`, by the running app too — see Key design constraints above).
- **Known limitation**: `llm_claude_model` and `eval_judge_model` default to the same model (`claude-opus-4-8`), so the report's Claude-generated rows are graded by the same underlying model that generated them — a milder version of the self-grading concern already discussed and accepted for the judge's own model choice. Not fixed (a third judge model would be scope creep); the flan-t5 rows don't have this issue since judge and generator are always different models there.
- `eval/judge.py` reconstructs the actual context block via `LLMService._build_context_block(...)` directly, rather than using `Citation.excerpt` (only the first 200 chars — shorter than the real per-chunk budget); judging against the excerpt would understate what the model actually saw and produce false "unsupported" verdicts.
- `eval/run_all.py` runs both and merges into `eval/report.md`; each script also writes/prints its own report independently.
- No calibration dataset existed anywhere in the repo before this — the `rerank_min_score=-3.0` calibration (above) was ad-hoc and never saved. `eval/dataset.json` is the first persisted version of that kind of check.

## When planning changes (plan mode)

- Read this whole file for context before proposing a plan.
- Reference existing module boundaries (parser/chunker/embedder/retriever/reranker/llm/dependencies) — extend them, don't restructure unless explicitly asked.
- Flag any new config fields needed in config.py, following existing naming/pattern conventions.
- Explicitly call out anywhere the plan deviates from a decision this file documents as deliberate (e.g. thresholds, budget math, calibrated constants) and explain why.
- Note any place `reingest.py` or the `dependencies.py` singletons need updating as a result of the change.
- Don't write code in plan mode — plan only, wait for review.