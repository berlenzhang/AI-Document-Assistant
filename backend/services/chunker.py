from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

Unit = tuple[str, "int | None", int]  # (sentence_text, page_number, word_count)


def _split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _segment_units(pages: list[dict]) -> list[Unit]:
    """Flatten pages into an ordered list of (sentence, page, word_count) units."""
    units: list[Unit] = []
    for page in pages:
        for paragraph in re.split(r"\n+", page["text"]):
            for sentence in _split_into_sentences(paragraph):
                word_count = len(sentence.split())
                if word_count:
                    units.append((sentence, page["page"], word_count))
    return units


def chunk_document(
    pages: list[dict],
    filename: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_length: int = 50,
) -> list[dict]:
    units = _segment_units(pages)
    if not units:
        return []

    chunks: list[dict] = []
    chunk_index = 0
    i = 0
    n = len(units)
    step = max(1, chunk_size - overlap)

    while i < n:
        # Oversized-sentence fallback: a single unit alone exceeds chunk_size,
        # so it can't be accumulated normally. Fall back to raw word-slicing
        # for just that one unit.
        if units[i][2] > chunk_size:
            words = units[i][0].split()
            for start in range(0, len(words), step):
                text = " ".join(words[start:start + chunk_size])
                if len(text) >= min_length:
                    chunks.append({
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "page": units[i][1],
                            "chunk_index": chunk_index,
                        },
                    })
                    chunk_index += 1
            i += 1
            continue

        window: list[Unit] = []
        word_count = 0
        while i < n and word_count + units[i][2] <= chunk_size:
            window.append(units[i])
            word_count += units[i][2]
            i += 1

        text = " ".join(u[0] for u in window)
        if len(text) >= min_length:
            chunks.append({
                "text": text,
                "metadata": {
                    # The excerpt shown/highlighted in the UI is always taken
                    # from the start of this text, so the page must be where
                    # the text starts, not a word-count majority page.
                    "source": filename,
                    "page": window[0][1],
                    "chunk_index": chunk_index,
                },
            })
            chunk_index += 1

        if i >= n:
            break

        # Seed overlap: back up `i` to the trailing sentences of this window
        # whose combined word count is closest to `overlap`, without undoing
        # all forward progress.
        overlap_words = 0
        back = 0
        for unit in reversed(window):
            if overlap_words >= overlap:
                break
            overlap_words += unit[2]
            back += 1
        i -= min(back, len(window) - 1)

    return chunks
