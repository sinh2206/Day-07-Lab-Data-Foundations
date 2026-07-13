from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = max(1, chunk_size)
        self.overlap = min(max(0, overlap), self.chunk_size - 1)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence.strip()]
        return [" ".join(sentences[index : index + self.max_sentences_per_chunk])
                for index in range(0, len(sentences), self.max_sentences_per_chunk)]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = max(1, chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return [chunk.strip() for chunk in self._split(text, self.separators) if chunk.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [current_text[index : index + self.chunk_size]
                    for index in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        if not separator:
            return self._split(current_text, remaining_separators[1:])

        chunks: list[str] = []
        current = ""
        parts = current_text.split(separator)
        for index, part in enumerate(parts):
            part += separator if index < len(parts) - 1 else ""
            if len(part) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split(part, remaining_separators[1:]))
            elif len(current) + len(part) <= self.chunk_size:
                current += part
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude = math.sqrt(_dot(vec_a, vec_a) * _dot(vec_b, vec_b))
    return _dot(vec_a, vec_b) / magnitude if magnitude else 0.0


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size, min(50, chunk_size // 5)).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }
        return {
            name: {
                "chunks": chunks,
                "count": len(chunks),
                "avg_length": sum(map(len, chunks)) / len(chunks) if chunks else 0.0,
            }
            for name, chunks in strategies.items()
        }
