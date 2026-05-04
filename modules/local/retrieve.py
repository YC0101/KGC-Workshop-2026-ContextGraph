"""In-memory RAG retrieve tool over the AFS sample documents.

Chunks the markdown files in modules/01-knowledge-base-setup/data on first use,
embeds each chunk with LM Studio (Qwen3), and serves top-k cosine matches
for queries from the agent.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from strands import tool

from .config import DATA_DIR
from .embed import embed_texts


@dataclass
class Chunk:
    source: str
    text: str


def _split_markdown(text: str) -> list[str]:
    """Split on top-level (## / ###) markdown headings, keeping the heading with its body."""
    parts = re.split(r"(?=^#{2,3}\s)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(DATA_DIR).glob("*.md")):
        for piece in _split_markdown(path.read_text()):
            chunks.append(Chunk(source=path.name, text=piece))
    return chunks


_chunks: list[Chunk] | None = None
_matrix: np.ndarray | None = None


def _ensure_indexed() -> tuple[list[Chunk], np.ndarray]:
    global _chunks, _matrix
    if _chunks is None:
        _chunks = _load_chunks()
        _matrix = embed_texts(c.text for c in _chunks)
    return _chunks, _matrix  # type: ignore[return-value]


@tool
def retrieve(query: str, top_k: int = 5) -> str:
    """Search the AFS knowledge base for relevant chunks.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of chunks to return.

    Returns:
        Concatenated top-k chunks with source citations.
    """
    chunks, matrix = _ensure_indexed()
    q_vec = embed_texts([query])[0]
    scores = matrix @ q_vec
    idxs = np.argsort(-scores)[:top_k]

    blocks = []
    for rank, i in enumerate(idxs, 1):
        c = chunks[i]
        blocks.append(
            f"[{rank}] source={c.source} score={scores[i]:.3f}\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)
