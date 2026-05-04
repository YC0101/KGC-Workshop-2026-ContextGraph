"""Strands @tool that retrieves from the LocalKB built by Module 01.

If the default KB is empty (user hasn't run create_knowledge_base.py yet)
this auto-ingests the AFS sample docs so the tool is usable from a fresh
checkout. After that it uses whatever's in the persistent Chroma collection.
"""
from strands import tool

from .kb import default_kb


_kb = None


def _get_kb():
    global _kb
    if _kb is None:
        _kb = default_kb()
        if _kb.count == 0:
            _kb.ingest_data_dir()
    return _kb


@tool
def retrieve(query: str, top_k: int = 5) -> str:
    """Search the AFS knowledge base for relevant chunks.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of chunks to return.

    Returns:
        Concatenated top-k chunks with source citations.
    """
    kb = _get_kb()
    hits = kb.retrieve(query, top_k=top_k)
    blocks = []
    for rank, h in enumerate(hits, 1):
        src = h.metadata.get("source", "?")
        section = h.metadata.get("section_title", "")
        header = f"[{rank}] source={src} score={h.score:.3f}"
        if section:
            header += f" section={section!r}"
        blocks.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(blocks)
