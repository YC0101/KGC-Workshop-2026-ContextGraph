"""Chunking strategies — pure Python, mirrors the four strategies in
modules/02-custom-chunking/code/compare_chunking.py but returns a uniform
list of (text, metadata) tuples ready for KB ingestion.
"""
import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class Chunk:
    text: str
    metadata: dict


Chunker = Callable[[str, str], list[Chunk]]
"""A chunker takes (document_text, source_name) and returns chunks."""


def fixed_size(text: str, source: str, max_chars: int = 1000, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    start, idx = 0, 0
    while start < len(text):
        end = start + max_chars
        chunks.append(Chunk(
            text=text[start:end],
            metadata={"source": source, "strategy": "fixed_size", "index": idx},
        ))
        idx += 1
        start = end - overlap
    return chunks


def semantic(text: str, source: str, max_chars: int = 1500) -> list[Chunk]:
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[Chunk] = []
    current = ""
    idx = 0
    for para in paragraphs:
        if len(current) + len(para) > max_chars:
            if current:
                chunks.append(Chunk(
                    text=current.strip(),
                    metadata={"source": source, "strategy": "semantic", "index": idx},
                ))
                idx += 1
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(Chunk(
            text=current.strip(),
            metadata={"source": source, "strategy": "semantic", "index": idx},
        ))
    return chunks


def hierarchical(text: str, source: str) -> list[Chunk]:
    """Parent-child: section heading is the parent, paragraphs within are children.
    For ingestion we emit both, distinguished by metadata.kind = parent|child.
    """
    sections = re.split(r"\n(?=## )", text)
    chunks: list[Chunk] = []
    for sec_idx, section in enumerate(sections):
        if not section.strip():
            continue
        first_line = section.strip().split("\n", 1)[0]
        title = first_line.strip("# ").strip()
        chunks.append(Chunk(
            text=section.strip(),
            metadata={"source": source, "strategy": "hierarchical", "kind": "parent",
                      "section": title, "index": sec_idx},
        ))
        for c_idx, child in enumerate(re.split(r"\n\n+", section)):
            if len(child.strip()) > 20:
                chunks.append(Chunk(
                    text=child.strip(),
                    metadata={"source": source, "strategy": "hierarchical", "kind": "child",
                              "section": title, "index": c_idx, "parent_index": sec_idx},
                ))
    return chunks


def markdown_header(text: str, source: str) -> list[Chunk]:
    """Custom: split on markdown headers, attach structural metadata."""
    sections = re.split(r"\n(?=##+ )", text)
    chunks: list[Chunk] = []
    for idx, section in enumerate(sections):
        if not section.strip():
            continue
        first_line = section.strip().split("\n", 1)[0]
        m = re.match(r"^(#+)\s+(.+)", first_line)
        title = m.group(2) if m else "Introduction"
        level = len(m.group(1)) if m else 1
        chunks.append(Chunk(
            text=section.strip(),
            metadata={
                "source": source,
                "strategy": "markdown_header",
                "section_title": title,
                "header_level": level,
                "has_table": ("|" in section and "---" in section),
                "has_code": "```" in section,
                "index": idx,
            },
        ))
    return chunks


STRATEGIES: dict[str, Chunker] = {
    "fixed_size": fixed_size,
    "semantic": semantic,
    "hierarchical": hierarchical,
    "markdown_header": markdown_header,
}
