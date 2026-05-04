"""Local knowledge base backed by Chroma + LM Studio embeddings.

Drop-in replacement for the parts of Bedrock Knowledge Bases the workshop
exercises: ingest documents under a chosen chunking strategy, then retrieve
top-k chunks for a query.

Multiple KBs (one per chunking strategy) coexist as separate Chroma
collections under ~/.cache/kgc-context-graph/chroma.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings

from .chunking import STRATEGIES, Chunk, Chunker
from .config import DATA_DIR, EMBED_DIM
from .embed import embed_texts


CHROMA_DIR = Path.home() / ".cache" / "kgc-context-graph" / "chroma"


@dataclass
class Hit:
    text: str
    metadata: dict
    score: float


def _client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


class LocalKB:
    def __init__(self, name: str, chunker: Chunker | str = "markdown_header"):
        self.name = name
        if isinstance(chunker, str):
            chunker = STRATEGIES[chunker]
        self.chunker = chunker
        self._client = _client()
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "embed_dim": EMBED_DIM},
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine", "embed_dim": EMBED_DIM},
        )

    def ingest_paths(self, paths: Iterable[Path]) -> int:
        chunks: list[Chunk] = []
        for path in paths:
            text = Path(path).read_text()
            chunks.extend(self.chunker(text, Path(path).name))
        return self._add(chunks)

    def ingest_data_dir(self) -> int:
        return self.ingest_paths(sorted(Path(DATA_DIR).glob("*.md")))

    def _add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vecs = embed_texts(c.text for c in chunks)
        self._collection.add(
            ids=[f"{c.metadata['source']}::{c.metadata.get('strategy','?')}::{i}"
                 for i, c in enumerate(chunks)],
            embeddings=vecs.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[Hit]:
        q_vec = embed_texts([query])[0]
        res = self._collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=top_k,
        )
        hits: list[Hit] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            # cosine distance → similarity
            hits.append(Hit(text=text, metadata=meta or {}, score=1.0 - dist))
        return hits

    @property
    def count(self) -> int:
        return self._collection.count()


def default_kb() -> LocalKB:
    """The KB that Module 03's retrieve tool reads from."""
    return LocalKB("workshop-kb", chunker="markdown_header")
