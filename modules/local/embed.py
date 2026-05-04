"""Embedding helper hitting LM Studio's OpenAI-compatible /embeddings endpoint."""
from typing import Iterable

import numpy as np
from openai import OpenAI

from .config import EMBED_MODEL, LMSTUDIO_BASE_URL

_client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
