"""Strands model factory pointing at Atlas Cloud (OpenAI-compatible)."""
from strands.models.openai import OpenAIModel

from .config import ATLAS_API_KEY, ATLAS_BASE_URL, ATLAS_MODEL


def build_llm(model_id: str = ATLAS_MODEL, **params) -> OpenAIModel:
    return OpenAIModel(
        client_args={"api_key": ATLAS_API_KEY, "base_url": ATLAS_BASE_URL},
        model_id=model_id,
        params=params,
    )
