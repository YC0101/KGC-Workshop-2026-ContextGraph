"""Loads .env and exposes local-stack settings."""
import os
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

ATLAS_API_KEY = os.environ["ATLAS_API_KEY"]
ATLAS_BASE_URL = os.environ.get("ATLAS_BASE_URL", "https://api.atlascloud.ai/v1")
ATLAS_MODEL = os.environ.get("ATLAS_MODEL", "anthropic/claude-sonnet-4.6")

LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-qwen3-embedding-0.6b")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "workshop-pass")

DATA_DIR = REPO_ROOT / "modules" / "01-knowledge-base-setup" / "data"
