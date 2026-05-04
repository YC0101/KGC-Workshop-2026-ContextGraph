"""Create a Chroma collection ingested with the SEMANTIC chunking strategy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modules.local.kb import LocalKB  # noqa: E402

print("Creating data source with SEMANTIC chunking...")
kb = LocalKB("workshop-kb-semantic", chunker="semantic")
kb.reset()
n = kb.ingest_data_dir()
print(f"✅ Semantic collection 'workshop-kb-semantic' ready: {n} chunks")
