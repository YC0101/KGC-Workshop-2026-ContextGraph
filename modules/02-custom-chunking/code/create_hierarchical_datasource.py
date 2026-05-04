"""Create a Chroma collection ingested with the HIERARCHICAL chunking strategy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modules.local.kb import LocalKB  # noqa: E402

print("Creating data source with HIERARCHICAL chunking...")
kb = LocalKB("workshop-kb-hierarchical", chunker="hierarchical")
kb.reset()
n = kb.ingest_data_dir()
print(f"✅ Hierarchical collection 'workshop-kb-hierarchical' ready: {n} chunks "
      "(parent + child entries combined)")
