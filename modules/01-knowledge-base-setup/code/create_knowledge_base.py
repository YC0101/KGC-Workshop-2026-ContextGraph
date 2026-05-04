"""Create the local-stack Knowledge Base — Chroma + LM Studio Qwen3 embeddings.

Replaces the original Bedrock create_knowledge_base + S3 + OpenSearch flow.
The KB persists in ~/.cache/kgc-context-graph/chroma so subsequent runs
just reload the index.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modules.local.kb import LocalKB  # noqa: E402

kb = LocalKB("workshop-kb", chunker="markdown_header")
print(f"Resetting KB collection 'workshop-kb' (chunker=markdown_header)...")
kb.reset()

print("Ingesting AFS sample documents from modules/01-knowledge-base-setup/data/...")
n = kb.ingest_data_dir()

print(f"\n✅ Ingestion complete!")
print(f"   Chunks indexed: {n}")
print(f"   Storage: ~/.cache/kgc-context-graph/chroma/")
print(f"   Embedding model: Qwen3-Embedding-0.6B via LM Studio (1024-dim, cosine)")
