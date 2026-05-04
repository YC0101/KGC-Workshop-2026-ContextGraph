"""Connect to local Neo4j and reset the context graph schema.

Replaces the original Neptune Analytics provisioning step. The Docker
container is launched separately (see project README); this script just
verifies connectivity, drops any existing data, and creates the
:Embeddable vector index that all later module-04 scripts rely on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modules.local.config import EMBED_DIM, NEO4J_URI  # noqa: E402
from modules.local.graph import LocalGraph, VECTOR_INDEX_NAME  # noqa: E402

print(f"Connecting to Neo4j at {NEO4J_URI}...")
with LocalGraph() as g:
    g.reset()
    print(f"\n✅ Context graph ready!")
    print(f"   URI: {NEO4J_URI}")
    print(f"   Vector index: {VECTOR_INDEX_NAME} on :Embeddable(embedding) ({EMBED_DIM}-dim, cosine)")
    print(f"   All previous nodes/edges have been dropped.")
