"""Query the local-stack KB. Atlas Cloud generates a grounded answer from
the top-k Chroma chunks; --verbose prints the chunks and similarity scores.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from openai import OpenAI  # noqa: E402

from modules.local.config import (  # noqa: E402
    ATLAS_API_KEY, ATLAS_BASE_URL, ATLAS_MODEL,
)
from modules.local.kb import LocalKB  # noqa: E402

verbose = "--verbose" in sys.argv
query = " ".join(arg for arg in sys.argv[1:] if arg != "--verbose")

if not query:
    print("Usage: python query_kb.py [--verbose] <your question>")
    print("Examples:")
    print('  python query_kb.py "What metrics track PO aging?"')
    print('  python query_kb.py "What are the GRC controls for reconciliation?"')
    print('  python query_kb.py --verbose "What is the PO aging threshold?"')
    sys.exit(1)

print(f"🔍 Query: {query}\n")

kb = LocalKB("workshop-kb", chunker="markdown_header")
hits = kb.retrieve(query, top_k=5)

if not hits:
    print("No chunks indexed. Did you run create_knowledge_base.py first?")
    sys.exit(1)

context = "\n\n---\n\n".join(
    f"[source={h.metadata.get('source')} score={h.score:.3f}]\n{h.text}"
    for h in hits
)

client = OpenAI(api_key=ATLAS_API_KEY, base_url=ATLAS_BASE_URL)
resp = client.chat.completions.create(
    model=ATLAS_MODEL,
    messages=[
        {"role": "system",
         "content": "Answer the user's question using ONLY the provided context. "
                    "Cite source filenames in brackets like [metric-definitions.md]."},
        {"role": "user",
         "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ],
)

print("💬 Answer:")
print(resp.choices[0].message.content)

print(f"\n📚 Sources ({len(hits)} chunks):")
seen: set[str] = set()
for h in hits:
    src = h.metadata.get("source", "?")
    if src not in seen:
        seen.add(src)
        print(f"  - {src}")

if verbose:
    print("\n" + "=" * 60)
    print("RETRIEVAL DETAILS")
    print("=" * 60)
    for i, h in enumerate(hits, 1):
        print(f"\n--- Chunk {i} (score: {h.score:.4f}) ---")
        print(f"Source: {h.metadata.get('source')}")
        print(f"Section: {h.metadata.get('section_title','')}")
        print(f"Text: {h.text[:200]}...")
