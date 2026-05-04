"""Hybrid retrieval: LocalKB (Chroma) + LocalGraph (Neo4j) fused with RRF.

Both retrievers run on the same query, their hits are sorted independently,
then merged with Reciprocal Rank Fusion. Atlas Cloud generates the final
grounded answer over the fused context.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from openai import OpenAI  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from modules.local.config import ATLAS_API_KEY, ATLAS_BASE_URL, ATLAS_MODEL  # noqa: E402
from modules.local.graph import LocalGraph  # noqa: E402
from modules.local.kb import default_kb  # noqa: E402

console = Console()

query = " ".join(sys.argv[1:]) or (
    "What metrics should I check for month-end close and why were the "
    "thresholds set this way?"
)
console.print(f"\n🔍 [bold]Query:[/] {query}\n")

# --- Source 1: Local Knowledge Base (Chroma vector search) ---
console.print("[bold cyan]Source 1: Local KB (Chroma vector search)[/]")
kb = default_kb()
if kb.count == 0:
    kb.ingest_data_dir()
kb_hits = kb.retrieve(query, top_k=5)
vector_chunks = []
for h in kb_hits:
    vector_chunks.append({"text": h.text, "score": h.score, "source": "KB"})
    console.print(f"  [{h.score:.3f}] {h.text[:100]}...".replace("\n", " "))

# --- Source 2: Context Graph (Neo4j vector + traversal) ---
console.print(f"\n[bold cyan]Source 2: Context Graph (Neo4j vector + traversal)[/]")
graph_chunks = []
with LocalGraph() as g:
    graph_hits = g.vector_search(query, top_k=5)
    for h in graph_hits:
        p = h.properties
        desc = p.get("description") or p.get("decision") or ""
        connections = g.neighbors(h.name, limit=5) if h.name else []
        context_parts = [f"{h.name}: {desc}"]
        for c in connections:
            if c.get("target"):
                context_parts.append(f"  → {c['rel']}: {c['target']}")
        graph_chunks.append({
            "text": "\n".join(context_parts),
            "score": h.score,
            "source": "Graph",
        })
        console.print(f"  [{h.score:.3f}] {h.name}: {desc[:80]}...".replace("\n", " "))
        for c in connections[:3]:
            if c.get("target"):
                console.print(f"    → {c['rel']}: {c['target']}")

    # --- Source 3: Decision/DecisionTrace nodes ---
    console.print(f"\n[bold cyan]Source 3: Decision Traces (Event Clock)[/]")
    decision_hits = g.vector_search(query, top_k=3, labels=["Decision", "DecisionTrace"])
    for h in decision_hits:
        p = h.properties
        title = p.get("title") or p.get("decision") or "N/A"
        rationale = p.get("rationale") or p.get("context") or ""
        graph_chunks.append({
            "text": f"Decision: {title}. Rationale: {rationale}",
            "score": h.score,
            "source": "Decision",
        })
        console.print(f"  [{h.score:.3f}] {title}: {rationale[:80]}...".replace("\n", " "))

# --- Reciprocal Rank Fusion ---
console.print(f"\n[bold cyan]Fused Results (Reciprocal Rank Fusion)[/]")
K = 60
all_items: dict[str, dict] = {}
for rank, chunk in enumerate(sorted(vector_chunks, key=lambda x: -x["score"])):
    key = chunk["text"][:100]
    all_items.setdefault(key, {"text": chunk["text"], "rrf_score": 0, "sources": []})
    all_items[key]["rrf_score"] += 1 / (K + rank + 1)
    all_items[key]["sources"].append(chunk["source"])
for rank, chunk in enumerate(sorted(graph_chunks, key=lambda x: -x["score"])):
    key = chunk["text"][:100]
    all_items.setdefault(key, {"text": chunk["text"], "rrf_score": 0, "sources": []})
    all_items[key]["rrf_score"] += 1 / (K + rank + 1)
    all_items[key]["sources"].append(chunk["source"])

fused = sorted(all_items.values(), key=lambda x: -x["rrf_score"])[:5]

table = Table(title="Top 5 Fused Results")
table.add_column("RRF Score", justify="right", width=10)
table.add_column("Sources", width=18)
table.add_column("Content", width=70)
for item in fused:
    table.add_row(
        f"{item['rrf_score']:.4f}",
        ", ".join(sorted(set(item["sources"]))),
        item["text"][:120].replace("\n", " ") + "...",
    )
console.print(table)

# --- Final answer with full fused context ---
console.print(f"\n[bold cyan]Final Answer (with full fused context)[/]")
context = "\n\n---\n\n".join(item["text"][:500] for item in fused)

client = OpenAI(api_key=ATLAS_API_KEY, base_url=ATLAS_BASE_URL)
resp = client.chat.completions.create(
    model=ATLAS_MODEL,
    messages=[
        {"role": "system",
         "content": "Answer using the provided context. Include both WHAT is "
                    "true now and WHY (cite decisions and rationale)."},
        {"role": "user",
         "content": f"Question: {query}\n\nContext:\n{context}"},
    ],
    max_tokens=1024,
)

console.print(f"\n💬 {resp.choices[0].message.content}")
