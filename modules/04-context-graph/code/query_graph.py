"""Query the local context graph: vector search → graph traversal → decisions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from modules.local.graph import LocalGraph  # noqa: E402

console = Console()

query = " ".join(sys.argv[1:]) or "Why was the PO aging threshold changed from 60 to 90 days?"
console.print(f"\n🔍 [bold]Query:[/] {query}\n")

with LocalGraph() as g:
    # 1. Vector search across all embeddable nodes
    console.print("[bold cyan]1. Vector Search Results[/]")
    hits = g.vector_search(query, top_k=5)
    for h in hits:
        desc = (h.properties.get("description") or h.properties.get("decision") or "")[:100]
        console.print(
            f"  [{h.score:.3f}] ({', '.join(l for l in h.labels if l != 'Embeddable')}) "
            f"{h.name}: {desc}"
        )

    # 2. Graph traversal — follow relationships from top result
    console.print(f"\n[bold cyan]2. Graph Traversal (relationships from top hit)[/]")
    if hits and hits[0].name:
        for r in g.neighbors(hits[0].name, limit=10):
            target_labels = ", ".join(l for l in (r.get("target_labels") or []) if l != "Embeddable")
            console.print(
                f"  {r['source']} --[{r['rel']}]--> ({target_labels}) {r['target']}"
            )

    # 3. Decision-only vector search
    console.print(f"\n[bold cyan]3. Related Decisions (Event Clock)[/]")
    decision_hits = g.vector_search(query, top_k=3, labels=["Decision"])
    for h in decision_hits:
        p = h.properties
        console.print(Panel(
            f"[bold]{p.get('title', 'N/A')}[/] ({p.get('date', 'N/A')})\n\n"
            f"[dim]Context:[/] {p.get('context', 'N/A')}\n"
            f"[dim]Decision:[/] {p.get('decision', 'N/A')}\n"
            f"[dim]Rationale:[/] {p.get('rationale', 'N/A')}",
            title=f"Decision {p.get('id', '')} (score: {h.score:.3f})",
            border_style="green",
        ))

console.print(f"\n[bold cyan]4. State Clock vs Event Clock[/]")
console.print(Panel(
    "Vector-only RAG returns WHAT is true:\n"
    "  → The policy text, the current rule\n\n"
    "Graph-enhanced RAG also returns WHY:\n"
    "  → The decision that created the rule\n"
    "  → The context that motivated it\n"
    "  → Related entities and their connections\n\n"
    "This is the two clock problem solved.",
    title="State Clock vs Event Clock",
    border_style="yellow",
))
