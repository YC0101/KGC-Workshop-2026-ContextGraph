"""Evaluate retrieval quality across all 4 chunking strategies.

Iterates: fixed_size (baseline), semantic, hierarchical, markdown_header (default).
For each strategy it reads the corresponding Chroma collection if present, or
ingests on the fly so the script is self-sufficient. Then it queries each
collection with the same question and prints a side-by-side comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from modules.local.kb import LocalKB  # noqa: E402

console = Console()

STRATEGY_TO_COLLECTION = {
    "fixed_size": "workshop-kb-fixed-size",
    "semantic": "workshop-kb-semantic",
    "hierarchical": "workshop-kb-hierarchical",
    "markdown_header": "workshop-kb",  # the default
}

query = " ".join(sys.argv[1:]) or "What metrics track PO aging for the LLE category?"
console.print(f"\n🔍 [bold]Query:[/] {query}\n")

for strategy, name in STRATEGY_TO_COLLECTION.items():
    kb = LocalKB(name, chunker=strategy)
    if kb.count == 0:
        console.print(f"[yellow]Collection {name} empty — ingesting on the fly...[/]")
        kb.ingest_data_dir()

    hits = kb.retrieve(query, top_k=3)

    table = Table(title=f"[bold cyan]{strategy}[/] → {name} ({kb.count} chunks)")
    table.add_column("#", width=3)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Source", width=28)
    table.add_column("Preview", width=70)
    for i, h in enumerate(hits, 1):
        preview = h.text[:120].replace("\n", " ")
        src = str(h.metadata.get("source", "?"))
        table.add_row(str(i), f"{h.score:.3f}", src, preview + "...")
    console.print(table)
    console.print()
