"""Evaluate KB-only vs Graph-enhanced answers, side by side."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from strands import Agent  # noqa: E402

from modules.local.config import (  # noqa: E402
    ATLAS_API_KEY, ATLAS_BASE_URL, ATLAS_MODEL,
)
from modules.local.kb import default_kb  # noqa: E402
from modules.local.llm import build_llm  # noqa: E402
from modules.local.retrieve import retrieve  # noqa: E402
from tools.graph_tools import search_context_graph  # noqa: E402

console = Console()

test_queries = [
    {
        "query": "Why is the PO aging threshold set to 90 days?",
        "expected": "Should explain Q4-2024 audit decision, not just state the threshold",
    },
    {
        "query": "What led to adding GENERATOR to the LLE categories?",
        "expected": "Should reference DEC-004 and FY2024 GAAP update",
    },
    {
        "query": "How do AFS and OFA reconcile asset values?",
        "expected": "Should explain matching rules and the decision to use DataStudio Redshift",
    },
]


atlas = OpenAI(api_key=ATLAS_API_KEY, base_url=ATLAS_BASE_URL)
kb = default_kb()


def kb_only_answer(query: str) -> str:
    """Plain RAG: top-k chunks from Chroma, then Atlas synthesis. No graph."""
    hits = kb.retrieve(query, top_k=5)
    context = "\n\n---\n\n".join(
        f"[{h.metadata.get('source')}] {h.text}" for h in hits
    )
    resp = atlas.chat.completions.create(
        model=ATLAS_MODEL,
        messages=[
            {"role": "system",
             "content": "Answer using ONLY the provided context. Cite source filenames."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


graph_agent = Agent(
    model=build_llm(),
    tools=[retrieve, search_context_graph],
    system_prompt=(
        "Answer using both the knowledge base (current facts) "
        "and the context graph (decision history, relationships). "
        "Explain both WHAT is true and WHY."
    ),
)


console.print("\n[bold]📊 Evaluation: KB-Only vs Graph-Enhanced[/]\n")

for test in test_queries:
    query = test["query"]
    console.print(f"[bold cyan]Query:[/] {query}")
    console.print(f"[dim]Expected: {test['expected']}[/]\n")

    kb_answer = kb_only_answer(query)
    graph_answer = str(graph_agent(query))

    table = Table(show_header=True, header_style="bold")
    table.add_column("KB-Only Answer", width=55)
    table.add_column("Graph-Enhanced Answer", width=55)
    table.add_row(
        kb_answer[:600] + ("..." if len(kb_answer) > 600 else ""),
        graph_answer[:600] + ("..." if len(graph_answer) > 600 else ""),
    )
    console.print(table)
    console.print()

console.print("[bold green]✅ Evaluation complete![/]")
console.print(
    "Notice how graph-enhanced answers include decision context, "
    "rationale, and relationships that KB-only answers miss."
)
