"""Show all DecisionTrace nodes captured in the local context graph."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from modules.local.graph import LocalGraph  # noqa: E402

console = Console()
console.print("\n[bold]📝 Decision Traces in Context Graph[/]\n")

with LocalGraph() as g:
    rows = g.cypher(
        """
        MATCH (t:DecisionTrace)
        OPTIONAL MATCH (t)-[:CONSIDERED]->(e)
        RETURN t.id AS id, t.timestamp AS timestamp,
               t.user_query AS query, t.reasoning AS reasoning,
               t.decision AS decision,
               collect(coalesce(e.name, e.id)) AS entities_considered
        ORDER BY t.timestamp DESC
        """
    )

if not rows:
    console.print("[dim]No decision traces found. Run the integrated agent first.[/]")
else:
    console.print(f"Found {len(rows)} decision trace(s):\n")
    for r in rows:
        entities = ", ".join(e for e in r.get("entities_considered", []) if e) or "None"
        console.print(Panel(
            f"[bold]Query:[/] {r.get('query', 'N/A')}\n\n"
            f"[bold]Reasoning:[/] {r.get('reasoning', 'N/A')}\n\n"
            f"[bold]Decision:[/] {r.get('decision', 'N/A')}\n\n"
            f"[bold]Entities Considered:[/] {entities}\n"
            f"[dim]Timestamp: {r.get('timestamp', 'N/A')}[/]",
            title=f"🔍 {r.get('id', 'Unknown')}",
            border_style="green",
        ))
