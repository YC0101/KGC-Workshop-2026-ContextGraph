"""Interactive chat with the fully integrated Agentic RAG system."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strands import Agent  # noqa: E402

from modules.local.llm import build_llm  # noqa: E402
from modules.local.retrieve import retrieve  # noqa: E402
from tools.graph_tools import (  # noqa: E402
    search_context_graph, record_decision, ingest_email_decision,
)


agent = Agent(
    model=build_llm(),
    tools=[retrieve, search_context_graph, record_decision, ingest_email_decision],
    system_prompt=(
        "You are the AFS Metrics Assistant for AWS CapEx financial operations. "
        "Use retrieve for metric definitions and policies. "
        "Use search_context_graph for decision history and relationships. "
        "Use record_decision when you make classifications or recommendations. "
        "Always explain both WHAT is true and WHY."
    ),
)

print("🤖 AFS Metrics Assistant")
print("=" * 40)
print("I can answer questions using:")
print("  📚 Knowledge Base (metric definitions, recon rules)")
print("  🔗 Context Graph (decision history, system relationships)")
print("  📝 Decision Recording (audit trail)")
print("  📧 Email Decision Capture (extract decisions from emails)")
print()
print("Try asking:")
print('  "What metrics track PO aging for LLE?"')
print('  "Why is the PO aging threshold 90 days?"')
print('  "Classify risk on 47 POs totaling $3.2M"')
print('  Paste an email and ask: "Capture the decision from this email"')
print()
print("Type 'quit' to exit.\n")

while True:
    try:
        query = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not query or query.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    result = agent(query)
    print(f"\nAssistant: {result}\n")
