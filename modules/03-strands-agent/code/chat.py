"""Interactive chat with the AFS Metrics Assistant (local stack)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strands import Agent  # noqa: E402

from local.llm import build_llm  # noqa: E402
from local.retrieve import retrieve  # noqa: E402

agent = Agent(
    model=build_llm(),
    tools=[retrieve],
    system_prompt=(
        "You are the AFS Metrics Assistant. "
        "Search the knowledge base to answer questions about metric definitions, "
        "reconciliation rules, namespace configurations, and compliance policies. "
        "Be concise and cite specific metrics, thresholds, and policy sections."
    ),
)

print("🤖 AFS Metrics Assistant")
print("Ask me about metric definitions, reconciliation rules, or compliance policies.")
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
