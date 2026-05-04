"""A simple Strands agent with local-stack retrieval (Atlas LLM + LM Studio embeds)."""
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
        "Use the retrieve tool to search the knowledge base for "
        "metric definitions, reconciliation rules, and compliance policies. "
        "Always cite specific metrics, thresholds, and policy sections."
    ),
)

queries = [
    "What metrics track PO aging for the LLE category?",
    "What are the reconciliation rules between AFS and OFA?",
    "What is the escalation process for reconciliation variances over $1M?",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"🔍 Query: {query}")
    print(f"{'='*60}")
    result = agent(query)
    print(f"\n💬 Answer:\n{result}")
