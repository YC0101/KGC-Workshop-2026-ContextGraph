"""Multi-agent system with AFS specialist agents."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strands import Agent, tool  # noqa: E402

from local.llm import build_llm  # noqa: E402
from local.retrieve import retrieve  # noqa: E402

model = build_llm()

metrics_agent = Agent(
    model=model,
    tools=[retrieve],
    system_prompt=(
        "You are an AFS metrics specialist. You answer questions about "
        "metric definitions, namespaces, dimensions, filters, and periods. "
        "Always search the knowledge base and cite specific metric names and configurations."
    ),
)

reconciliation_agent = Agent(
    model=model,
    tools=[retrieve],
    system_prompt=(
        "You are an AFS reconciliation specialist. You answer questions about "
        "reconciliation rules between systems (OFA, FASL+, AP, SC.os, AFS), "
        "matching logic, tolerances, and escalation procedures."
    ),
)

compliance_agent = Agent(
    model=model,
    tools=[retrieve],
    system_prompt=(
        "You are an AFS compliance specialist. You answer questions about "
        "SOX requirements, GRC controls, period close procedures, "
        "data retention policies, and audit requirements."
    ),
)


@tool
def ask_metrics_expert(question: str) -> str:
    """Ask the metrics specialist about metric definitions, namespaces, dimensions, or filters.

    Args:
        question: The metrics-related question.
    """
    return str(metrics_agent(question))


@tool
def ask_reconciliation_expert(question: str) -> str:
    """Ask the reconciliation specialist about matching rules, tolerances, or system reconciliation.

    Args:
        question: The reconciliation-related question.
    """
    return str(reconciliation_agent(question))


@tool
def ask_compliance_expert(question: str) -> str:
    """Ask the compliance specialist about SOX, GRC controls, close procedures, or audit requirements.

    Args:
        question: The compliance-related question.
    """
    return str(compliance_agent(question))


router = Agent(
    model=model,
    tools=[ask_metrics_expert, ask_reconciliation_expert, ask_compliance_expert],
    system_prompt=(
        "You are the AFS Metrics Assistant router. Classify questions and delegate:\n"
        "- Metric definitions, namespaces, dimensions → ask_metrics_expert\n"
        "- Reconciliation rules, matching, tolerances → ask_reconciliation_expert\n"
        "- SOX, GRC, close procedures, audit → ask_compliance_expert\n\n"
        "If a question spans multiple domains, consult multiple specialists."
    ),
)

queries = [
    "What metrics track PO aging and what's the reconciliation rule for PO to receipt matching?",
    "What are the GRC controls related to asset creation and what's the metric definition for failed creates?",
    "What's the month-end close procedure and which metrics need to be refreshed before close?",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"🔍 Query: {query}")
    print(f"{'='*60}")
    result = router(query)
    print(f"\n💬 Answer:\n{result}")
