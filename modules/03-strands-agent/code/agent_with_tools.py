"""Strands agent with AFS-specific custom tools and local-stack retrieval."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strands import Agent, tool  # noqa: E402

from local.llm import build_llm  # noqa: E402
from local.retrieve import retrieve  # noqa: E402


@tool
def classify_risk(po_count: int, total_amount: float, aging_days: int) -> str:
    """Classify financial risk level for a set of POs or assets.

    Args:
        po_count: Number of POs or assets in the group.
        total_amount: Total dollar amount of the group.
        aging_days: Number of days the items have been aging.

    Returns:
        Risk classification with recommended action.
    """
    if total_amount > 500000 or aging_days > 120:
        return (
            f"HIGH RISK: {po_count} items, ${total_amount:,.0f} total, "
            f"{aging_days} days aging. "
            "Action: Escalate to FBI CapEx lead within 24 hours. "
            "Review required before period close."
        )
    if total_amount > 100000 or aging_days > 90:
        return (
            f"MEDIUM RISK: {po_count} items, ${total_amount:,.0f} total, "
            f"{aging_days} days aging. "
            "Action: Review within 48 hours. Flag for close certification."
        )
    return (
        f"LOW RISK: {po_count} items, ${total_amount:,.0f} total, "
        f"{aging_days} days aging. "
        "Action: Standard processing. No escalation needed."
    )


@tool
def get_period_info() -> str:
    """Get the current accounting period information.

    Returns:
        Current period details including close deadlines.
    """
    now = datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    quarter = f"Q{(now.month - 1) // 3 + 1}"
    return (
        f"Current period: {month} ({quarter})\n"
        f"Month-end close deadline: BD+3\n"
        f"Metric refresh deadline: BD+1\n"
        f"Certification deadline: BD+2\n"
        f"Current timestamp: {now.isoformat()}"
    )


@tool
def check_grc_control(control_id: str) -> str:
    """Check the status of a GRC control.

    Args:
        control_id: The GRC control identifier (e.g., GRC-001).

    Returns:
        Control details and current compliance status.
    """
    controls = {
        "GRC-001": ("OFA to FASL+ reconciliation variance < $10K", "Monthly", "FBI CapEx"),
        "GRC-002": ("All POs invoiced within 90 days or escalated", "Monthly", "Accounting"),
        "GRC-003": ("Asset create success rate > 99.5%", "Monthly", "AFS Engineering"),
        "GRC-004": ("ERV variance < 5% for 95% of assets", "Quarterly", "Finance"),
        "GRC-005": ("Full asset inventory reconciliation", "Annually", "FBI CapEx"),
    }
    if control_id in controls:
        desc, freq, owner = controls[control_id]
        return f"Control {control_id}: {desc}\nFrequency: {freq}\nOwner: {owner}\nStatus: ACTIVE"
    return f"Control {control_id} not found. Valid controls: {', '.join(controls.keys())}"


agent = Agent(
    model=build_llm(),
    tools=[retrieve, classify_risk, get_period_info, check_grc_control],
    system_prompt=(
        "You are the AFS Metrics Assistant for AWS CapEx financial operations. You can:\n"
        "1. Search the knowledge base for metric definitions, reconciliation rules, and policies\n"
        "2. Classify financial risk levels for POs and assets\n"
        "3. Check current accounting period and close deadlines\n"
        "4. Look up GRC control details\n\n"
        "Always search the knowledge base first for metric and policy questions. "
        "Use classify_risk when asked to assess risk on financial items."
    ),
)

scenarios = [
    "We have 47 POs totaling $3.2M that have been aging for 95 days without invoices. What's the risk level and what should we do?",
    "What is the current accounting period and when is the metric refresh deadline?",
    "Check GRC-003 control status and tell me what the asset create success rate threshold is.",
]

for scenario in scenarios:
    print(f"\n{'='*60}")
    print(f"📋 Scenario: {scenario}")
    print(f"{'='*60}")
    result = agent(scenario)
    print(f"\n💬 Response:\n{result}")
