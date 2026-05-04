"""Capture an agent decision trace as a :DecisionTrace node in the graph."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modules.local.graph import LocalGraph  # noqa: E402


user_query = "Classify risk on 47 EMEA POs aging >90 days totaling $3.2M"
retrieved_context = [
    "po_aging_not_invoiced: COUNT+SUM, threshold >90 days, dimensions: entity, currency",
    "Escalation: >$500K or >120 days → FBI CapEx lead within 24 hours",
    "GRC-002: All POs invoiced within 90 days or escalated",
]
agent_reasoning = (
    "47 POs totaling $3.2M exceeds $500K HIGH risk threshold. "
    "12 POs >$500K each are HIGH risk. 35 POs <$100K are LOW risk. "
    "GRC-002 requires escalation for POs >90 days."
)
agent_decision = (
    "Classified 12 HIGH risk ($2.4M) and 35 LOW risk ($800K). "
    "Escalating HIGH risk to FBI CapEx lead."
)

print("Simulating agent interaction...\n")
print(f"  User Query: {user_query}")
print(f"  Retrieved: {len(retrieved_context)} chunks")
print(f"  Reasoning: {agent_reasoning}")
print(f"  Decision: {agent_decision}")

trace_id = f"TRACE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
timestamp = datetime.now(timezone.utc).isoformat()
trace_text = f"{user_query} {agent_reasoning} {agent_decision}"

print(f"\nRecording decision trace: {trace_id}")

with LocalGraph() as g:
    g.upsert_node(
        label="DecisionTrace",
        identity_key="id",
        identity_value=trace_id,
        properties={
            "name": trace_id,
            "timestamp": timestamp,
            "user_query": user_query,
            "retrieved_chunks": len(retrieved_context),
            "reasoning": agent_reasoning,
            "decision": agent_decision,
            "confidence": 0.92,
        },
        embed_text=trace_text,
    )
    for entity_name in ["po_aging_not_invoiced", "GRC-002", "EMEA"]:
        g.upsert_edge(trace_id, "CONSIDERED", entity_name)
    g.upsert_edge(trace_id, "RECOMMENDED", "po_aging_not_invoiced")

print("\n✅ Decision trace recorded in context graph!")
print("\nThis trace captures:")
print("  • WHAT was asked (user query)")
print("  • WHAT was retrieved (KB chunks)")
print("  • WHY the decision was made (reasoning)")
print("  • WHAT was decided (recommendation)")
print("  • WHEN it happened (timestamp)")
print("  • WHAT was considered (entity links)")

print(f"\n{'='*60}")
print("Querying decision traces by similarity...")
print(f"{'='*60}")

with LocalGraph() as g:
    hits = g.vector_search("compute service recommendation", top_k=3, labels=["DecisionTrace"])
    for h in hits:
        p = h.properties
        print(f"\n  [{h.score:.3f}] {p.get('id', '')}")
        print(f"  Query: {p.get('user_query', '')}")
        print(f"  Decision: {p.get('decision', '')}")
        print(f"  When: {p.get('timestamp', '')}")
