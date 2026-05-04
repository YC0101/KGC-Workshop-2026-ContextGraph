"""Strands tools backed by the local Neo4j context graph + Atlas Cloud."""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from openai import OpenAI  # noqa: E402
from strands import tool  # noqa: E402

from modules.local.config import (  # noqa: E402
    ATLAS_API_KEY, ATLAS_BASE_URL, ATLAS_MODEL,
)
from modules.local.graph import LocalGraph  # noqa: E402


@tool
def search_context_graph(query: str) -> str:
    """Search the context graph for entities, relationships, and decision history.

    Use this tool to find WHY things are the way they are — decision traces,
    entity relationships, and historical context. This complements the knowledge
    base which tells you WHAT is true.

    Args:
        query: The search query about entities, decisions, or relationships.

    Returns:
        Relevant entities, their relationships, and any related decision traces.
    """
    parts: list[str] = ["## Entities and Relationships"]
    with LocalGraph() as g:
        for h in g.vector_search(query, top_k=5):
            if h.score < 0.3:
                continue
            labels = ", ".join(l for l in h.labels if l != "Embeddable")
            desc = (h.properties.get("description") or h.properties.get("decision") or "N/A")
            parts.append(f"- [{labels}] **{h.name}**: {desc}")
            for c in g.neighbors(h.name or "", limit=5):
                if c.get("target"):
                    parts.append(f"  → {c['rel']}: {c['target']}")

        parts.append("\n## Decision History")
        for h in g.vector_search(query, top_k=3, labels=["Decision", "DecisionTrace"]):
            if h.score < 0.3:
                continue
            p = h.properties
            title = p.get("title") or p.get("decision") or "N/A"
            when = p.get("date") or p.get("timestamp") or "N/A"
            parts.append(f"- **{title}** ({when})")
            if p.get("context"):
                parts.append(f"  Context: {p['context']}")
            if p.get("rationale"):
                parts.append(f"  Rationale: {p['rationale']}")

    return "\n".join(parts) if len(parts) > 2 else "No relevant context found in the graph."


@tool
def record_decision(
    query: str, reasoning: str, decision: str, entities_considered: str
) -> str:
    """Record an agent decision trace in the context graph for future reference.

    Call this after making a recommendation or decision to capture the reasoning
    for future queries. This builds the 'event clock' — the history of WHY
    decisions were made.

    Args:
        query: The original user question that prompted this decision.
        reasoning: The agent's reasoning process and factors considered.
        decision: The final recommendation or decision made.
        entities_considered: Comma-separated list of entities/topics considered.

    Returns:
        Confirmation that the decision trace was recorded.
    """
    trace_id = f"TRACE-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    timestamp = datetime.now(timezone.utc).isoformat()
    embed_text = f"{query} {reasoning} {decision}"

    with LocalGraph() as g:
        g.upsert_node(
            label="DecisionTrace",
            identity_key="id",
            identity_value=trace_id,
            properties={
                "name": trace_id,
                "timestamp": timestamp,
                "user_query": query,
                "reasoning": reasoning,
                "decision": decision,
            },
            embed_text=embed_text,
        )
        for entity_name in [e.strip() for e in entities_considered.split(",") if e.strip()]:
            g.upsert_edge(trace_id, "CONSIDERED", entity_name)

    return (
        f"Decision trace {trace_id} recorded at {timestamp}. "
        f"This reasoning is now available for future queries."
    )


_atlas = OpenAI(api_key=ATLAS_API_KEY, base_url=ATLAS_BASE_URL)


@tool
def ingest_email_decision(email_text: str) -> str:
    """Extract a decision from an email thread and store it in the context graph.

    Use this when you encounter an email that contains a decision about metric
    thresholds, category changes, reconciliation rules, or system configurations.
    Atlas Cloud extracts structured decision data, then a Decision node is
    written to the graph (linked to existing entities where possible).

    Args:
        email_text: The full email text containing the decision.

    Returns:
        Confirmation of the decision captured with its ID and linked entities.
    """
    extract_prompt = (
        "Extract the decision from this email. Return ONLY valid JSON:\n"
        "{\n"
        '  "decision_id": "short-id like DEC-006 or Q4-AUDIT",\n'
        '  "title": "brief title",\n'
        '  "date": "YYYY-MM-DD",\n'
        '  "decided_by": "person name and role",\n'
        '  "decision": "what was decided",\n'
        '  "rationale": "why it was decided",\n'
        '  "impact": "expected impact",\n'
        '  "entities_affected": ["list of metrics, systems, or configs affected"]\n'
        "}\n\n"
        f"Email:\n{email_text[:4000]}"
    )

    resp = _atlas.chat.completions.create(
        model=ATLAS_MODEL,
        messages=[{"role": "user", "content": extract_prompt}],
        max_tokens=1024,
    )
    content = resp.choices[0].message.content or ""
    content = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", content.strip(), flags=re.MULTILINE)
    start = content.find("{")
    end = content.rfind("}") + 1
    decision = json.loads(content[start:end])

    dec_id = decision.get("decision_id") or f"EMAIL-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    title = decision.get("title", "")
    date = decision.get("date", "")
    decided_by = decision.get("decided_by", "")
    dec_text = decision.get("decision", "")
    rationale = decision.get("rationale", "")
    impact = decision.get("impact", "")
    entities = decision.get("entities_affected", []) or []

    embed_text = f"{title}: {dec_text} {rationale} {impact}"
    linked: list[str] = []

    with LocalGraph() as g:
        g.upsert_node(
            label="Decision",
            identity_key="id",
            identity_value=dec_id,
            properties={
                "name": dec_id,
                "title": title,
                "date": date,
                "decided_by": decided_by,
                "decision": dec_text,
                "rationale": rationale,
                "impact": impact,
                "source": "email",
            },
            embed_text=embed_text,
        )
        for entity_name in entities:
            entity_name = str(entity_name).strip()
            if not entity_name:
                continue
            existing = g.cypher(
                "MATCH (e {name: $name}) RETURN e.name AS name LIMIT 1",
                name=entity_name,
            )
            if not existing:
                # Create as a generic entity so we can still link
                g.upsert_node(
                    label="Entity",
                    identity_key="name",
                    identity_value=entity_name,
                    properties={"name": entity_name, "source": "email_ingest"},
                )
                linked.append(f"{entity_name} (new)")
            else:
                linked.append(entity_name)
            g.upsert_edge(dec_id, "AFFECTS", entity_name)

    return (
        f"Decision captured: {dec_id}\n"
        f"Title: {title}\n"
        f"Date: {date}\n"
        f"Decided by: {decided_by}\n"
        f"Decision: {dec_text}\n"
        f"Rationale: {rationale}\n"
        f"Linked entities: {', '.join(linked) if linked else 'none'}\n"
        f"Source: email thread"
    )
