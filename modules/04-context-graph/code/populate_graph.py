"""Populate the local Neo4j context graph with entities and relationships.

For each AFS document:
  1. Create a :Document node with content embedding.
  2. Ask Atlas Cloud (Claude Sonnet 4.6) to extract entities, relationships,
     and decisions as JSON.
  3. Upsert each entity as a labelled node (with embedding for vector search).
  4. Upsert each relationship.
  5. Upsert each decision as a :Decision node and link to its document.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from openai import OpenAI  # noqa: E402

from modules.local.config import (  # noqa: E402
    ATLAS_API_KEY, ATLAS_BASE_URL, ATLAS_MODEL, DATA_DIR,
)
from modules.local.graph import LocalGraph  # noqa: E402

EXTRACTION_PROMPT = """Extract entities and relationships from this AFS financial operations document.

Document: {doc_name}
---
{text}
---

Return a JSON object with:
- "entities": list of {{"name": str, "type": str, "description": str}}
  Types: System, Metric, Namespace, Dimension, Policy, Decision, Process
- "relationships": list of {{"source": str, "target": str, "type": str, "description": str}}
  Relationship types: FEEDS_INTO, DEFINES, HAS_DIMENSION, RECONCILES, DECIDED, CAUSED_BY, REPLACES, CONTAINS, TRACKS
- "decisions": list of {{"id": str, "title": str, "date": str, "context": str, "decision": str, "rationale": str}}

Return ONLY valid JSON, no other text."""


client = OpenAI(api_key=ATLAS_API_KEY, base_url=ATLAS_BASE_URL)


def extract_entities(text: str, doc_name: str) -> dict:
    resp = client.chat.completions.create(
        model=ATLAS_MODEL,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(
            doc_name=doc_name, text=text[:6000]
        )}],
        max_tokens=8192,
    )
    content = resp.choices[0].message.content or ""
    content = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", content.strip(), flags=re.MULTILINE)
    start = content.find("{")
    end = content.rfind("}") + 1
    return json.loads(content[start:end])


def main() -> None:
    docs = sorted(Path(DATA_DIR).glob("*.md"))
    print(f"Processing {len(docs)} documents...\n")

    n_entities = n_relationships = n_decisions = 0

    with LocalGraph() as g:
        for path in docs:
            doc_name = path.name
            print(f"📄 Processing: {doc_name}")
            text = path.read_text()

            g.upsert_node(
                label="Document",
                identity_key="name",
                identity_value=doc_name,
                properties={
                    "content_preview": text[:200],
                    "char_count": len(text),
                },
                embed_text=text[:2000],
            )

            try:
                extracted = extract_entities(text, doc_name)
            except Exception as e:
                print(f"  ⚠️  extraction failed: {e}")
                continue

            for entity in extracted.get("entities", []) or []:
                name = (entity.get("name") or "").strip()
                etype = (entity.get("type") or "Entity").strip() or "Entity"
                desc = (entity.get("description") or "").strip()
                if not name:
                    continue
                g.upsert_node(
                    label=etype,
                    identity_key="name",
                    identity_value=name,
                    properties={"description": desc, "type": etype},
                    embed_text=f"{name}: {desc}",
                )
                g.upsert_edge(doc_name, "MENTIONS", name)
                n_entities += 1
                print(f"  + Entity: [{etype}] {name}")

            for rel in extracted.get("relationships", []) or []:
                src = (rel.get("source") or "").strip()
                tgt = (rel.get("target") or "").strip()
                rtype = (rel.get("type") or "RELATED").strip() or "RELATED"
                desc = (rel.get("description") or "").strip()
                if not src or not tgt:
                    continue
                ok = g.upsert_edge(src, rtype, tgt, {"description": desc})
                if ok:
                    n_relationships += 1
                    print(f"  → Relationship: {src} --[{rtype}]--> {tgt}")

            for decision in extracted.get("decisions", []) or []:
                dec_id = (decision.get("id") or "").strip()
                if not dec_id:
                    continue
                title = decision.get("title", "")
                ctx = decision.get("context", "")
                dec_text = decision.get("decision", "")
                rationale = decision.get("rationale", "")
                g.upsert_node(
                    label="Decision",
                    identity_key="id",
                    identity_value=dec_id,
                    properties={
                        "title": title,
                        "date": decision.get("date", "unknown"),
                        "context": ctx,
                        "decision": dec_text,
                        "rationale": rationale,
                        "name": dec_id,
                    },
                    embed_text=f"{title}: {ctx} {dec_text} {rationale}",
                )
                g.upsert_edge(doc_name, "CONTAINS_DECISION", dec_id)
                n_decisions += 1
                print(f"  ★ Decision: {title}")

            print()

    print(f"✅ Graph populated!")
    print(f"   Entities: {n_entities}")
    print(f"   Relationships: {n_relationships}")
    print(f"   Decisions: {n_decisions}")


if __name__ == "__main__":
    main()
