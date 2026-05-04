"""Local context graph backed by Neo4j 5.x with native vector index.

Drop-in replacement for the workshop's Neptune Analytics usage. Every node
that should be vector-searchable carries an `:Embeddable` label plus a
`name` and `embedding` property. A single vector index over
`:Embeddable(embedding)` powers all top-k queries; for label-restricted
queries we filter by `labels(node)` after the index hit.

The default schema mirrors what populate_graph.py emits:
  Document, Decision, DecisionTrace, plus dynamic entity labels
  (System, Metric, Namespace, Dimension, Policy, Process, ...).
"""
from dataclasses import dataclass
from typing import Iterable

from neo4j import GraphDatabase

from .config import EMBED_DIM, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from .embed import embed_texts


VECTOR_INDEX_NAME = "embeddable_vec"


@dataclass
class GraphHit:
    labels: list[str]
    name: str | None
    properties: dict
    score: float


class LocalGraph:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def reset(self) -> None:
        """Drop all data and rebuild the vector index."""
        with self._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
            s.run(f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS")
            s.run(
                f"""
                CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
                FOR (n:Embeddable) ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                dim=EMBED_DIM,
            )

    def upsert_node(
        self,
        label: str,
        identity_key: str,
        identity_value: str,
        properties: dict,
        embed_text: str | None = None,
    ) -> None:
        """MERGE a node by (label, identity_key=identity_value).

        If embed_text is provided, the node is embedded and tagged :Embeddable
        so the vector index picks it up.
        """
        embedding = None
        if embed_text:
            embedding = embed_texts([embed_text])[0].tolist()

        properties = {k: v for k, v in properties.items() if v is not None}

        with self._driver.session() as s:
            if embedding is None:
                s.run(
                    f"""
                    MERGE (n:`{label}` {{ `{identity_key}`: $id_val }})
                    SET n += $props
                    """,
                    id_val=identity_value,
                    props=properties,
                )
            else:
                s.run(
                    f"""
                    MERGE (n:`{label}` {{ `{identity_key}`: $id_val }})
                    SET n:Embeddable, n += $props, n.embedding = $vec
                    """,
                    id_val=identity_value,
                    props=properties,
                    vec=embedding,
                )

    def upsert_edge(
        self,
        source_name: str,
        rel_type: str,
        target_name: str,
        properties: dict | None = None,
    ) -> bool:
        """MERGE source -[rel_type]-> target. Matches nodes by `name` regardless of label.

        Returns True if both endpoints existed and the edge was created/found.
        """
        with self._driver.session() as s:
            res = s.run(
                f"""
                MATCH (src {{name: $src}})
                MATCH (tgt {{name: $tgt}})
                MERGE (src)-[r:`{rel_type}`]->(tgt)
                ON CREATE SET r += $props
                RETURN elementId(r) AS rid
                """,
                src=source_name,
                tgt=target_name,
                props=(properties or {}),
            )
            return res.single() is not None

    def vector_search(
        self,
        query: str,
        top_k: int = 5,
        labels: Iterable[str] | None = None,
    ) -> list[GraphHit]:
        """Top-k cosine search on the :Embeddable index, optionally restricted by label.

        With a label filter, we oversample by counting how many nodes carry
        any matching label and fetch up to that many (capped at 200) so even
        sparsely populated labels return their best matches.
        """
        q_vec = embed_texts([query])[0].tolist()
        labels = list(labels) if labels else None
        ask_k = top_k
        if labels:
            with self._driver.session() as s:
                row = s.run("MATCH (n:Embeddable) RETURN count(n) AS c").single()
            total = row["c"] if row else top_k
            ask_k = min(total, 500)

        with self._driver.session() as s:
            cypher = f"""
                CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $k, $vec)
                YIELD node, score
                RETURN labels(node) AS labels, properties(node) AS props, score
            """
            res = s.run(cypher, k=ask_k, vec=q_vec)
            hits: list[GraphHit] = []
            for row in res:
                lbls = list(row["labels"])
                if labels and not any(lbl in lbls for lbl in labels):
                    continue
                props = dict(row["props"])
                props.pop("embedding", None)  # drop the giant vector from props
                hits.append(GraphHit(
                    labels=lbls,
                    name=props.get("name") or props.get("id") or props.get("title"),
                    properties=props,
                    score=row["score"],
                ))
                if len(hits) >= top_k:
                    break
            return hits

    def cypher(self, query: str, **params) -> list[dict]:
        """Run a raw Cypher query, return list of records as dicts (embeddings stripped)."""
        with self._driver.session() as s:
            res = s.run(query, **params)
            out: list[dict] = []
            for record in res:
                row = {}
                for key, val in record.items():
                    if hasattr(val, "items"):
                        clean = {k: v for k, v in val.items() if k != "embedding"}
                        row[key] = clean
                    else:
                        row[key] = val
                out.append(row)
            return out

    def neighbors(self, node_name: str, limit: int = 10) -> list[dict]:
        """Return outgoing + incoming edges for the given named node."""
        with self._driver.session() as s:
            res = s.run(
                """
                MATCH (n {name: $name})-[r]-(connected)
                RETURN n.name AS source,
                       type(r) AS rel,
                       labels(connected) AS target_labels,
                       coalesce(connected.name, connected.id, connected.title) AS target,
                       coalesce(connected.description, '') AS target_description
                LIMIT $limit
                """,
                name=node_name,
                limit=limit,
            )
            return [dict(r) for r in res]
