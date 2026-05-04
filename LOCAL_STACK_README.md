# Local Stack — running the workshop without AWS

This branch (`local-stack`) replaces every AWS-bound dependency in the original
workshop with a fully local equivalent, so you can run all five modules on your
laptop for the cost of a few Atlas Cloud tokens (probably under $0.50 for the
entire walkthrough). Same architecture, same lessons, no cloud bill.

## What was swapped

| Original (AWS) | Local replacement |
|---|---|
| Bedrock Claude (LLM) | **Atlas Cloud** OpenAI-compatible API → `anthropic/claude-sonnet-4.6` |
| Bedrock Titan v2 (embeddings) | **LM Studio** local server → `Qwen3-Embedding-0.6B` (1024-dim) |
| Bedrock Knowledge Base + OpenSearch Serverless | **Chroma** persistent collection at `~/.cache/kgc-context-graph/chroma/` |
| Lambda custom chunker | In-process `markdown_header` chunker in `modules/local/chunking.py` |
| Neptune Analytics (graph + vector) | **Neo4j 5.20** in Docker, native vector index over `:Embeddable(embedding)` |
| Bedrock retrieve_and_generate | Plain Python: top-k chunks → Atlas chat completion |
| AWS CDK / CloudFormation | None — nothing to provision |

Net diff vs upstream `main`: roughly **-1500 / +900** lines.

## Prerequisites

1. **macOS or Linux** with Python 3.11+
2. **Docker Desktop** running (for Neo4j) — `docker --version` should work
3. **LM Studio** with `Qwen3-Embedding-0.6B-GGUF` (Q8 quant) loaded and the
   local server running on `http://127.0.0.1:1234`
4. **Atlas Cloud API key** from <https://www.atlascloud.ai/> (cheap; pay per token)

## One-time setup

```bash
# 1. Clone and switch to this branch
git clone https://github.com/YC0101/KGC-Workshop-2026-ContextGraph.git
cd KGC-Workshop-2026-ContextGraph
git checkout local-stack

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ipykernel  # only if you want to run workshop.ipynb

# 3. Secrets — copy and edit .env
cat > .env <<'EOF'
ATLAS_API_KEY=apikey-XXXXXXXXXXXXXXXXXXXXXXXXX
ATLAS_BASE_URL=https://api.atlascloud.ai/v1
ATLAS_MODEL=anthropic/claude-sonnet-4.6

LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
EMBED_MODEL=text-embedding-qwen3-embedding-0.6b
EMBED_DIM=1024

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=workshop-pass
EOF

# 4. Start Neo4j
docker run -d --name neo4j-kgc \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/workshop-pass \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.20

# wait ~20s, then verify
docker logs neo4j-kgc 2>&1 | grep -q "Started." && echo "neo4j up"
```

Open Neo4j Browser at <http://localhost:7474> (login `neo4j` / `workshop-pass`)
to inspect data later.

## Run order

Every script is idempotent — you can re-run any of them. Suggested first-time
walkthrough:

### Module 01 — Knowledge Base

```bash
python modules/01-knowledge-base-setup/code/create_knowledge_base.py
python modules/01-knowledge-base-setup/code/query_kb.py --verbose "What is the PO aging threshold?"
```

`create_knowledge_base.py` ingests the 5 AFS sample docs into Chroma using the
`markdown_header` chunker. `query_kb.py` retrieves top-k chunks and asks Atlas
for a grounded answer.

### Module 02 — Custom Chunking

```bash
# Pure-Python comparison (no API calls)
python modules/02-custom-chunking/code/compare_chunking.py

# Build alternative collections (only needed once)
python modules/02-custom-chunking/code/create_semantic_datasource.py
python modules/02-custom-chunking/code/create_hierarchical_datasource.py
python modules/02-custom-chunking/code/create_custom_datasource.py   # fixed-size baseline

# Side-by-side retrieval comparison
python modules/02-custom-chunking/code/evaluate_chunking.py "What is the PO aging threshold?"
```

> Note: `create_custom_datasource.py` provides the **fixed-size baseline** in
> the local stack (it's the strategy we want to compare against; the
> default KB already uses the smart `markdown_header` chunker). See the
> docstring in that file for why it's named this way.

### Module 03 — Strands agent

```bash
python modules/03-strands-agent/code/simple_agent.py        # KB retrieve only
python modules/03-strands-agent/code/agent_with_tools.py    # + classify_risk, get_period_info, check_grc_control
python modules/03-strands-agent/code/multi_agent.py         # router + 3 specialists
python modules/03-strands-agent/code/chat.py                # interactive
```

### Module 04 — Context Graph

```bash
python modules/04-context-graph/code/create_graph.py        # reset Neo4j + create vector index
python modules/04-context-graph/code/populate_graph.py      # entity extraction (~2 min on Atlas)
python modules/04-context-graph/code/query_graph.py "Why was the PO aging threshold raised?"
python modules/04-context-graph/code/add_decision_trace.py  # demo writing a DecisionTrace
python modules/04-context-graph/code/hybrid_retrieval.py "What metrics matter for month-end close and why?"
```

`hybrid_retrieval.py` is the showcase script — KB + graph fused with
Reciprocal Rank Fusion, then Atlas synthesis.

### Module 05 — Integrated agent

```bash
# Scripted demo: 4 queries that exercise all 4 tools
python modules/05-putting-it-together/code/integrated_agent.py

# Free-form chat
python modules/05-putting-it-together/code/interactive.py

# Inspect what the agent recorded
python modules/05-putting-it-together/code/show_decisions.py

# KB-only vs Graph-enhanced answer comparison
python modules/05-putting-it-together/code/evaluate.py
```

## Architecture

```
                     ┌─────────────────────────┐
                     │  Strands Agent          │
                     │  (Atlas Sonnet 4.6)     │
                     └─────┬───┬───┬───┬───────┘
                           │   │   │   │
        ┌──────────────────┘   │   │   └────────────────────┐
        │              ┌───────┘   └──────┐                 │
        ▼              ▼                  ▼                 ▼
   ┌─────────┐   ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐
   │ retrieve│   │search_context│  │record_decision │  │ingest_email_     │
   │ (KB)    │   │_graph        │  │                │  │decision          │
   └────┬────┘   └──────┬───────┘  └────────┬───────┘  └────────┬─────────┘
        │               │                   │                   │
        ▼               ▼                   ▼                   ▼
   ┌─────────────┐  ┌───────────────────────────────────────────────────┐
   │  Chroma     │  │             Neo4j 5.20 (Docker)                   │
   │  ~/.cache/  │  │  :Embeddable vector index (1024-dim, cosine)      │
   │  kgc-...    │  │  Documents, Decisions, DecisionTraces,            │
   │             │  │  Systems, Metrics, Namespaces, Policies, ...      │
   └─────┬───────┘  └─────────────┬─────────────────────────────────────┘
         │                        │
         └────────────────────────┴──── embeddings via LM Studio (Qwen3)
```

Every node and chunk is embedded by **Qwen3-Embedding-0.6B** running locally in
LM Studio. All LLM calls (extraction during `populate_graph`, synthesis at
query time) go to **Atlas Cloud** Claude Sonnet 4.6 over an OpenAI-compatible
endpoint.

## Cost & resource notes

- **Atlas Cloud**: full walkthrough end-to-end uses roughly 100k–300k input
  tokens + ~30k output. At Sonnet 4.6 rates that's well under $1.
- **LM Studio**: ~700 MB RAM while the embedding model is loaded.
- **Neo4j**: ~500 MB RAM idle, more under load. Default container has no
  persistent volume — data lives only as long as the container does. Add
  `-v neo4j-data:/data` if you want persistence across `docker rm`.
- **Chroma**: tens of MB on disk, persistent at
  `~/.cache/kgc-context-graph/chroma/`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop not running |
| `[Errno 61] Connection refused` to `127.0.0.1:1234` | LM Studio server not started, or model not loaded |
| `Authentication error` from Atlas | Bad/expired key in `.env` |
| Empty results from `evaluate_chunking.py` | Run the 3 `create_*_datasource.py` scripts first, or just re-run — it auto-ingests missing collections |
| `db.index.vector.queryNodes` not found | Neo4j version <5.13. Pull `neo4j:5.20` instead |
| Module 03 `retrieve` returns nothing | Run `modules/01-knowledge-base-setup/code/create_knowledge_base.py` first to populate the default KB |

## Layout of the local-stack additions

```
modules/local/
  config.py     # loads .env, exposes constants
  llm.py        # build_llm() → Strands OpenAIModel pointed at Atlas
  embed.py      # embed_texts() against LM Studio
  chunking.py   # 4 chunking strategies
  kb.py         # LocalKB wrapping Chroma
  graph.py      # LocalGraph wrapping Neo4j + vector index
  retrieve.py   # @tool that reads from the default LocalKB
```

Everything else is the original module-N script refactored to import from this
package instead of `boto3`.
