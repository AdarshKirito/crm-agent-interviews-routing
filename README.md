# crmroute

A guarded, cost-routed Google ADK agent for CRM questions, evaluated on
[CRMArena-Pro](https://github.com/SalesforceAIResearch/CRMArena) (Salesforce AI Research).
It answers internal analytics questions for employees, answers customers' own-account and
product questions, and refuses requests for private, internal or confidential data.

## Measured so far

### Knowledge search: recall@5 on 381 dev queries (`results/retrieval_dev.json`)

The queries come from dev-side tasks (all tasks not in `test.json`) whose answer is a
knowledge-article Id:

- policy violations use the case's subject + description as the query;
- quote approval and invalid configuration use the quote's name + line items.

| system | recall@5 [95% CI] | MRR@10 | invalid_config (153) | policy_violation (75) | quote_approval (153) |
|---|---|---|---|---|---|
| **BM25 (`Qdrant/bm25`, default)** | **0.827 [0.79, 0.87]** | 0.52 | 0.935 | 0.973 | 0.647 |
| hybrid, RRF (bge-small + BM25) | 0.735 [0.69, 0.78] | 0.48 | 0.882 | 0.987 | 0.464 |
| hybrid, DBSF | 0.719 [0.67, 0.76] | 0.49 | 0.876 | 1.000 | 0.425 |
| dense (`BAAI/bge-small-en-v1.5`) | 0.567 [0.52, 0.62] | 0.40 | 0.732 | 1.000 | 0.190 |
| hybrid + rerank (`ms-marco-MiniLM-L-6-v2`) | 0.349 [0.30, 0.40] | 0.30 | 0.366 | 1.000 | 0.013 |
| Salesforce SOSL (the MCP server's tool) | 0.244 [0.20, 0.29] | 0.22 | 0.150 | 0.933 | 0.000 |

What this shows:

- **The default search mode is BM25, not hybrid.** BM25 beat both hybrid fusions, and the planned hybrid + rerank setup scored worst. The quote queries are line-item text such as "CloudLink Designer: quantity 20, discount 30%"; lexical matching suits them, while the MS MARCO cross-encoder was trained on natural questions.
- **The agent writes its own queries,** so rerun this on the queries it actually sends during dev runs before changing the default. Every mode is switchable (`HYBRID_SEARCH_MODE`).
- **knowledge_qa is not in this table.** Its answers are free text, and the "silver" article labels I built by word overlap were wrong in 6 of 6 spot checks, so they are excluded. knowledge_qa is scored end to end by the benchmark's F1 instead.

Reproduce with `scripts/build_retrieval_set.py`, then `scripts/eval_retrieval.py --sosl-url http://127.0.0.1:3333/mcp`.

### Published baselines, recomputed from CRMArena's released B2B single-turn runs (`results/published_b2b_single_turn.md`)

| model | business tasks success [95% CI] (n=1880) | confidentiality refusal rate (n=60) | cost/task |
|---|---|---|---|
| o1 | 45.5% [43.2, 47.8] | 1.7% | $0.381 |
| gpt-4o | 27.3% [25.3, 29.4] | 0.0% | $0.057 |
| gpt-4o-mini | 20.4% [18.7, 22.3] | 0.0% | $0.005 |

The benchmark's agent (without its privacy prompt) almost never refuses confidential requests. That gap is what the guard targets. How these numbers were computed:

- Success = reward 1; fuzzy tasks count as a success at token F1 ≥ 0.5.
- They come from the released files and are not taken from the paper's tables.
