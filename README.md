# crmroute

A guarded, cost-routed Google ADK agent for CRM questions, evaluated on
[CRMArena-Pro](https://github.com/SalesforceAIResearch/CRMArena) (Salesforce AI Research).
It answers internal analytics questions for employees, answers customers' own-account and
product questions, and refuses requests for private, internal or confidential data.

## Measured so far

### Published baselines, recomputed from CRMArena's released B2B single-turn runs (`results/published_b2b_single_turn.md`)

| model | business tasks success [95% CI] (n=1880) | confidentiality refusal rate (n=60) | cost/task |
|---|---|---|---|
| o1 | 45.5% [43.2, 47.8] | 1.7% | $0.381 |
| gpt-4o | 27.3% [25.3, 29.4] | 0.0% | $0.057 |
| gpt-4o-mini | 20.4% [18.7, 22.3] | 0.0% | $0.005 |

The benchmark's agent (without its privacy prompt) almost never refuses confidential requests. That gap is what the guard targets. How these numbers were computed:

- Success = reward 1; fuzzy tasks count as a success at token F1 ≥ 0.5.
- They come from the released files and are not taken from the paper's tables.
