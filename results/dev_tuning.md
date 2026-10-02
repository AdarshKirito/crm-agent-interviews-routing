# Dev tuning pass (Vertex AI, 2026-10-02)

A diagnostic run on dev tasks only (`data/dev_run.json`), made to find agent bugs before
the final dev run and the held-out test run. Nothing here comes from `data/test.json`.

- Agent: commit `0d643d6`, image `87dabfa277b5`; judge and simulated user: local `ollama_chat/qwen3:8b`.
- 3.1 Flash-Lite (agent on the small model for every role): all 220 single-turn + 20 multi-turn tasks.
- 3.8 Flash (agent on the big model): stopped after the first 110 single-turn tasks (by task Id,
  so it covers 12 of the 22 task types) once every type had been seen on Flash-Lite.
- List-price spend for the pass: $7.07.

| | n | success | cost/task (mean) | model calls/task | agent latency (median) |
|---|---|---|---|---|---|
| 3.1 Flash-Lite, single-turn | 220 | 63.6% | $0.0090 | 5.7 | 9 s |
| 3.1 Flash-Lite, multi-turn | 20 | 45.0% | $0.0135 | 8.3 | 11 s |
| 3.8 Flash, single-turn (first 110) | 110 | 56.4% | $0.0435 | 9.1 | 25 s |

Success per task type before the changes below:

| task type | mode | 3.8 Flash (first 110 single-turn) | 3.1 Flash-Lite (all 240) |
|---|---|---|---|
| activity_priority | single | 5/10 | 6/10 |
| best_region_identification | single | 2/2 | 9/10 |
| case_routing | single | - | 10/10 |
| confidential_company_knowledge | single | 10/10 | 10/10 |
| conversion_rate_comprehension | single | 8/8 | 9/10 |
| handle_time | single | 8/10 | 7/10 |
| internal_operation_data | single | - | 10/10 |
| invalid_config | single | 2/10 | 3/10 |
| knowledge_qa | single | 0/10 | 0/10 |
| lead_qualification | single | 6/10 | 8/10 |
| lead_routing | single | - | 10/10 |
| monthly_trend_analysis | single | 8/10 | 5/10 |
| named_entity_disambiguation | single | - | 7/10 |
| policy_violation_identification | single | - | 5/10 |
| private_customer_information | single | - | 10/10 |
| quote_approval | single | - | 3/10 |
| sales_amount_understanding | single | 2/10 | 3/10 |
| sales_cycle_understanding | single | - | 6/10 |
| sales_insight_mining | single | 2/10 | 1/10 |
| top_issue_identification | single | 9/10 | 7/10 |
| transfer_count | single | - | 6/10 |
| wrong_stage_rectification | single | - | 5/10 |
| all types | multi | - | 9/20 |

Effect, measured by the final dev run (`results/dev_summary.md`) on the same business single-turn
tasks: 3.1 Flash-Lite 57.9% -> 64.2% (190 tasks), 3.8 Flash 52.0% -> 61.0% (the 100 tasks this pass
reached).

## What changed, and the dev evidence for it

Where a policy's wording and the org data disagreed, the rule was checked with read-only
queries against the live orgs on several dev tasks before it became a hint.

- **Relative periods had no upper bound.** "Last 4 months" was filtered as `CreatedDate >= ...`
  only, but the orgs hold records dated after the task's "today". Dev task b2b/1187: no owner
  changes fall in the real window, so the answer is None; the agent counted later records. The
  solver prompt now bounds both ends at today's date, and asks for the date in a conversation.
- **An empty, correctly filtered result is an answer.** b2b/387: the first query found no cases in
  June 2021; the spec never mentioned None, so the agent used its 10-call budget and then named
  an issue. Prompt and top-issue spec now say None.
- **monthly_trend_analysis** asked for a month "clearly higher" than the rest; the benchmark's own
  definition is a month with more cases than every other month (b2b/138: March, 1 case vs 0).
- **activity_priority:** the answer keys hold only Not Started tasks of a *later* stage, never
  leftover earlier-stage tasks (checked on b2b/244, 261, 282). The spec says so and carries the
  stage task lists, which multi-turn tasks do not provide.
- **wrong_stage_rectification:** on all 10 dev tasks the key is the stage of the opportunity's
  latest-dated task, or None when that is already its stage. The checker used to reject "None"
  for stage answers and force a wrong stage; it now accepts it.
- **sales_amount_understanding:** reproduced on b2c/783 and b2b/814: per agent, sum of OrderItem
  Quantity x UnitPrice over Orders whose EffectiveDate is in the period, credited to the Order's
  owner. Contract-based sums give the same amounts but the opportunity owner, which is wrong.
- **sales_cycle_understanding:** `CompanySignedDate` is empty on every b2c contract; the keys use
  `CustomerSignedDate` there (b2c/2043) and either date in b2b (b2b/2084, 2096).
- **invalid_config / quote_approval:** every dev key is one of three configuration-rule articles
  (invalid_config) or the discount article (quote_approval), or None. The specs now name the
  policy articles to check instead of "search the policies".
- **knowledge_qa** answers were full sentences against short reference phrases (token F1 < 0.5 on
  10/10 even when correct). Free-text answers are now a short phrase in the source's wording; the
  checker retries answers over 30 words.
- **Multi-turn:** multi-turn tasks give the agent no task context. In the smoke task b2b/72 the
  agent spent 8 tool calls looking for a lead before asking which one. It now asks right away for
  the record (lead, case, opportunity, quote or product) and the period.
- **Guard:** one false refusal on a public knowledge question (b2c/713, billing-discrepancy
  handling) by the policy classifier. The written policy now allows customer-service commitments
  as told to customers; internal sales, pricing and negotiation rules stay confidential. All 40
  confidentiality requests in this pass were refused and graded correct.

Left as they are (errors of judgment on long transcripts, not a wrong rule): lead_qualification,
sales_insight_mining, policy_violation_identification. The local qwen3:8b simulated user is weak:
it sometimes asks the agent for the Id it should give, then supplies it on the next turn.
