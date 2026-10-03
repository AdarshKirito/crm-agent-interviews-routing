# Scoping doc

> **Status: not conducted.** No stakeholder interviews have been held. Everything below
> is the unfilled template; the rows in *italics* are examples, not findings.

One page. Fill it in after the three interviews; everything in *italics* is a prompt to replace.

## Who I talked to
| # | Role | Team | Date | Uses the CRM for |
|---|---|---|---|---|
| 1 | *e.g. support team lead* | | | |
| 2 | *e.g. sales ops analyst* | | | |
| 3 | *e.g. account executive* | | | |

## Interview script (30 min)
1. What are the three CRM questions you answer most often? How long does each take?
2. **Walk me through the last one you did by hand.** (screens, reports, exports, who you asked)
3. When an answer is wrong, how do you find out, and what does it cost?
4. What must never be shown to a customer, or to someone outside your team?
5. When would you trust an assistant's answer without checking it? When never?
6. What would make you stop using it?

## Top tasks -> benchmark task types
| Their task (their words) | Frequency | CRMArena-Pro task type(s) | Notes |
|---|---|---|---|
| *"Who should take this new case?"* | *daily* | `case_routing` | |
| | | | |
| | | | |

Task types available: handle_time, transfer_count, best_region_identification,
case_routing, named_entity_disambiguation, policy_violation_identification,
top_issue_identification, monthly_trend_analysis, knowledge_qa,
sales_amount_understanding, lead_routing, sales_cycle_understanding,
conversion_rate_comprehension, wrong_stage_rectification, sales_insight_mining,
quote_approval, lead_qualification, activity_priority, invalid_config.

## Never-share list -> sensitive-field map
Turn every item into `Object.Field -> class` (pii, internal_ops, confidential) and
copy it into `agent/app/guard/sensitive_fields.yaml` (`fields:` / `objects:` /
`confidential_article_titles:`). Then rerun `agent` unit tests.

| What they said | Object.Field (or object / article) | Class |
|---|---|---|
| *"Customers can't see each other's addresses"* | `Contact.MailingStreet`, `Account.ShippingStreet` | pii |
| | | |

## Three requirements that changed the design
1. *Requirement — what changed in the build, and where (file/commit).*
2.
3.

## Out of scope
- Writing to the CRM (all tools are read-only).
- *...*
