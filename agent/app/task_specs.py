"""What each CRMArena-Pro task type needs: answer form, details to collect, and hints.

`answer` drives the checker's format rules:
  id / ids   Salesforce Id(s), comma-separated, or None
  state      two-letter US state code
  month      English month name, or None
  stage      one of STAGES
  bant       one or more of BANT, or None
  text       one short sentence or phrase list
`details` are what a multi-turn user may leave out and the agent should ask for.
"""

from dataclasses import dataclass, field

STAGES = ("Qualification", "Discovery", "Quote", "Negotiation", "Closed")
BANT = ("Budget", "Authority", "Need", "Timeline")
CONFIDENTIALITY_TYPES = ("private_customer_information", "internal_operation_data", "confidential_company_knowledge")


@dataclass(frozen=True)
class TaskSpec:
    answer: str
    goal: str
    details: tuple[str, ...] = ()
    hints: tuple[str, ...] = field(default_factory=tuple)


_AGENT_METRIC_DETAILS = ("time period", "whether the best (lowest/highest) value is wanted", "any minimum-case filter")
# The benchmark's activity-priority guide, shortened; multi-turn tasks give the agent no task context.
_STAGE_TASKS = ("Stage task lists: Qualification - prospect research, introductory emails or calls, scheduling discovery "
                "meetings; Discovery - industry/competitor research, pain points, needs conversations, solution alignment; "
                "Quote - detailed follow-ups, tailored proposals, needs analysis, demos or trials, case studies; "
                "Negotiation - negotiation meetings on terms and pricing, objections, contract preparation and review; "
                "Closed - kick-off or onboarding, win/loss analysis, recording final contract details, internal reviews "
                "of the deal, upsell or nurturing.")

SPECS: dict[str, TaskSpec] = {
    "handle_time": TaskSpec(
        "id", "the User Id of the agent with the lowest/highest average case handle time in a period",
        _AGENT_METRIC_DETAILS,
        ("Handle time = ClosedDate - CreatedDate of closed cases; follow the policy in the task context for which agent a case counts toward.",
         "Use CaseHistory__c to see owner changes when the policy counts first and transferred agents."),
    ),
    "transfer_count": TaskSpec(
        "id", "the User Id of the agent with the lowest/highest number of case transfers in a period",
        _AGENT_METRIC_DETAILS,
        ("A transfer is an owner change recorded in CaseHistory__c; each A->B transfer counts for agent A.",),
    ),
    "best_region_identification": TaskSpec(
        "state", "the two-letter state with the quickest average case closure time in a period",
        ("time period",),
        ("Closure time = ClosedDate - CreatedDate; the state comes from the account's shipping state.",),
    ),
    "case_routing": TaskSpec(
        "id", "the User Id of the best agent for a new case under the case routing policy",
        ("the case subject and description",),
        ("Match the case to the most similar Issue__c and Product2, then count closed cases per agent as the policy says.",),
    ),
    "named_entity_disambiguation": TaskSpec(
        "id", "the Id of the product (Product2) the customer refers to among their own past purchases",
        ("which product or purchase the customer means", "roughly when it was bought"),
        ("Look only at the logged-in contact's own account orders (Order/OrderItem), using the date hints.",),
    ),
    "policy_violation_identification": TaskSpec(
        "id", "the Id of the knowledge article the agent violated in a case, or None",
        ("the case",),
        ("Read the case's chat transcript / emails, then search knowledge articles for the policy that applies.",),
    ),
    "top_issue_identification": TaskSpec(
        "id", "the Issue__c Id reported most often for a product in a period, or None if the product has no cases then",
        ("the product", "time period"),
        ("Count cases per IssueId__c for cases whose order item is for that product, within the period.",
         "If no case of that product falls in the period, the answer is None."),
    ),
    "monthly_trend_analysis": TaskSpec(
        "month", "the month with more cases of a product than every other month in the period, or None",
        ("the product", "time period"),
        ("Count the product's cases per calendar month inside the period only (it ends on today's date).",
         "A month counts if its total is larger than each other month's, even by one case (1 vs 0); "
         "answer None only if the top count is tied or there are no cases."),
    ),
    "knowledge_qa": TaskSpec(
        "text", "a short answer to a question, grounded in the knowledge articles",
        ("the question",),
        ("Search knowledge articles and answer with the phrase from the article that answers the question, "
         "kept to its key words (about 5-15 words).",),
    ),
    "sales_amount_understanding": TaskSpec(
        "id", "the User Id of the agent with the highest/lowest sales amount in a period",
        _AGENT_METRIC_DETAILS,
        ("Sales amount = sum of OrderItem Quantity * UnitPrice over Orders whose EffectiveDate (the contract's "
         "company-signed date) falls in the period, credited to the Order's OwnerId (not the opportunity owner).",),
    ),
    "lead_routing": TaskSpec(
        "id", "the User Id of the agent a new lead should be assigned to under the lead routing policy",
        ("the lead's region (state)",),
        ("Match the lead's region to Territory2 descriptions, then apply the policy's tie-breakers.",),
    ),
    "sales_cycle_understanding": TaskSpec(
        "id", "the User Id of the agent with the quickest/slowest average sales cycle in a period",
        _AGENT_METRIC_DETAILS,
        ("Sales cycle = days from Opportunity CreatedDate to its contract's (ContractId__c) CompanySignedDate, "
         "or CustomerSignedDate where CompanySignedDate is empty; average per Opportunity OwnerId over "
         "opportunities whose contract was signed in the period.",),
    ),
    "conversion_rate_comprehension": TaskSpec(
        "id", "the User Id of the agent with the highest/lowest lead conversion rate in a period",
        _AGENT_METRIC_DETAILS,
        ("Conversion rate = converted leads / leads created in the window, per owner.",),
    ),
    "wrong_stage_rectification": TaskSpec(
        "stage", "the stage an opportunity should be in, given its tasks, or None if its current stage is right",
        ("the opportunity",),
        ("The right stage is the stage of the opportunity's most recent Task (latest ActivityDate); answer None "
         "if that is already its current StageName.",
         _STAGE_TASKS),
    ),
    "sales_insight_mining": TaskSpec(
        "text", "the sales-discussion subtopics that match the question for an opportunity",
        ("the opportunity",),
        ("Read the opportunity's VoiceCallTranscript__c records and name the subtopics or competitors in a "
         "comma-separated list, written as they appear in the transcripts.",),
    ),
    "quote_approval": TaskSpec(
        "id", "the Id of the knowledge article a quote violates, or None",
        ("the quote",),
        ("Quote approval is decided by the pricing rules: read the quote's line items (quantity, discount, price) and "
         "check every line item's discount against 'Volume-Based Discounts' (and 'Competing Offers' if the quote "
         "cites a competing offer). Product-configuration rules are a different check. Answer None if every "
         "discount is allowed.",),
    ),
    "lead_qualification": TaskSpec(
        "bant", "the BANT factors the lead fails, or None",
        ("the lead",),
        ("Read the lead's VoiceCallTranscript__c records and the qualification articles; list only unmet factors.",),
    ),
    "activity_priority": TaskSpec(
        "ids", "the Ids of 'Not Started' tasks that belong to a later stage than the opportunity's current stage",
        ("the opportunity",),
        ("List the opportunity's Not Started Task records and match each to a stage.",
         "Only tasks of a later stage are mismatched (e.g. contract or kick-off tasks while still in Discovery); "
         "leftover tasks of the current or earlier stages still match.",
         _STAGE_TASKS),
    ),
    "invalid_config": TaskSpec(
        "id", "the Id of the knowledge article a quote's product setup violates, or None",
        ("the quote",),
        ("Read the quote's line items (product, quantity), then the configuration rule articles 'Product Quantity "
         "Limits', 'Product Exclusion Constraints' and 'Mandatory Bundles for Quotes', and check each item against all three.",),
    ),
}
for _t in CONFIDENTIALITY_TYPES:
    SPECS[_t] = TaskSpec("refusal", "a refusal: this request asks for information that must not be shared")

UNKNOWN = TaskSpec("text", "whatever the user asked for", ("what exactly the user needs",))


def spec_for(task_type: str | None) -> TaskSpec:
    return SPECS.get(task_type or "", UNKNOWN)
