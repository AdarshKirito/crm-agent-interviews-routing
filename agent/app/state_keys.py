"""Session-state keys. Keys set by the benchmark adapter at session creation are marked (in)."""

ORG = "crm_org"  # (in) b2b | b2c
AUDIENCE = "crm_audience"  # (in) customer | employee  (the benchmark's customer-facing flag)
INTERACTIVE = "crm_interactive"  # (in) True when a simulated user is on the other side
TASK_CONTEXT = "crm_task_context"  # (in) task details the baseline also receives
EXTRA_CONTEXT = "crm_extra_context"  # (in) "aided" eval-mode domain notes

TURN = "crm_turn"
CONVERSATION = "crm_conversation"  # [{"role": "user"|"agent", "text": ...}]
DETAILS = "crm_details"  # details gathered so far in a multi-turn task
CLARIFICATIONS = "crm_clarifications"  # clarifying questions asked so far
NOTES = "crm_notes"  # compact tool results carried to later turns
EVIDENCE = "crm_evidence"  # text of every tool result (for the checker)
SELF_IDS = "crm_self_ids"  # the logged-in customer's own record ids
SCREEN = "crm_screen"  # request-screening signals
GUARD = "crm_guard"  # final guard decision
ROUTE = "crm_route"  # predicted task type and chosen model
DRAFT = "crm_draft"  # solver output
RETRIES = "crm_retries"
FEEDBACK = "crm_feedback"
CHECK = "crm_check"
FINAL = "crm_final_response"
USAGE = "crm_usage"
BLOCKED_TOOLS = "crm_blocked_tool_calls"
