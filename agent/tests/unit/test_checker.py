from app.checker import check_answer
from app.task_specs import spec_for

EVIDENCE = '{"records":[{"OwnerId":"005Wt000003NDqDIAW","n":4},{"OwnerId":"005Wt000003NBcAIAW","n":2}]}'


def run(kind, answer, task, evidence=EVIDENCE, customer=False, interactive=False, asked=0):
    return check_answer(kind, answer, spec_for(task), evidence, customer=customer, interactive=interactive,
                        clarifications=asked, max_clarifications=3)


def test_id_answer_is_verified_and_expanded_to_18_chars():
    r = run("answer", "The agent is 005Wt000003NDqD.", "handle_time")
    assert r.ok and r.answer == "005Wt000003NDqDIAW"


def test_unverified_id_fails():
    r = run("answer", "005Wt000009ZZZZIAW", "handle_time")
    assert not r.ok and "do not appear" in r.problems[0]


def test_multiple_ids_for_single_id_task_fails_but_ids_task_passes():
    assert not run("answer", "005Wt000003NDqDIAW, 005Wt000003NBcAIAW", "handle_time").ok
    assert run("answer", "005Wt000003NDqDIAW, 005Wt000003NBcAIAW", "activity_priority").ok


def test_none_answers():
    assert run("answer", "None", "policy_violation_identification").answer == "None"
    assert run("answer", "none.", "monthly_trend_analysis").answer == "None"


def test_state_month_stage_bant_normalisation():
    assert run("answer", "Michigan", "best_region_identification").answer == "MI"
    assert run("answer", "MI", "best_region_identification").answer == "MI"
    assert run("answer", "It is november", "monthly_trend_analysis").answer == "November"
    assert run("answer", "negotiation", "wrong_stage_rectification").answer == "Negotiation"
    assert run("answer", "authority and budget", "lead_qualification").answer == "Budget, Authority"
    assert not run("answer", "Proposal", "wrong_stage_rectification").ok


def test_clarify_rules():
    assert not run("clarify", "Which quarter?", "handle_time", interactive=False).ok
    assert run("clarify", "Which quarter?", "handle_time", interactive=True).ok
    assert not run("clarify", "Which quarter?", "handle_time", interactive=True, asked=3).ok


def test_refusal_only_for_customers():
    assert not run("refuse", "I cannot share that.", "handle_time", customer=False).ok
    assert run("refuse", "That is confidential.", "internal_operation_data", customer=True).ok


def test_long_free_text_fails():
    assert not run("answer", "word " * 90, "knowledge_qa").ok
    # a full explanatory sentence is too long for a short-phrase reference
    assert not run("answer", "word " * 31, "knowledge_qa").ok
    assert run("answer", "Because breaches could be catastrophic.", "knowledge_qa").ok
    assert run("answer", "Flexible financing options, customized payment plans, leasing options, and extended warranties.",
               "knowledge_qa").ok


def test_stage_answer_may_be_none_when_current_stage_is_right():
    assert run("answer", "None", "wrong_stage_rectification").answer == "None"
    r = run("answer", "It should be negotiation.", "wrong_stage_rectification")
    assert r.ok and r.answer == "Negotiation"
    assert not run("answer", "Quote or Negotiation", "wrong_stage_rectification").ok
