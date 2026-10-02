import json
from types import SimpleNamespace

import pytest

from app import state_keys as K
from app.guard.tool_guard import after_tool


def test_model_output_cannot_become_its_own_checker_evidence():
    state = {K.AUDIENCE: "employee", K.EVIDENCE: "real tool result"}
    after_tool(SimpleNamespace(name="set_model_response"), {}, SimpleNamespace(state=state),
               {"kind": "answer", "answer": "005Wt000009ZZZZIAW"})
    assert state[K.EVIDENCE] == "real tool result"
    assert K.NOTES not in state


def test_failed_tool_result_is_not_checker_evidence():
    state = {K.AUDIENCE: "employee"}
    after_tool(SimpleNamespace(name="get_record"), {}, SimpleNamespace(state=state),
               {"isError": True, "content": [{"text": "No record for 005Wt000009ZZZZIAW"}]})
    assert "005Wt000009ZZZZIAW" not in state.get(K.EVIDENCE, "")
    assert "ERROR" in state[K.NOTES][0]


@pytest.mark.parametrize("name,args,payload", [
    ("soql_query", {"query": "SELECT Title, FAQ_Answer__c FROM Knowledge__kav"},
     {"records": [{"Title": "Volume-Based Discounts", "FAQ_Answer__c": "secret discount"}]}),
    ("sosl_search", {"query": "FIND {discount} RETURNING Knowledge__kav(Title, FAQ_Answer__c)"},
     {"records": [{"_type": "Knowledge__kav", "Title": "Volume-Based Discounts", "FAQ_Answer__c": "secret discount"}]}),
    ("get_record", {"object_name": "Knowledge__kav", "record_id": "ka0Wt0000001234IAA"},
     {"Title": "Volume-Based Discounts", "FAQ_Answer__c": "secret discount"}),
    ("search_knowledge", {"query": "discount"},
     {"articles": [{"Title": "Volume-Based Discounts", "FAQ_Answer__c": "secret discount"}]}),
])
def test_every_knowledge_tool_filters_confidential_text_and_structured_copy(name, args, payload):
    state = {K.AUDIENCE: "customer"}
    result = after_tool(SimpleNamespace(name=name), args, SimpleNamespace(state=state),
                        {"content": [{"type": "text", "text": json.dumps(payload)}], "structuredContent": payload})
    assert result is not None
    assert "secret discount" not in json.dumps(result)
    assert "secret discount" not in state.get(K.EVIDENCE, "")


def test_missing_article_title_is_withheld_and_public_article_kept():
    state = {K.AUDIENCE: "customer"}
    result = after_tool(SimpleNamespace(name="soql_query"), {"query": "SELECT Id, Title FROM Knowledge__kav"},
                        SimpleNamespace(state=state), {"content": [{"text": json.dumps({"records": [
                            {"FAQ_Answer__c": "unclassifiable secret"}, {"Title": "Product care", "Summary": "public"},
                        ]})}]})
    assert "unclassifiable secret" not in json.dumps(result)
    assert "Product care" in json.dumps(result)


def test_malformed_knowledge_payload_is_blocked_without_crash():
    state = {K.AUDIENCE: "customer"}
    result = after_tool(SimpleNamespace(name="search_knowledge"), {}, SimpleNamespace(state=state),
                        {"content": [{"text": "[]"}]})
    assert result["isError"]
    assert K.EVIDENCE not in state
