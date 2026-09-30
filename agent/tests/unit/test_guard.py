from app.guard.pii import analyze_request, scrub
from app.guard.prompt_guard import parse_score
from app.guard.sensitive import load_map, self_ids_from_context

SELF = "003Wt00000JqmLtIAJ"
SELF_ACCOUNT = "001Wt00000PFj4dIAD"
OTHER = "003Ws00000DYImHIAX"


def test_self_ids_from_both_context_formats():
    assert self_ids_from_context(f"The customer you are interacting with is logged in as Id: {SELF}.") == {SELF}
    assert self_ids_from_context(f"- Contact Id interacting: {SELF}\n- Today's date: 2022-03-2") == {SELF}


def test_customer_may_read_own_contact_and_orders():
    smap = load_map()
    own = {SELF, SELF_ACCOUNT}
    assert smap.check_soql(f"SELECT Id, AccountId FROM Contact WHERE Id = '{SELF}'", own).allowed
    assert smap.check_soql(
        f"SELECT Id, Product2Id, Product2.Name, Order.EffectiveDate FROM OrderItem WHERE Order.AccountId = '{SELF_ACCOUNT}'", own
    ).allowed
    # 15-character form of the customer's own id also counts
    assert smap.check_soql(f"SELECT Id FROM Contact WHERE Id = '{SELF[:15]}'", own).allowed


def test_customer_blocked_from_other_customers_and_internal_objects():
    smap = load_map()
    own = {SELF}
    v = smap.check_soql("SELECT Id, MailingCity FROM Contact WHERE Name = 'Ava Brown'", own)
    assert not v.allowed and v.category == "private_customer_information"
    v = smap.check_soql(f"SELECT Id FROM Order WHERE Account.Id IN (SELECT AccountId FROM Contact WHERE Id = '{OTHER}')", own)
    assert not v.allowed
    v = smap.check_soql(f"SELECT Id FROM Contact WHERE Id = '{SELF}' OR Name = 'Ying Liu'", own)
    assert not v.allowed
    v = smap.check_soql("SELECT OwnerId, COUNT(Id) FROM Case GROUP BY OwnerId", own)
    assert not v.allowed
    v = smap.check_soql("SELECT Id, Body__c FROM VoiceCallTranscript__c", own)
    assert not v.allowed and v.category == "internal_ops"
    v = smap.check_soql(f"SELECT Id, Owner.Name FROM Case WHERE ContactId = '{SELF}'", own)
    assert not v.allowed  # Owner -> User is internal


def test_customer_knowledge_and_product_queries_allowed():
    smap = load_map()
    assert smap.check_soql("SELECT Id, Title, Summary FROM Knowledge__kav WHERE Title LIKE '%battery%'", set()).allowed
    assert smap.check_soql("SELECT Id, Name FROM Product2 WHERE Name LIKE '%Designer%'", set()).allowed
    assert smap.check_sosl("FIND {battery care} IN ALL FIELDS RETURNING Knowledge__kav(Id, Title)").allowed
    assert not smap.check_sosl("FIND {Ava Brown} IN NAME FIELDS RETURNING Contact(Id, Name)").allowed


def test_get_record_scoping():
    smap = load_map()
    assert smap.check_get_record("Contact", SELF, ["Id", "Email"], {SELF}).allowed
    assert not smap.check_get_record("Contact", OTHER, None, {SELF}).allowed
    assert not smap.check_get_record("Quote", "0Q0Wt000001WRAzKAO", None, {SELF}).allowed


def test_confidential_articles_and_request_terms():
    smap = load_map()
    assert smap.is_confidential_article("Competitor: Quantum Circuits Inc")
    assert smap.is_confidential_article("Volume-Based Discounts")
    assert not smap.is_confidential_article("Enhancing Access to Online Training Modules")
    assert "confidential" in smap.request_signals("What is a noted weakness of CircuitWave Technologies?")


def test_presidio_request_and_scrub_keep_salesforce_ids():
    from app.guard import pii

    entities = analyze_request("Which city is listed for Ava Brown in our records?")
    if pii.ENGINE == "presidio":  # name detection needs spaCy (blocked on some Windows hosts)
        assert any(e["type"] == "PERSON" for e in entities)
    text, removed = scrub("Contact 003Wt00000JqmLtIAJ: email jane.doe@example.com, phone 415-555-0133")
    assert "003Wt00000JqmLtIAJ" in text
    assert "jane.doe@example.com" not in text and "415-555-0133" not in text
    assert set(removed) >= {"EMAIL_ADDRESS", "PHONE_NUMBER"}


def test_prompt_guard_score_parsing():
    assert parse_score("0.9995") == 0.9995
    assert parse_score("  1.2e-05 ") == 1.2e-05
    assert parse_score("MALICIOUS") == 1.0
    assert parse_score("BENIGN") == 0.0
    assert parse_score("") is None


def test_pattern_fallback_finds_contact_details_and_ids(monkeypatch):
    from app.guard import pii

    monkeypatch.setattr(pii, "_engines", lambda: (None, None))
    found = {e["type"] for e in pii.analyze_request("Email me at a.b@example.com about 003Ws00000DYTkrIAH")}
    assert found == {"EMAIL_ADDRESS", "SALESFORCE_ID"}
    text, removed = pii.scrub("Id 003Wt00000JqmLtIAJ, call +1 415-555-0133 or mail jane@example.org")
    assert text == "Id 003Wt00000JqmLtIAJ, call [REDACTED] or mail [REDACTED]"
    assert removed == ["EMAIL_ADDRESS", "PHONE_NUMBER"]


def test_policy_node_reads_last_json_verdict_and_stays_out_of_history():
    import asyncio
    from types import SimpleNamespace

    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    from app.nodes import make_policy_check

    class Reply:
        model = "groq/openai/gpt-oss-20b"

        def __init__(self, text):
            self.text, self.seen = text, []

        async def generate_content_async(self, request, stream=False):
            self.seen.append(request)
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text=self.text)]))

    ctx = SimpleNamespace(state={"crm_conversation": [{"role": "user", "text": "What is a weakness of CircuitWave?"}]})
    for text in ['{"decision":"refuse","category":"confidential_company_knowledge","rationale":"competitor analysis"}',
                 'We need a GuardVerdict. {"decision": "refuse", "category": "confidential_company_knowledge", "rationale": "x"}']:
        llm = Reply(text)
        verdict = asyncio.run(make_policy_check(llm, llm.model)(ctx))
        assert verdict["decision"] == "refuse" and verdict["category"] == "confidential_company_knowledge"
        assert llm.seen[0].config.response_schema is not None
    assert asyncio.run(make_policy_check(Reply("no json here"), "m")(ctx)) == {}
    assert ctx.state["crm_calls"][-1]["component"] == "policy_check"
