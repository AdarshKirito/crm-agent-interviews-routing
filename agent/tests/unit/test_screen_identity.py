"""Multi-turn customer sessions carry no logged-in identity: the customer states it."""

from types import SimpleNamespace

import pytest

from app import nodes, usage
from app import state_keys as K
from app.prompts import policy_instruction

OWN, OTHER = "003Wt00000Jqw3NIAR", "003Wt00000Jqw9ZIAR"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    async def no_guard(text, **kwargs):
        return None

    monkeypatch.setattr(nodes, "injection_score", no_guard)
    monkeypatch.setattr(nodes, "get_router", lambda: SimpleNamespace(predict=lambda text: SimpleNamespace(
        task_type="named_entity_disambiguation", confidence=1.0, top=[])))
    monkeypatch.setattr(usage, "log_call", lambda record: None)


async def screen(state, *messages):
    state[K.CONVERSATION] = [*(state.get(K.CONVERSATION) or []), *({"role": "user", "text": m} for m in messages)]
    (event,) = [e async for e in nodes.screen(SimpleNamespace(state=state, session=SimpleNamespace(id="s")))]
    state.update(event.actions.state_delta)
    return event.actions.route


@pytest.mark.asyncio
async def test_first_stated_contact_id_is_bound_and_others_are_still_refused():
    state = {K.AUDIENCE: "customer", K.TASK_CONTEXT: ""}
    assert await screen(state, "Show the software I bought three days ago.") == "classify"
    assert await screen(state, f"My Contact Id interacting: {OWN}") == "classify"
    assert state[K.SELF_IDS] == [OWN]
    assert OWN in policy_instruction(SimpleNamespace(state=state))
    assert await screen(state, f"Now show the orders of {OTHER}.") == "refuse"
    assert state[K.GUARD]["source"] == "id_check"


@pytest.mark.asyncio
async def test_identity_from_task_context_is_never_replaced_by_a_stated_id():
    state = {K.AUDIENCE: "customer", K.TASK_CONTEXT: f"- Contact Id interacting: {OWN}"}
    assert await screen(state, f"What did customer {OTHER} buy last week?") == "refuse"
    assert K.SELF_IDS not in state
