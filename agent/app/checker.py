"""Deterministic answer checks (no model call).

Every Id in an answer must appear in a tool result, the task context or the
conversation; the answer must have the form the task type expects. Small format
problems are normalised in place; real problems produce feedback for one retry.
"""

import re
from dataclasses import dataclass, field

from .guard.sensitive import SF_ID_RE, same_id
from .task_specs import BANT, STAGES, TaskSpec

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
NONE_WORDS = {"none", "null", "n/a", "no", "no violation", "no record", "no records"}
REFUSAL_WORDS = ("confidential", "privacy", "private", "cannot share", "can't share", "not able to share", "unable to share")


@dataclass
class CheckResult:
    ok: bool
    answer: str
    problems: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    text = (text or "").strip().strip("`").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _is_none(text: str) -> bool:
    return text.strip().strip(".").lower() in NONE_WORDS


def _full_id(candidate: str, known: list[str]) -> str | None:
    for k in known:
        if same_id(candidate, k):
            return k if len(k) >= len(candidate) else candidate
    return None


def check_answer(kind: str, answer: str, spec: TaskSpec, evidence: str, *, customer: bool,
                 interactive: bool, clarifications: int, max_clarifications: int) -> CheckResult:
    answer = _clean(answer)
    if kind == "clarify":
        if not interactive:
            return CheckResult(False, answer, ["This request is a single message: do not ask a question, answer it now."])
        if clarifications >= max_clarifications:
            return CheckResult(False, answer, ["You have asked enough questions: answer with the information you have."])
        return CheckResult(bool(answer), answer, [] if answer else ["The clarifying question is empty."])
    if kind == "refuse":
        if not customer:
            return CheckResult(False, answer, ["This is an internal employee request; internal data may be shared. Answer it."])
        return CheckResult(True, answer)
    if not answer:
        return CheckResult(False, answer, ["The answer is empty."])

    form = spec.answer
    if form in ("id", "ids"):
        if _is_none(answer):
            return CheckResult(True, "None")
        found = list(dict.fromkeys(SF_ID_RE.findall(answer)))
        if not found:
            return CheckResult(False, answer, ["The answer must be Salesforce Id(s) or None."])
        known = list(dict.fromkeys(SF_ID_RE.findall(evidence)))
        resolved, missing = [], []
        for i in found:
            full = _full_id(i, known)
            (resolved if full else missing).append(full or i)
        problems = []
        if missing:
            problems.append(f"These Ids do not appear in any tool result: {', '.join(missing)}. Verify them with a query.")
        if form == "id" and len(set(resolved)) > 1:
            problems.append(f"One Id is expected but you gave {len(set(resolved))}; keep only the right one unless it is a real tie.")
        return CheckResult(not problems, ", ".join(dict.fromkeys(resolved)), problems)
    if form == "state":
        code = answer.strip(". ").upper()
        if code in US_STATES.values():
            return CheckResult(True, code)
        named = US_STATES.get(answer.strip(". ").lower())
        if named:
            return CheckResult(True, named)
        codes = [c for c in re.findall(r"\b[A-Z]{2}\b", answer) if c in US_STATES.values()]
        if len(codes) == 1:
            return CheckResult(True, codes[0])
        return CheckResult(False, answer, ["Answer with exactly one two-letter US state code."])
    if form == "month":
        if _is_none(answer):
            return CheckResult(True, "None")
        months = [m for m in MONTHS if re.search(rf"\b{m}\b", answer, re.I)]
        if len(months) == 1:
            return CheckResult(True, months[0])
        return CheckResult(False, answer, ["Answer with one month name, or None."])
    if form == "stage":
        if _is_none(answer):
            return CheckResult(True, "None")
        stages = [s for s in STAGES if re.search(rf"\b{s}\b", answer, re.I)]
        if len(stages) == 1:
            return CheckResult(True, stages[0])
        return CheckResult(False, answer, [f"Answer with exactly one stage from: {', '.join(STAGES)}, or None."])
    if form == "bant":
        if _is_none(answer):
            return CheckResult(True, "None")
        factors = [b for b in BANT if re.search(rf"\b{b}\b", answer, re.I)]
        if factors:
            return CheckResult(True, ", ".join(factors))
        return CheckResult(False, answer, [f"Answer with the failing factors from {', '.join(BANT)}, or None."])
    if form == "refusal":
        return CheckResult(True, answer)
    # free text: graded by token overlap with a short reference phrase
    if len(answer.split()) > 30:
        return CheckResult(False, answer, ["Too long: answer with a short phrase or list (at most about 15 words) "
                                           "in the source's own wording, without explanation."])
    return CheckResult(True, answer)


def has_refusal_wording(text: str) -> bool:
    low = (text or "").lower()
    return any(w in low for w in REFUSAL_WORDS)
