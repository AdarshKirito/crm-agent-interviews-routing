"""Sensitive-data map (sensitive_fields.yaml) and the checks built on it."""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

MAP_PATH = Path(__file__).with_name("sensitive_fields.yaml")

# Salesforce record Ids: 15/18 chars, standard key prefixes start with a digit,
# custom objects with "a", knowledge articles with "ka".
SF_ID_RE = re.compile(r"\b(?:[0-9][A-Za-z0-9]{2}|a[0-9][A-Za-z0-9]|k[aA][0-9A-Za-z])[A-Za-z0-9]{12}(?:[A-Za-z0-9]{3})?\b")
PERSON_ID_PREFIXES = ("003", "001", "00Q")  # Contact, Account, Lead
SELF_ID_PATTERNS = (
    re.compile(r"logged in as Id:\s*([A-Za-z0-9]{15,18})"),
    re.compile(r"Contact Id interacting:\s*([A-Za-z0-9]{15,18})"),
)
_STRING_LITERAL = re.compile(r"'(?:\\'|[^'])*'")
_SELECT = re.compile(r"\bSELECT\s+(.*?)\s+FROM\s+([A-Za-z_]\w*)", re.I | re.S)
_FROM = re.compile(r"\bFROM\s+([A-Za-z_]\w*)", re.I)
_RETURNING = re.compile(r"\bRETURNING\s+(.+)$", re.I | re.S)
_SOQL_WORDS = {
    "COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX", "DISTINCT", "TYPEOF", "WHEN", "THEN", "ELSE", "END",
    "FORMAT", "TOLABEL", "CONVERTCURRENCY", "CALENDAR_MONTH", "CALENDAR_YEAR", "CALENDAR_QUARTER", "DAY_ONLY",
    "HOUR_IN_DAY", "FISCAL_YEAR", "FISCAL_QUARTER", "WEEK_IN_YEAR", "DAY_IN_MONTH", "SELECT", "AS", "GROUPING",
}
_CHILD_RELATIONSHIPS = {"Orders": "Order", "OrderItems": "OrderItem", "Cases": "Case", "Contacts": "Contact"}


def ids_in(text: str) -> set[str]:
    return set(SF_ID_RE.findall(text or ""))


def self_ids_from_context(context: str) -> set[str]:
    found = set()
    for pattern in SELF_ID_PATTERNS:
        found.update(pattern.findall(context or ""))
    return found


def same_id(a: str, b: str) -> bool:
    """15- and 18-character forms of one Id share the first 15 characters."""
    return a[:15] == b[:15]


def in_ids(value: str, ids: set[str]) -> bool:
    return any(same_id(value, i) for i in ids)


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""
    category: str = ""


class SensitiveMap:
    def __init__(self, data: dict):
        self.allowed_objects = set(data.get("customer_allowed_objects") or [])
        self.scoped_objects = set(data.get("customer_scoped_objects") or [])
        self.relationships = {**_CHILD_RELATIONSHIPS, **(data.get("relationships") or {})}
        self.object_class = dict(data.get("objects") or {})
        self.field_class = dict(data.get("fields") or {})
        self.article_patterns = [re.compile(p, re.I) for p in data.get("confidential_article_titles") or []]
        self.request_terms = {c: [t.lower() for t in terms] for c, terms in (data.get("request_terms") or {}).items()}

    # ---- requests -------------------------------------------------------------------------
    def request_signals(self, text: str) -> dict[str, list[str]]:
        low = (text or "").lower()
        return {c: hits for c, terms in self.request_terms.items() if (hits := [t for t in terms if t in low])}

    def is_confidential_article(self, title: str | None) -> bool:
        return any(p.search(title or "") for p in self.article_patterns)

    # ---- tool calls (customer-facing sessions only) -----------------------------------------
    def _resolve(self, base: str, path: str) -> tuple[str, str]:
        """Resolve a SOQL field path relative to `base` to (object, field)."""
        parts = path.split(".")
        obj = base
        for rel in parts[:-1]:
            obj = self.relationships.get(rel, rel)
        return obj, parts[-1]

    def _check_refs(self, refs: list[tuple[str, str]], objects: set[str], text: str, self_ids: set[str]) -> Verdict:
        for obj in sorted(objects):
            if obj not in self.allowed_objects:
                cat = self.object_class.get(obj, "internal_ops")
                return Verdict(False, f"{obj} records are internal and cannot be used for a customer.", cat)
        scoped = objects & self.scoped_objects
        if scoped:
            mentioned = ids_in(text)
            if not any(in_ids(i, self_ids) for i in mentioned):
                return Verdict(False, f"{', '.join(sorted(scoped))} data must be limited to the logged-in customer's own records "
                                      "(filter by their contact or account Id).", "private_customer_information")
            others = [i for i in mentioned if i.startswith(PERSON_ID_PREFIXES) and not in_ids(i, self_ids)]
            if others:
                return Verdict(False, f"Id {others[0]} belongs to another customer.", "private_customer_information")
            if re.search(r"\bOR\b", _STRING_LITERAL.sub("''", text), re.I):
                return Verdict(False, "Queries on customer records may not use OR conditions in a customer session.",
                               "private_customer_information")
        for obj, field in refs:
            cls = self.field_class.get(f"{obj}.{field}")
            if cls == "internal_ops" or cls == "confidential":
                return Verdict(False, f"{obj}.{field} is internal data.", "internal_operation_data")
        return Verdict(True)

    def check_soql(self, query: str, self_ids: set[str]) -> Verdict:
        bare = _STRING_LITERAL.sub("''", query)
        refs: list[tuple[str, str]] = []
        objects = {self.relationships.get(o, o) for o in _FROM.findall(bare)}
        for select_list, base in _SELECT.findall(bare):
            base = self.relationships.get(base, base)
            for token in re.findall(r"[A-Za-z_][\w.]*", select_list):
                if token.upper() in _SOQL_WORDS:
                    continue
                obj, field = self._resolve(base, token)
                objects.add(obj)
                refs.append((obj, field))
        return self._check_refs(refs, objects, query, self_ids)

    def check_sosl(self, query: str) -> Verdict:
        m = _RETURNING.search(_STRING_LITERAL.sub("''", query))
        returning = re.sub(r"\([^)]*\)", "", m.group(1)) if m else ""
        returning = re.split(r"\b(?:LIMIT|WITH|UPDATE)\b", returning, flags=re.I)[0]
        objects = {part.strip().split()[0] for part in returning.split(",") if part.strip()}
        if not objects:
            return Verdict(False, "Text searches in a customer session must name the objects to return.", "private_customer_information")
        blocked = sorted(o for o in objects if o not in self.allowed_objects or o in self.scoped_objects)
        if blocked:
            return Verdict(False, f"Text search over {', '.join(blocked)} is not allowed in a customer session.",
                           "private_customer_information" if set(blocked) & self.scoped_objects else "internal_operation_data")
        return Verdict(True)

    def check_get_record(self, object_name: str, record_id: str, fields: list[str] | None, self_ids: set[str]) -> Verdict:
        refs = [(object_name, f) for f in (fields or [])]
        return self._check_refs(refs, {object_name}, record_id, self_ids)


@lru_cache(maxsize=1)
def load_map() -> SensitiveMap:
    return SensitiveMap(yaml.safe_load(MAP_PATH.read_text(encoding="utf-8")))
