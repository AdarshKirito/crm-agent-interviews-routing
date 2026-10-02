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
# Positive filters that actually constrain an object's records to the customer.
# Merely mentioning an owned Id (e.g. in Description or Id != ...) is not scope.
_OWNERSHIP_FIELDS = {
    "Contact": {"Id": "003", "AccountId": "001", "Account.Id": "001"},
    "Account": {"Id": "001"},
    "Order": {"Id": "801", "AccountId": "001", "Account.Id": "001"},
    "OrderItem": {"Id": "802", "OrderId": "801", "Order.Id": "801",
                  "Order.AccountId": "001", "Order.Account.Id": "001"},
    "Case": {"Id": "500", "ContactId": "003", "Contact.Id": "003",
             "AccountId": "001", "Account.Id": "001"},
}


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
        if re.search(r"\bFIELDS\s*\(", bare, re.I):
            return Verdict(False, "Customer queries must name their fields explicitly.", "internal_operation_data")
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
        verdict = self._check_refs(refs, objects, query, self_ids)
        if not verdict.allowed or not objects & self.scoped_objects:
            return verdict
        # Fail closed for nested queries: an inner owned Id does not prove the
        # outer query is scoped. The solver can make explicit scoped calls instead.
        if len(re.findall(r"\bSELECT\b", bare, re.I)) != 1:
            return Verdict(False, "Use separate customer-scoped queries instead of nested queries.",
                           "private_customer_information")
        base = next(iter(_FROM.findall(bare)), "")
        if not self._has_ownership_filter(query, base, self_ids):
            return Verdict(False, "Use a positive Id, ContactId, or AccountId filter on the customer's own records.",
                           "private_customer_information")
        return verdict

    def _has_ownership_filter(self, query: str, base: str, self_ids: set[str]) -> bool:
        # Blank quoted values without changing offsets; locate the actual WHERE,
        # never a word supplied inside a string literal.
        bare = _STRING_LITERAL.sub(lambda match: " " * len(match.group()), query)
        where = re.search(r"\bWHERE\b", bare, re.I)
        if not where or re.search(r"\b(?:OR|NOT)\b", bare[where.end():], re.I):
            return False
        suffix = re.search(r"\b(?:GROUP\s+BY|ORDER\s+BY|LIMIT|OFFSET|WITH)\b", bare[where.end():], re.I)
        end = where.end() + suffix.start() if suffix else len(query)
        clause = query[where.end():end]
        fields = {field.lower(): prefix for field, prefix in _OWNERSHIP_FIELDS.get(base, {}).items()}
        for match in re.finditer(r"(?<![\w.])([A-Za-z_]\w*(?:\.\w+)*)\s*(?:=\s*('[^']*')|\bIN\s*\(([^()]*)\))", clause, re.I):
            prefix = fields.get(match.group(1).lower())
            values = match.group(2) or match.group(3) or ""
            if not prefix or not re.fullmatch(r"\s*'[A-Za-z0-9]{15,18}'(?:\s*,\s*'[A-Za-z0-9]{15,18}')*\s*", values):
                continue
            ids = ids_in(values)
            if ids and all(value.startswith(prefix) and in_ids(value, self_ids) for value in ids):
                return True
        return False

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
        if not fields and any(key.startswith(f"{object_name}.") and cls in ("internal_ops", "confidential")
                              for key, cls in self.field_class.items()):
            return Verdict(False, f"Name explicit safe fields when reading {object_name}; all fields include internal data.",
                           "internal_operation_data")
        refs = [self._resolve(object_name, f) for f in (fields or [])]
        return self._check_refs(refs, {object_name, *(obj for obj, _ in refs)}, record_id, self_ids)


@lru_cache(maxsize=1)
def load_map() -> SensitiveMap:
    return SensitiveMap(yaml.safe_load(MAP_PATH.read_text(encoding="utf-8")))
