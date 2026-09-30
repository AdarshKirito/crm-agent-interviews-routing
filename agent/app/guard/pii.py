"""Find personal data in requests and remove it from customer-facing answers.

Uses Presidio (spaCy NER + pattern recognizers). If Presidio cannot load -- e.g. on a
Windows host where Smart App Control blocks spaCy's compiled parser -- it falls back to
the same pattern recognizers without NER: contact details, financial identifiers and
Salesforce Ids are still found, person and place names are not. `ENGINE` says which
one is active; the screening signals record it. Measured runs use the Docker image,
where Presidio loads.
"""

import logging
import re
import threading
from functools import lru_cache

from .sensitive import SF_ID_RE

logger = logging.getLogger(__name__)

REQUEST_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "US_SSN", "CREDIT_CARD", "SALESFORCE_ID"]
# Answers are often Salesforce Ids, names of products or places in knowledge text, so
# only contact details and financial/government identifiers are removed.
REDACT_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "US_BANK_NUMBER"]

# Pattern-only fallback (same entity names as Presidio's recognizers).
_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE_NUMBER": re.compile(r"(?<!\w)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\w)"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "IBAN_CODE": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "SALESFORCE_ID": SF_ID_RE,
}

_lock = threading.Lock()
ENGINE = "unloaded"


@lru_cache(maxsize=1)
def _engines():
    global ENGINE
    with _lock:
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine

            nlp = NlpEngineProvider(
                nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}
            ).create_engine()
            analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
            analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="SALESFORCE_ID",
                    patterns=[Pattern("salesforce_id", SF_ID_RE.pattern, 0.6)],
                    context=["id", "record", "contact", "account", "case", "order"],
                )
            )
            ENGINE = "presidio"
            return analyzer, AnonymizerEngine()
        except (ImportError, OSError) as err:
            ENGINE = "patterns_only"
            logger.warning("Presidio unavailable (%s); using pattern-only PII detection", err)
            return None, None


def warm_up() -> str:
    _engines()
    return ENGINE


def _pattern_hits(text: str, entities: list[str]) -> list[tuple[str, int, int]]:
    hits = []
    for entity in entities:
        pattern = _PATTERNS.get(entity)
        if pattern:
            hits += [(entity, m.start(), m.end()) for m in pattern.finditer(text)]
    return sorted(hits, key=lambda h: h[1])


def analyze_request(text: str) -> list[dict]:
    text = text or ""
    analyzer, _ = _engines()
    if analyzer is None:
        return [{"type": e, "text": text[s:t], "score": 1.0} for e, s, t in _pattern_hits(text, REQUEST_ENTITIES)]
    results = analyzer.analyze(text=text, language="en", entities=REQUEST_ENTITIES, score_threshold=0.5)
    return [
        {"type": r.entity_type, "text": text[r.start : r.end], "score": round(r.score, 2)}
        for r in sorted(results, key=lambda r: r.start)
    ]


def scrub(text: str) -> tuple[str, list[str]]:
    """Replace contact details and financial identifiers; never touch Salesforce Ids."""
    if not text:
        return text, []
    analyzer, anonymizer = _engines()
    protected = [(m.start(), m.end()) for m in SF_ID_RE.finditer(text)]

    def overlaps_id(start: int, end: int) -> bool:
        return any(start < p_end and p_start < end for p_start, p_end in protected)

    if analyzer is None:
        hits = [(e, s, t) for e, s, t in _pattern_hits(text, REDACT_ENTITIES) if not overlaps_id(s, t)]
        out, last = [], 0
        for _, s, t in hits:
            if s < last:
                continue
            out.append(text[last:s] + "[REDACTED]")
            last = t
        return ("".join(out) + text[last:]) if hits else text, sorted({e for e, _, _ in hits})

    from presidio_anonymizer.entities import OperatorConfig

    results = [
        r
        for r in analyzer.analyze(text=text, language="en", entities=REDACT_ENTITIES, score_threshold=0.5)
        if not overlaps_id(r.start, r.end)
    ]
    if not results:
        return text, []
    out = anonymizer.anonymize(
        text=text, analyzer_results=results, operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
    )
    return out.text, sorted({r.entity_type for r in results})
