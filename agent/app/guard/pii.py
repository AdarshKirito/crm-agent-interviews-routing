"""Presidio: find personal data in requests and remove it from customer-facing answers."""

import threading
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .sensitive import SF_ID_RE

REQUEST_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "US_SSN", "CREDIT_CARD", "SALESFORCE_ID"]
# Answers are often Salesforce Ids, names of products or places in knowledge text, so
# only contact details and financial/government identifiers are removed.
REDACT_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "US_BANK_NUMBER"]

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    with _lock:
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
        return analyzer, AnonymizerEngine()


def warm_up() -> None:
    _engines()


def analyze_request(text: str) -> list[dict]:
    analyzer, _ = _engines()
    results = analyzer.analyze(text=text or "", language="en", entities=REQUEST_ENTITIES, score_threshold=0.5)
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
    results: list[RecognizerResult] = [
        r
        for r in analyzer.analyze(text=text, language="en", entities=REDACT_ENTITIES, score_threshold=0.5)
        if not any(r.start < end and start < r.end for start, end in protected)
    ]
    if not results:
        return text, []
    out = anonymizer.anonymize(
        text=text, analyzer_results=results, operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
    )
    return out.text, sorted({r.entity_type for r in results})
