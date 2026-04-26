# anonymizer.py — NER-based PII anonymization for CRM inputs
#
# Replaces personally identifiable information (names, companies, locations,
# money amounts, contact details) with consistent numbered placeholders before
# text is sent to the model, stored in session history, or written to logs.
#
# Uses spaCy's en_core_web_lg model for named entity recognition.
# See ANONYMIZATION.md for setup instructions and design rationale.

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────
# spaCy NER reliably catches names, orgs, locations, and money but misses
# structured PII like emails, phone numbers, and URLs. Regex handles those.

_REGEX_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL",   re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")),
    ("PHONE",   re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b"
    )),
    ("URL",     re.compile(r"https?://[^\s]+")),
]

# ── spaCy entity label → placeholder prefix ───────────────────────────────────
# Labels not in this map are ignored (e.g. ORDINAL, CARDINAL, WORK_OF_ART).
# CRM-relevant: who (PERSON→CONTACT), what company (ORG→COMPANY),
# where (GPE/LOC/FAC→LOCATION), how much (MONEY→DEAL_VALUE), when (DATE/TIME).

_LABEL_MAP: Dict[str, str] = {
    "PERSON":   "CONTACT",
    "ORG":      "COMPANY",
    "GPE":      "LOCATION",   # countries, cities, states
    "LOC":      "LOCATION",   # non-GPE locations (mountains, bodies of water)
    "FAC":      "LOCATION",   # buildings, airports, highways
    "MONEY":    "DEAL_VALUE",
    "DATE":     "DATE",
    "TIME":     "TIME",
    "PERCENT":  "PERCENT",
    "NORP":     "GROUP",      # nationalities, religions, political groups
    "PRODUCT":  "PRODUCT",
    "EVENT":    "EVENT",
}


@dataclass
class AnonymizationResult:
    """Output of a single anonymization pass.

    anonymized: text with all PII replaced by placeholders.
    mapping:    {original_text: placeholder} for every replacement made.
                Useful for auditing and, if needed, de-anonymization.
    """
    anonymized: str
    mapping: Dict[str, str] = field(default_factory=dict)


def _collect_spans(
    text: str,
    doc,
) -> List[Tuple[int, int, str, str]]:
    """Return non-overlapping (start, end, label, surface_text) sorted by position.

    Regex runs first; spaCy spans that overlap an already-matched regex span
    are dropped. This ensures structured PII (email, phone) is always caught
    even when spaCy misidentifies them as a different entity type.
    """
    spans: List[Tuple[int, int, str, str]] = []

    # Regex pass — emails, phones, URLs
    for label, pattern in _REGEX_PATTERNS:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), label, m.group()))

    # spaCy NER pass — names, orgs, locations, money, dates …
    for ent in doc.ents:
        if ent.label_ in _LABEL_MAP:
            spans.append((ent.start_char, ent.end_char, ent.label_, ent.text))

    # Sort: leftmost first; ties broken by longer span first.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    # Remove overlaps — keep the first (leftmost/longest) match at each position.
    filtered: List[Tuple[int, int, str, str]] = []
    last_end = 0
    for span in spans:
        if span[0] >= last_end:
            filtered.append(span)
            last_end = span[1]

    return filtered


class Anonymizer:
    """Wraps a spaCy pipeline and performs PII replacement on arbitrary text.

    The model is loaded lazily on first use so that importing this module
    does not block startup when anonymization is disabled in config.
    spaCy models are thread-safe after loading — safe for concurrent requests.
    """

    def __init__(self, model_name: str = "en_core_web_lg"):
        self._model_name = model_name
        self._nlp = None  # loaded on first call to anonymize()

    def _load(self) -> None:
        try:
            import spacy
            logger.info("Loading spaCy model %s", self._model_name)
            self._nlp = spacy.load(self._model_name)
            logger.info("spaCy model ready")
        except OSError:
            raise RuntimeError(
                f"spaCy model '{self._model_name}' not found. "
                f"Run: python -m spacy download {self._model_name}"
            )

    def anonymize(self, text: str) -> AnonymizationResult:
        """Replace all detected PII in text with consistent numbered placeholders.

        The same surface string always maps to the same placeholder within one
        call — so if "Jane Smith" appears three times, all three become
        "[CONTACT_1]" and a different person becomes "[CONTACT_2]".

        Returns AnonymizationResult with the cleaned text and the replacement map.
        """
        if not text or not text.strip():
            return AnonymizationResult(anonymized=text or "")

        if self._nlp is None:
            self._load()

        doc = self._nlp(text)
        spans = _collect_spans(text, doc)

        if not spans:
            return AnonymizationResult(anonymized=text)

        # Build within-document consistent mapping.
        # seen:     surface_text → placeholder (for deduplication)
        # counters: placeholder_prefix → count (for numbering)
        seen: Dict[str, str] = {}
        counters: Dict[str, int] = {}
        mapping: Dict[str, str] = {}

        for _, _, label, surface in spans:
            if surface not in seen:
                prefix = _LABEL_MAP.get(label, label)
                counters[prefix] = counters.get(prefix, 0) + 1
                placeholder = f"[{prefix}_{counters[prefix]}]"
                seen[surface] = placeholder
                mapping[surface] = placeholder

        # Replace right-to-left so earlier character positions stay valid.
        result = text
        for start, end, _, surface in sorted(spans, key=lambda s: s[0], reverse=True):
            result = result[:start] + seen[surface] + result[end:]

        logger.debug("Anonymized %d entities: %s", len(mapping), list(mapping.values()))
        return AnonymizationResult(anonymized=result, mapping=mapping)


# Module-level singleton — one spaCy model shared across all requests.
_anonymizer: Optional[Anonymizer] = None


def get_anonymizer(model_name: str = "en_core_web_lg") -> Anonymizer:
    """Return the shared Anonymizer instance, creating it if necessary."""
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = Anonymizer(model_name=model_name)
    return _anonymizer


def anonymize(text: str) -> AnonymizationResult:
    """Module-level convenience function — anonymize text using the shared instance."""
    return get_anonymizer().anonymize(text)
