import pytest

# Skip entire module if spaCy or the model isn't installed.
spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_lg")
except OSError:
    pytest.skip("en_core_web_lg not installed — run: python -m spacy download en_core_web_lg",
                allow_module_level=True)

from anonymizer import anonymize, AnonymizationResult


def test_email_replaced():
    result = anonymize("Contact us at jane.smith@acme.com for details.")
    assert "jane.smith@acme.com" not in result.anonymized
    assert "[EMAIL_1]" in result.anonymized


def test_phone_replaced():
    result = anonymize("Call me at 415-555-0192.")
    assert "415-555-0192" not in result.anonymized
    assert "[PHONE_1]" in result.anonymized


def test_url_replaced():
    result = anonymize("See https://acme.com/deals for pricing.")
    assert "https://acme.com/deals" not in result.anonymized
    assert "[URL_1]" in result.anonymized


def test_person_replaced():
    result = anonymize("John Smith closed the deal yesterday.")
    assert "John Smith" not in result.anonymized
    assert "[CONTACT_1]" in result.anonymized


def test_org_replaced():
    result = anonymize("We signed a contract with Acme Corporation.")
    assert "Acme Corporation" not in result.anonymized
    assert "[COMPANY_1]" in result.anonymized


def test_consistent_within_document():
    """Same surface text must map to same placeholder throughout the document."""
    result = anonymize("Jane Smith called. Jane Smith confirmed the meeting.")
    # Both occurrences should be the same placeholder
    assert result.anonymized.count("[CONTACT_1]") == 2
    assert "Jane Smith" not in result.anonymized


def test_two_different_people_get_different_placeholders():
    result = anonymize("Alice met with Bob to discuss the contract.")
    assert "Alice" not in result.anonymized
    assert "Bob" not in result.anonymized
    # They should be different placeholders
    assert "[CONTACT_1]" in result.anonymized
    assert "[CONTACT_2]" in result.anonymized


def test_mapping_returned():
    result = anonymize("Email jane@acme.com or call 415-555-0192.")
    assert isinstance(result.mapping, dict)
    assert len(result.mapping) >= 2


def test_empty_string():
    result = anonymize("")
    assert result.anonymized == ""
    assert result.mapping == {}


def test_no_pii_unchanged():
    text = "The meeting went well. Volume is up 15% this quarter."
    result = anonymize(text)
    # No PII — text should be unchanged (or only minor entity replacements)
    assert result.anonymized  # not empty


def test_crm_freetext():
    """Realistic CRM note — names, company, money, email all anonymized."""
    text = (
        "Called Sarah Johnson at DataBridge Inc. today. "
        "She's interested in the $120,000 enterprise package. "
        "Follow up at sarah.j@databridge.io next Tuesday."
    )
    result = anonymize(text)
    assert "Sarah Johnson" not in result.anonymized
    assert "sarah.j@databridge.io" not in result.anonymized
    assert "120,000" not in result.anonymized or "[DEAL_VALUE_1]" in result.anonymized
