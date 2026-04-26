# Anonymization

## What it is

Anonymization replaces personally identifiable information (PII) in user input
with generic numbered placeholders before that text reaches the model, the session
store, or log files. Real names, company names, email addresses, and deal values
never leave your application boundary.

**Example — raw CRM input:**
```
Called Sarah Johnson at DataBridge Inc. today. She's interested in the
$120,000 enterprise package. Follow up at sarah.j@databridge.io next Tuesday.
```

**After anonymization:**
```
Called [CONTACT_1] at [COMPANY_1] today. She's interested in the
[DEAL_VALUE_1] enterprise package. Follow up at [EMAIL_1] next [DATE_1].
```

The model works with placeholders. The model output references `[CONTACT_1]`
and `[COMPANY_1]` — never the real names.

---

## Why this matters for CRM

CRM data contains some of the most sensitive PII in a business:
- Customer names and contact details
- Company names and deal values
- Internal notes about relationships and negotiations

**Legal exposure without anonymization:**
- **GDPR** — processing EU customer names requires a legal basis and data minimisation.
  Sending real names to a model that logs them violates minimisation.
- **CCPA** — California residents have the right to limit use of personal data.
- **HIPAA** — if any CRM records touch healthcare clients, real names in logs
  are a breach.

Anonymizing at ingestion means your model, logs, and session store never contain
real customer data — compliance becomes structurally enforced, not policy-dependent.

---

## Why NER over rule-based

### Rule-based approach
A rule-based anonymizer uses regex and lookup tables:
- Regex for emails, phones, URLs
- Hardcoded lists of known company names
- Pattern matching for currency (`$\d+`)

**Problems for CRM:**
- Can't catch names it hasn't seen before (`"Priya Mehta"` is not in any list)
- Company names are infinite — `"DataBridge Inc."`, `"Vertex Solutions GmbH"`
- Free-text notes don't follow predictable patterns
- Misses context — `"Apple"` in `"she uses an Apple laptop"` vs `"Apple Inc. signed"`

### NER-based approach (what's implemented)
Named Entity Recognition (NER) reads the sentence as a whole and classifies
each span by semantic type: PERSON, ORG, MONEY, DATE, etc. It understands
context — it knows `"Apple"` is a company when followed by `"signed a contract"`.

**Why spaCy over a transformer NER:**

| | spaCy en_core_web_lg | Transformer NER (e.g. roberta-base) |
|---|---|---|
| Accuracy | Good (F1 ~86%) | Higher (F1 ~91%) |
| Latency | ~5–15ms per request | ~80–200ms per request |
| Memory | ~700MB | ~1.5–3GB additional |
| GPU required | No | No, but faster with GPU |
| Setup | One pip install + download | Requires transformers + torch |

For a production service handling concurrent CRM requests, adding 80–200ms
per request is significant. spaCy's `en_core_web_lg` hits the right balance:
accurate enough for names and companies, fast enough to be invisible in latency.

**If you need higher accuracy** (e.g. medical records, legal documents):
swap `en_core_web_lg` for `en_core_web_trf` in `anonymizer.py`:
```python
_anonymizer = Anonymizer(model_name="en_core_web_trf")
```
Install: `pip install spacy[transformers]` then `python -m spacy download en_core_web_trf`

---

## What gets anonymized

| Entity type | spaCy label | Placeholder |
|---|---|---|
| Person names | PERSON | `[CONTACT_N]` |
| Companies / orgs | ORG | `[COMPANY_N]` |
| Countries, cities, states | GPE | `[LOCATION_N]` |
| Non-GPE locations | LOC | `[LOCATION_N]` |
| Buildings, airports | FAC | `[LOCATION_N]` |
| Money amounts | MONEY | `[DEAL_VALUE_N]` |
| Dates | DATE | `[DATE_N]` |
| Times | TIME | `[TIME_N]` |
| Percentages | PERCENT | `[PERCENT_N]` |
| Nationalities / groups | NORP | `[GROUP_N]` |
| Products | PRODUCT | `[PRODUCT_N]` |
| Events | EVENT | `[EVENT_N]` |
| Email addresses | regex | `[EMAIL_N]` |
| Phone numbers | regex | `[PHONE_N]` |
| URLs | regex | `[URL_N]` |

`N` is a per-document counter — `[CONTACT_1]` is the first person seen,
`[CONTACT_2]` is the second. The same surface string always maps to the same
placeholder within a single request so the model can follow references.

---

## Setup

**1. Install spaCy:**
```bash
pip install spacy>=3.7.0
# or if using requirements.txt:
pip install -r requirements.txt
```

**2. Download the language model:**
```bash
python -m spacy download en_core_web_lg
```

This downloads ~700MB to your Python environment. Only needed once.

**3. Enable anonymization:**

In your `.env` file:
```
ANONYMIZE=true
```

Or as an environment variable:
```bash
ANONYMIZE=true uvicorn main:app --host 0.0.0.0 --port 8000
```

Anonymization is **off by default** — existing requests are unaffected until
you explicitly enable it.

---

## How to verify it's working

Start the server with `ANONYMIZE=true` and send a request with a real name:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Meeting with Sarah Johnson at Acme Corp. Deal value $85,000. Email: sarah@acme.com"}],
    "max_tokens": 256
  }'
```

The model prompt will contain `[CONTACT_1]`, `[COMPANY_1]`, `[DEAL_VALUE_1]`,
`[EMAIL_1]` — none of the real values. Check server logs for:
```
INFO  main — Anonymized 4 entities before inference
```

---

## Limitations

**False positives** — spaCy occasionally tags non-PII as entities. `"Tuesday"` becomes
`[DATE_1]`, `"Q3"` might become `[DATE_1]`. This is usually acceptable for CRM —
dates and times in customer notes are still sensitive.

**False negatives** — spaCy may miss PII in very short or ambiguous text, highly
informal writing, or non-English input. For non-English CRM data, download the
appropriate spaCy model (e.g. `de_core_news_lg` for German).

**No de-anonymization** — the mapping (`result.mapping`) is returned per-request
but not stored anywhere. If you need to map placeholders back to real names in
the model output, you must store the mapping yourself (e.g. in your session store
alongside the conversation history). This is intentionally not done by default —
storing the mapping would re-introduce the PII.

**Structured fields** — if your CRM sends structured JSON (`{"name": "Jane", "company": "Acme"}`),
run anonymization on each string field individually before assembling the prompt.

---

## Future improvements

- **Custom entity types** — add domain-specific entities (e.g. `DEAL_ID`, `CONTRACT_NUM`)
  using spaCy's rule-based EntityRuler layered on top of the NER pipeline.
- **Language detection** — auto-select the right spaCy model based on detected language.
- **Reversible anonymization** — store the per-request mapping encrypted in the session
  store so the model output can be de-anonymized before returning to the client.
