# Gemma Revenue Brief API

FastAPI service wrapping a fine-tuned Gemma 3 1B MLX model. Produces structured NDJSON revenue briefs from weekly business data.

## Pipeline

```
Weekly input (free text / structured)
    ↓
Prompt template (prompt_templates/default.txt)
    ↓
Fine-tuned Gemma 3 1B (MLX, Apple Silicon)
    ↓
NDJSON validation + auto-retry (up to 3 attempts)
    ↓
Stored report  →  JSON response  →  HTML dashboard
```

## One-time setup

```bash
cd service
pip install -r requirements.txt
```

Copy and edit the env file (optional — defaults work for local dev):

```bash
cp .env.example .env
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `mlx-community/gemma-3-1b-it-4bit` | Path to base model weights |
| `ADAPTER_PATH` | `../phase2_ndjson/adapters` | Path to LoRA adapter directory |
| `API_KEYS` | `dev-key-12345` | Comma-separated list of valid API keys |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `MAX_TOKENS` | `1024` | Default max tokens for generation |
| `ANONYMIZE` | `false` | Enable NER-based PII anonymization (see ANONYMIZATION.md) |
| `RATE_LIMIT_RPM` | `60` | Max requests per API key per minute (0 = unlimited) |
| `MAX_REPORT_RETRIES` | `3` | Auto-retry attempts when NDJSON output is invalid |

## Run the server

```bash
cd service
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API endpoints

### GET /health

No auth required. Returns model and adapter load status.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true, "adapter_loaded": true}
```

---

### POST /v1/reports  ← primary endpoint

Generate a structured NDJSON revenue brief. Automatically retries up to `max_retries` times if the model outputs invalid NDJSON.

```bash
curl -X POST http://localhost:8000/v1/reports \
  -H "Authorization: Bearer dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Week ending 2024-04-26. Top deal: Acme Corp $180K closed. Pipeline: 3 deals at risk due to budget freeze. Churn: 2 accounts flagged.",
    "template": "default",
    "max_retries": 3
  }'
```

Response:
```json
{
  "id": "report-abc123",
  "created_at": 1714128000,
  "template": "default",
  "valid": true,
  "attempts": 1,
  "output": "{\"record_type\": \"session_summary\", ...}\n{\"record_type\": \"highlight\", ...}",
  "errors": [],
  "usage": {"prompt_tokens": 312, "completion_tokens": 890, "total_tokens": 1202}
}
```

---

### GET /v1/reports

List all stored reports, newest first.

```bash
curl http://localhost:8000/v1/reports \
  -H "Authorization: Bearer dev-key-12345"
```

---

### GET /v1/reports/{id}

Retrieve a stored report by ID.

```bash
curl http://localhost:8000/v1/reports/report-abc123 \
  -H "Authorization: Bearer dev-key-12345"
```

---

### GET /v1/reports/{id}/html

Render a stored report as an HTML page for leadership dashboards.

```bash
open "http://localhost:8000/v1/reports/report-abc123/html"
```

Or fetch and save:

```bash
curl http://localhost:8000/v1/reports/report-abc123/html \
  -H "Authorization: Bearer dev-key-12345" \
  -o brief.html && open brief.html
```

---

### DELETE /v1/reports/{id}

Delete a stored report.

```bash
curl -X DELETE http://localhost:8000/v1/reports/report-abc123 \
  -H "Authorization: Bearer dev-key-12345"
```

---

### GET /v1/templates

List available prompt templates (files in `prompt_templates/`).

```bash
curl http://localhost:8000/v1/templates \
  -H "Authorization: Bearer dev-key-12345"
```

```json
{"templates": ["default"]}
```

---

### POST /v1/chat/completions

Raw OpenAI-compatible chat completions endpoint. Supports `session_id` for multi-turn memory and `stream=true` for SSE streaming. See MEMORY.md for session usage.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "...workout log..."}],
    "temperature": 0,
    "max_tokens": 1024
  }'
```

---

### POST /v1/validate

Validate a raw NDJSON string against the report schema.

```bash
curl -X POST http://localhost:8000/v1/validate \
  -H "Authorization: Bearer dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"content": "{\"record_type\": \"session_summary\"}"}'
```

---

### DELETE /v1/sessions/{session_id}

Clear stored conversation history for a session.

```bash
curl -X DELETE http://localhost:8000/v1/sessions/my-session \
  -H "Authorization: Bearer dev-key-12345"
```

---

## Prompt templates

Templates live in `prompt_templates/*.txt`. Variables are `{input}` (the weekly data).

To add a template for CRM (when the real schema arrives):
1. Create `prompt_templates/crm_weekly.txt` with the new schema and action IDs.
2. POST to `/v1/reports` with `"template": "crm_weekly"`.

---

## Auto-retry

When the model outputs invalid NDJSON, the service:
1. Validates the output.
2. Appends the bad output + specific errors to the conversation.
3. Asks the model to fix it.
4. Repeats up to `max_retries` times (default 3).

If all attempts fail, the report is stored with `"valid": false` and the errors are returned so callers can decide how to handle it.

---

## Rate limiting

Requests are limited per API key using a sliding 60-second window.
Default: 60 requests/minute. Override with `RATE_LIMIT_RPM=0` to disable.
Exceeded limit returns `429 {"error": "rate limit exceeded"}`.

---

## Other guides

- **ANONYMIZATION.md** — NER-based PII anonymization for CRM inputs
- **MEMORY.md** — Conversation session memory, storage backends (SQLite / Redis)
- **TRANSITION.md** — How to migrate from MLX (Mac) to a Linux GPU server
