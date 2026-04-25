# Gemma Revenue Brief API

FastAPI service wrapping a fine-tuned Gemma 3 1B MLX model. Exposes OpenAI-compatible chat completions API.

## One-time setup

```bash
cd service
pip install -r requirements.txt
```

Copy and edit the env file (optional — defaults work for local dev):

```bash
cp .env.example .env   # or set env vars directly
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `./base_model` | Path to base model weights |
| `ADAPTER_PATH` | `./adapters` | Path to LoRA adapter directory |
| `API_KEYS` | `dev-key-12345` | Comma-separated list of valid API keys |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `MAX_TOKENS` | `1024` | Default max tokens for generation |

## Point service at your trained adapter

```bash
export ADAPTER_PATH=/path/to/your/adapters
export MODEL_PATH=/path/to/base/gemma-3-1b
```

Or create a `.env` file in the `service/` directory:

```
MODEL_PATH=../base_model
ADAPTER_PATH=../adapters
API_KEYS=your-key-here
```

## Run the server

```bash
cd service
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Run without adapter (base model only)

Set `ADAPTER_PATH` to a path that does not exist. The service logs a warning and loads the base model. `/health` returns `"adapter_loaded": false`.

```bash
ADAPTER_PATH=/nonexistent uvicorn main:app --port 8000
```

## API key management

`API_KEYS` is a comma-separated string. To rotate or add keys, update the env var and restart:

```bash
export API_KEYS="key-abc123,key-xyz789,new-key-here"
```

All keys in the list are valid simultaneously. Remove a key by omitting it and restarting.

## Example curl commands

### GET /health (no auth required)

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true, "adapter_loaded": true}
```

### POST /v1/chat/completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-revenue-brief",
    "messages": [
      {"role": "user", "content": "Client: John Doe. Sessions: 3x/week. Goal: hypertrophy."}
    ],
    "temperature": 0,
    "max_tokens": 1024
  }'
```

### POST /v1/validate

```bash
curl -X POST http://localhost:8000/v1/validate \
  -H "Authorization: Bearer dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "{\"record_type\": \"session_summary\", \"week\": 1}\n{\"record_type\": \"recommendation\", \"action_id\": \"INCREASE_LOAD\"}"
  }'
```

```json
{"valid": true, "errors": [], "parsed": [...]}
```
