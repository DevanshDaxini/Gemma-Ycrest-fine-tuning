# config.py — centralised settings loaded from environment variables
#
# pydantic-settings reads each field from the matching env var (uppercased).
# If the env var is absent, the default value is used.
# A .env file in the working directory is also read automatically.
#
# Example .env:
#   MODEL_PATH=./base_model
#   ADAPTER_PATH=./adapters
#   API_KEYS=key-abc123,key-xyz789

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Path to the base model weights. Accepts a local directory OR a
    # HuggingFace repo ID (e.g. "mlx-community/gemma-3-1b-it-4bit").
    model_path: Path = Path("mlx-community/gemma-3-1b-it-4bit")

    # Path to the LoRA adapter directory produced by training.
    # If this path does not exist, the service loads the base model only.
    adapter_path: Path = Path("../phase2_ndjson/adapters")

    # Comma-separated list of valid API keys for service-to-service auth.
    # "dev-key-12345" is the default for local development only — always
    # override this in production.
    api_keys: str = "dev-key-12345"

    port: int = 8000
    host: str = "0.0.0.0"

    # Default maximum tokens to generate per request (can be overridden per-request).
    max_tokens: int = 1024

    # When True, all user input is anonymized before reaching the model, session
    # store, or logs. Requires spaCy + en_core_web_lg. See ANONYMIZATION.md.
    anonymize: bool = False

    # Max requests per API key per 60-second window. 0 = unlimited.
    rate_limit_rpm: int = 60

    # Default max retry attempts for report generation when NDJSON output is invalid.
    max_report_retries: int = 3

    @property
    def api_key_set(self) -> set:
        """Parse the comma-separated API_KEYS string into a set for O(1) lookup."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
