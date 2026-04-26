from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"
_cache: dict[str, str] = {}


def _load(name: str) -> str:
    if name not in _cache:
        path = _TEMPLATE_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Template '{name}' not found — expected {path}")
        _cache[name] = path.read_text(encoding="utf-8")
    return _cache[name]


def render(name: str, **kwargs: str) -> str:
    """Load template by name and substitute keyword placeholders."""
    return _load(name).format(**kwargs)


def list_templates() -> list[str]:
    """Return names of all available templates (filename without .txt)."""
    return sorted(p.stem for p in _TEMPLATE_DIR.glob("*.txt"))
