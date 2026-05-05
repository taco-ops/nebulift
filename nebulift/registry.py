"""Local model registry helpers for Nebulift."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_REGISTRY_PATH = Path("models/model_registry.json")


def _load_json_file(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def load_model_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    """Load the local model registry."""
    if not registry_path.exists():
        return {"schema_version": 1, "default_model_id": None, "models": {}}
    data = json.loads(registry_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected registry JSON object in {registry_path}")
    return data


def save_model_registry(
    registry: dict[str, Any],
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> None:
    """Save the local model registry."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2))


def register_model(
    model_path: Path,
    name: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    model_id: Optional[str] = None,
    evaluation_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
    source_manifest: Optional[Path] = None,
    promote: bool = False,
    replace: bool = False,
) -> dict[str, Any]:
    """Register a model checkpoint with optional evaluation metadata."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    registry = load_model_registry(registry_path)
    model_id = model_id or model_path.stem
    models = registry.setdefault("models", {})
    if model_id in models and not replace:
        raise ValueError(f"Model id already registered: {model_id}")

    model_record = {
        "model_id": model_id,
        "name": name,
        "model_path": str(model_path),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_path": str(evaluation_path) if evaluation_path else None,
        "calibration_path": str(calibration_path) if calibration_path else None,
        "source_manifest": str(source_manifest) if source_manifest else None,
        "evaluation": _load_json_file(evaluation_path),
        "calibration": _load_json_file(calibration_path),
    }
    models[model_id] = model_record
    if promote:
        registry["default_model_id"] = model_id
    save_model_registry(registry, registry_path)
    return model_record


def promote_model(
    model_id: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Promote a registered model as the default local model."""
    registry = load_model_registry(registry_path)
    models = registry.setdefault("models", {})
    if model_id not in models:
        raise ValueError(f"Model id not found in registry: {model_id}")

    model_path = Path(models[model_id]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Registered model file not found: {model_path}")

    registry["default_model_id"] = model_id
    save_model_registry(registry, registry_path)
    model_record = models[model_id]
    if not isinstance(model_record, dict):
        raise ValueError(f"Invalid model record for id: {model_id}")
    return model_record


def resolve_model_path(
    model_path: Optional[Path] = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    use_default_model: bool = True,
) -> Optional[Path]:
    """Resolve an explicit model path or promoted default model path."""
    if model_path is not None:
        return model_path
    if not use_default_model:
        return None

    registry = load_model_registry(registry_path)
    default_model_id = registry.get("default_model_id")
    if not default_model_id:
        return None

    model_record = registry.get("models", {}).get(default_model_id)
    if not model_record:
        raise ValueError(f"Default model id not found in registry: {default_model_id}")

    resolved_path = Path(model_record["model_path"])
    if not resolved_path.exists():
        raise FileNotFoundError(f"Default model file not found: {resolved_path}")
    return resolved_path


def print_model_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """Print registered local models."""
    registry = load_model_registry(registry_path)
    default_model_id = registry.get("default_model_id")
    models = registry.get("models", {})
    if not models:
        print(f"No models registered in {registry_path}")
        return

    for model_id, model_record in models.items():
        marker = " (default)" if model_id == default_model_id else ""
        print(f"{model_id}{marker}: {model_record['model_path']}")
