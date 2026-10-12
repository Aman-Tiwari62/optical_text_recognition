from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_vocabulary(
    records: list[str],
    blank_token: str = "<blank>",
    unknown_token: str = "<unk>",
) -> list[str]:
    """Build a train-split character vocabulary with fixed CTC and unknown-token IDs."""
    chars = {character for text in records for character in text}
    chars.discard(blank_token)
    chars.discard(unknown_token)
    return [blank_token, unknown_token, *sorted(chars)]


def save_vocabulary(vocabulary: list[str], path: str | Path) -> None:
    write_json(path, {"vocabulary": vocabulary})


def load_vocabulary(path: str | Path) -> list[str]:
    payload = read_json(path)
    return payload.get("vocabulary", [])


def safe_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Invalid integer value for {field_name}: {value!r}") from exc

