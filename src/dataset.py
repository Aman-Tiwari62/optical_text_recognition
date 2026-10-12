from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils import build_vocabulary, safe_int


@dataclass(frozen=True)
class IAMAnnotationRecord:
    word_id: str
    status: str
    graylevel: int
    x: int
    y: int
    width: int
    height: int
    tag: str
    transcription: str
    image_path: str

    @property
    def form_id(self) -> str:
        return "-".join(self.word_id.split("-")[:2])


def resolve_iam_image_path(word_id: str, image_root: str | Path) -> Path | None:
    """Resolve a IAM word ID to the corresponding image under the IAM words directory.

    The actual file layout is <image_root>/<writer>/<form>/<wordid>.<ext>. We search the
    standard IAM extensions instead of assuming a single fixed path or extension.
    """

    image_root = Path(image_root)
    parts = word_id.split("-")
    if len(parts) < 4:
        return None

    writer_id = parts[0]
    form_id = "-".join(parts[:2])
    candidate_dir = image_root / writer_id / form_id
    if not candidate_dir.exists():
        return None

    matches = sorted(candidate_dir.glob(f"{word_id}.*"))
    if not matches:
        return None
    return matches[0]


def parse_iam_annotation_line(line: str, include_err: bool = False) -> tuple[IAMAnnotationRecord | None, str | None]:
    """Parse a single IAM annotation line.

    Returns a tuple of (record, reason). The reason is None for a valid record, a string for
    a malformed or excluded record. This keeps parsing explicit about things we drop.
    """

    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None, "comment_or_blank"

    tokens = stripped.split()
    if len(tokens) < 9:
        return None, "malformed"

    word_id, status, gray_level, x, y, width, height, tag = tokens[:8]
    transcription = " ".join(tokens[8:])
    if not transcription:
        return None, "malformed"

    try:
        record = IAMAnnotationRecord(
            word_id=word_id,
            status=status,
            graylevel=safe_int(gray_level, "graylevel"),
            x=safe_int(x, "x"),
            y=safe_int(y, "y"),
            width=safe_int(width, "width"),
            height=safe_int(height, "height"),
            tag=tag,
            transcription=transcription,
            image_path="",
        )
    except ValueError:
        return None, "malformed"

    if status.lower() in {"err", "er"}:
        if include_err:
            return record, "err_included"
        return None, "excluded_err"

    if status.lower() not in {"ok", "err"}:
        return None, "malformed"

    return record, None


def load_iam_records(
    annotation_path: str | Path,
    image_root: str | Path,
    include_err: bool = False,
    sample_limit: int = 10,
) -> tuple[list[IAMAnnotationRecord], dict[str, Any]]:
    """Parse annotations, verify referenced images, and return selected usable records."""
    annotation_path = Path(annotation_path)
    image_root = Path(image_root)
    if not annotation_path.exists():
        fallback = annotation_path.parent / "words_new.txt"
        if fallback.exists():
            raise FileNotFoundError(
                "The IAM annotation file is missing at "
                f"{annotation_path}. The actual workspace data contains {fallback}. "
                "Rename or re-link the annotation file to the required paths without creating a "
                "second copy of the dataset."
            )
        raise FileNotFoundError(f"IAM annotation file not found: {annotation_path}")
    stats: dict[str, Any] = {
        "total_annotation_records": 0,
        "valid_annotation_records": 0,
        "missing_images": 0,
        "unreadable_images": 0,
        "malformed_records": 0,
        "duplicate_ids": 0,
        "records_excluded_due_to_segmentation_status": 0,
        "successfully_loaded_image_label_pairs": 0,
        "status_counts": {"ok": 0, "err": 0},
        "unreadable_image_paths": [],
        "sample_records": [],
    }
    seen_ids: set[str] = set()
    valid_records: list[IAMAnnotationRecord] = []

    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record, reason = parse_iam_annotation_line(line, include_err=True)
            if record is None and reason in {"comment_or_blank"}:
                continue
            stats["total_annotation_records"] += 1
            fields = line.split()
            status = fields[1].lower() if len(fields) > 1 else ""
            if status in {"ok", "err", "er"}:
                stats["status_counts"]["err" if status in {"err", "er"} else "ok"] += 1
            if record is None:
                if reason == "malformed":
                    stats["malformed_records"] += 1
                continue

            stats["valid_annotation_records"] += 1

            if record.word_id in seen_ids:
                stats["duplicate_ids"] += 1
                continue
            seen_ids.add(record.word_id)

            is_err = record.status.lower() in {"err", "er"}
            if is_err and not include_err:
                stats["records_excluded_due_to_segmentation_status"] += 1

            image_path = resolve_iam_image_path(record.word_id, image_root)
            if image_path is None:
                stats["missing_images"] += 1
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError):
                stats["unreadable_images"] += 1
                stats["unreadable_image_paths"].append(str(image_path))
                continue

            if is_err and not include_err:
                continue

            valid_records.append(
                IAMAnnotationRecord(
                    word_id=record.word_id,
                    status=record.status,
                    graylevel=record.graylevel,
                    x=record.x,
                    y=record.y,
                    width=record.width,
                    height=record.height,
                    tag=record.tag,
                    transcription=record.transcription,
                    image_path=str(image_path),
                )
            )

    stats["successfully_loaded_image_label_pairs"] = len(valid_records)
    stats["valid_samples"] = len(valid_records)
    stats["sample_records"] = [
        {
            "word_id": item.word_id,
            "status": item.status,
            "transcription": item.transcription,
            "image_path": item.image_path,
        }
        for item in valid_records[:sample_limit]
    ]
    return valid_records, stats


def validate_iam_annotations(
    annotation_path: str | Path,
    image_root: str | Path,
    include_err: bool = False,
    sample_limit: int = 10,
) -> dict[str, Any]:
    """Validate IAM annotations and image readability without mutating original data."""
    _, stats = load_iam_records(annotation_path, image_root, include_err, sample_limit)
    return stats


def split_records_by_form(
    records: list[IAMAnnotationRecord],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[IAMAnnotationRecord]]:
    """Split by form ID; the available word annotations do not contain reliable writer IDs."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-8):
        raise ValueError("Train, validation, and test ratios must be nonnegative and sum to 1.")

    grouped: dict[str, list[IAMAnnotationRecord]] = {}
    for record in records:
        grouped.setdefault(record.form_id, []).append(record)

    group_ids = sorted(grouped)
    if len(group_ids) < 3:
        raise ValueError("At least three distinct forms are required for nonempty train/val/test splits.")
    random.Random(seed).shuffle(group_ids)

    train_count = max(1, int(len(group_ids) * train_ratio))
    val_count = max(1, int(len(group_ids) * val_ratio))
    if train_count + val_count >= len(group_ids):
        train_count = len(group_ids) - 2
        val_count = 1

    per_split: dict[str, list[IAMAnnotationRecord]] = {"train": [], "val": [], "test": []}
    for idx, group_id in enumerate(group_ids):
        if idx < train_count:
            bucket = "train"
        elif idx < train_count + val_count:
            bucket = "val"
        else:
            bucket = "test"
        per_split[bucket].extend(grouped[group_id])
    return per_split


def save_split_metadata(
    split_map: dict[str, list[IAMAnnotationRecord]],
    output_path: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "grouping": "form_id",
        "writer_ids_available": False,
        "splits": {
            split: [
                {
                    "word_id": item.word_id,
                    "form_id": item.form_id,
                    "image_path": (
                        Path(item.word_id.split("-")[0]) / item.form_id / Path(item.image_path).name
                    ).as_posix(),
                    "transcription": item.transcription,
                }
                for item in items
            ]
            for split, items in split_map.items()
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


class IAMWordDataset(Dataset):
    """Dataset for IAM word images and their transcriptions."""

    def __init__(
        self,
        records: list[IAMAnnotationRecord],
        image_height: int = 32,
        pad_value: float = 0.0,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5,
        blank_token: str = "<blank>",
        unknown_token: str = "<unk>",
        vocabulary: list[str] | None = None,
    ) -> None:
        self.records = records
        self.image_height = image_height
        self.pad_value = pad_value
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.blank_token = blank_token
        self.unknown_token = unknown_token

        if vocabulary is None:
            vocabulary = build_vocabulary([record.transcription for record in records], blank_token=blank_token)
            vocabulary = build_vocabulary(
                [record.transcription for record in records],
                blank_token=blank_token,
                unknown_token=unknown_token,
            )
        self.vocabulary = vocabulary
        self.char_to_index = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.blank_index = self.char_to_index.get(blank_token, 0)
        self.unknown_index = self.char_to_index.get(unknown_token, self.blank_index)

    def encode_text(self, text: str) -> list[int]:
        encoded: list[int] = []
        for char in text:
            if char in self.char_to_index:
                encoded.append(self.char_to_index[char])
            else:
                encoded.append(self.unknown_index)
        return encoded

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("L")
        width, height = image.size
        scale = self.image_height / max(1, height)
        target_width = max(1, int(round(width * scale)))
        resized = image.resize((target_width, self.image_height), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)
        if self.normalize_std <= 0:
            raise ValueError("normalize_std must be greater than zero.")
        array = (array / 255.0 - self.normalize_mean) / self.normalize_std
        tensor = torch.from_numpy(array).unsqueeze(0)

        target = torch.tensor(self.encode_text(record.transcription), dtype=torch.long)
        return {
            "image": tensor,
            "target": target,
            "target_text": record.transcription,
            "image_path": record.image_path,
            "word_id": record.word_id,
        }


def collate_iam_batch(
    batch: list[dict[str, Any]],
    blank_index: int = 0,
    pad_value: float = 0.0,
    pool_steps: int = 3,
) -> dict[str, Any]:
    """Pad variable-width images and provide pixel widths plus CNN sequence lengths."""

    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    image_widths = torch.tensor([img.shape[-1] for img in images], dtype=torch.long)
    target_lengths = torch.tensor([target.numel() for target in targets], dtype=torch.long)
    stride = 2**pool_steps
    max_width = max(stride, math.ceil(int(image_widths.max().item()) / stride) * stride)
    input_lengths = torch.div(image_widths + stride - 1, stride, rounding_mode="floor")
    max_target_len = int(target_lengths.max().item())

    padded_images = torch.full(
        (len(images), 1, images[0].shape[1], max_width),
        fill_value=pad_value,
        dtype=images[0].dtype,
    )
    for idx, image in enumerate(images):
        padded_images[idx, :, :, : image.shape[-1]] = image

    padded_targets = torch.full((len(targets), max_target_len), fill_value=blank_index, dtype=torch.long)
    for idx, target in enumerate(targets):
        padded_targets[idx, : target.numel()] = target

    return {
        "images": padded_images,
        "targets": padded_targets,
        "input_lengths": input_lengths,
        "image_widths": image_widths,
        "target_lengths": target_lengths,
        "raw_paths": [item["image_path"] for item in batch],
        "word_ids": [item["word_id"] for item in batch],
        "transcriptions": [item["target_text"] for item in batch],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate IAM annotation records and image paths.")
    parser.add_argument("--annotation", type=str, default="data/raw/iam_words/words.txt", help="IAM annotation file path")
    parser.add_argument("--image-root", type=str, default="data/raw/iam_words/words", help="Directory containing IAM word images")
    parser.add_argument("--include-err", action="store_true", help="Include err-segmented words in the validation pass")
    args = parser.parse_args()

    stats = validate_iam_annotations(args.annotation, args.image_root, include_err=args.include_err)
    print("IAM dataset validation summary")
    for key in [
        "total_annotation_records",
        "valid_annotation_records",
        "valid_samples",
        "missing_images",
        "unreadable_images",
        "malformed_records",
        "duplicate_ids",
        "records_excluded_due_to_segmentation_status",
        "successfully_loaded_image_label_pairs",
    ]:
        print(f"{key}: {stats[key]}")
    print(f"segmentation_status_counts: {stats['status_counts']}")
    print(f"unreadable_image_paths: {stats['unreadable_image_paths']}")
    print("sample_records:")
    for item in stats["sample_records"]:
        print(item)
