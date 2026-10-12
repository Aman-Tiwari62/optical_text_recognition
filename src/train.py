from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ProjectConfig, get_project_config
from src.dataset import (
    IAMAnnotationRecord,
    IAMWordDataset,
    collate_iam_batch,
    load_iam_records,
    save_split_metadata,
    split_records_by_form,
)
from src.model import CNNBiLSTMCTC
from src.utils import build_vocabulary, load_vocabulary, save_vocabulary


def _check_form_disjointness(splits: dict[str, list[IAMAnnotationRecord]]) -> None:
    forms_by_split = {
        name: {record.form_id for record in records}
        for name, records in splits.items()
    }
    names = list(forms_by_split)
    for index, name in enumerate(names):
        for other_name in names[index + 1 :]:
            overlap = forms_by_split[name] & forms_by_split[other_name]
            if overlap:
                raise ValueError(
                    f"Form leakage between {name} and {other_name}: {sorted(overlap)[:5]}"
                )


def _load_saved_splits(
    path: Path,
    records: list[IAMAnnotationRecord],
) -> dict[str, list[IAMAnnotationRecord]]:
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    if payload.get("grouping") != "form_id" or "splits" not in payload:
        raise ValueError(
            f"Existing split metadata at {path} is not form-grouped. "
            "Use --refresh-splits to regenerate it."
        )

    records_by_id = {record.word_id: record for record in records}
    saved_ids = {
        word_id
        for split_records in payload["splits"].values()
        for word_id in (item["word_id"] for item in split_records)
    }
    if saved_ids != records_by_id.keys():
        raise ValueError(
            f"Saved split IDs in {path} do not match the currently valid dataset. "
            "Use --refresh-splits to regenerate the splits."
        )

    split_map = {
        split: [records_by_id[item["word_id"]] for item in saved_records]
        for split, saved_records in payload["splits"].items()
    }
    if set(split_map) != {"train", "val", "test"}:
        raise ValueError(f"Split metadata at {path} must contain train, val, and test.")
    _check_form_disjointness(split_map)
    return split_map


def prepare_splits(
    records: list[IAMAnnotationRecord],
    config: ProjectConfig,
    refresh: bool = False,
) -> dict[str, list[IAMAnnotationRecord]]:
    if config.splits_path.exists() and not refresh:
        return _load_saved_splits(config.splits_path, records)

    splits = split_records_by_form(
        records,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    _check_form_disjointness(splits)
    save_split_metadata(splits, config.splits_path, seed=config.seed)
    return splits


def build_data_loaders(
    config: ProjectConfig,
    batch_size: int | None = None,
    refresh_splits: bool = False,
) -> tuple[dict[str, DataLoader], list[str], dict[str, list[IAMAnnotationRecord]]]:
    """Load IAM records, build/reuse form-independent splits, and create DataLoaders."""
    records, stats = load_iam_records(
        config.annotation_file,
        config.image_root,
        include_err=config.include_err,
        sample_limit=10,
    )
    if not records:
        raise ValueError("No readable IAM image/transcription pairs are available.")
    print(
        f"Loaded {stats['valid_samples']} samples "
        f"(ok={stats['status_counts']['ok']}, err={stats['status_counts']['err']}, "
        f"missing={stats['missing_images']}, malformed={stats['malformed_records']})."
    )

    splits = prepare_splits(records, config, refresh=refresh_splits)
    for name, split_records in splits.items():
        forms = {record.form_id for record in split_records}
        print(f"{name}: {len(split_records)} samples from {len(forms)} forms.")

    vocabulary = build_vocabulary([record.transcription for record in splits["train"]])
    if config.vocab_path.exists():
        saved_vocabulary = load_vocabulary(config.vocab_path)
        if saved_vocabulary != vocabulary:
            raise ValueError(
                f"Saved vocabulary at {config.vocab_path} differs from the current training split. "
                "Remove or update only this generated processed file after confirming the split."
            )
    else:
        save_vocabulary(vocabulary, config.vocab_path)

    datasets = {
        name: IAMWordDataset(
            split_records,
            image_height=config.image_height,
            pad_value=config.pad_value,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
            vocabulary=vocabulary,
        )
        for name, split_records in splits.items()
    }
    collate = partial(
        collate_iam_batch,
        blank_index=vocabulary.index("<blank>"),
        pad_value=config.pad_value,
        pool_steps=len(config.cnn_channels),
    )
    generator = torch.Generator().manual_seed(config.seed)
    actual_batch_size = batch_size or config.batch_size
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=(name == "train"),
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }
    return loaders, vocabulary, splits


def _minimum_ctc_steps(target: torch.Tensor) -> int:
    adjacent_repeats = int((target[1:] == target[:-1]).sum().item()) if target.numel() > 1 else 0
    return int(target.numel()) + adjacent_repeats


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        if torch.any(input_lengths < target_lengths):
            raise ValueError("A batch contains a transcription longer than its CNN output sequence.")
        for row, target_length in enumerate(target_lengths.tolist()):
            if input_lengths[row].item() < _minimum_ctc_steps(targets[row, :target_length].cpu()):
                raise ValueError(
                    f"CTC requires more frames than available for {batch['word_ids'][row]!r}."
                )

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            log_probs = logits.log_softmax(dim=2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise FloatingPointError("CTC loss is not finite.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().item())
        num_batches += 1
    return total_loss / max(1, num_batches)


def train(config: ProjectConfig, refresh_splits: bool = False) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    loaders, vocabulary, _ = build_data_loaders(config, refresh_splits=refresh_splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTMCTC(
        image_height=config.image_height,
        cnn_channels=config.cnn_channels,
        rnn_hidden_size=config.rnn_hidden_size,
        rnn_num_layers=config.rnn_num_layers,
        vocab_size=len(vocabulary),
        dropout=config.dropout,
    ).to(device)
    criterion = nn.CTCLoss(blank=vocabulary.index("<blank>"), zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoints = config.root_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs_dir = config.root_dir / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics_path = logs_dir / f"training_metrics_{run_id}.jsonl"
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = 5

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss = run_epoch(model, loaders["val"], criterion, device)
        print(f"epoch {epoch}/{config.epochs}: train_ctc={train_loss:.4f}, val_ctc={val_loss:.4f}")
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "epoch": epoch,
                        "train_ctc_loss": train_loss,
                        "val_ctc_loss": val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                + "\n"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocabulary": vocabulary,
                    "config": config.as_dict(),
                    "epoch": epoch,
                    "val_ctc_loss": val_loss,
                },
                checkpoints / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without validation improvement.")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN-BiLSTM-CTC handwriting recognizer.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--refresh-splits",
        action="store_true",
        help="Regenerate processed splits using the configured seed and ratios.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate data and generate splits/vocabulary without training.",
    )
    args = parser.parse_args()

    config = get_project_config(args.config)
    if args.prepare_only:
        build_data_loaders(config, refresh_splits=args.refresh_splits)
    else:
        train(config, refresh_splits=args.refresh_splits)


if __name__ == "__main__":
    main()
