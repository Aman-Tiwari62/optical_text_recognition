from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    """Runtime configuration for the IAM handwriting recognition pipeline."""

    root_dir: Path
    annotation_file: Path
    image_root: Path
    processed_dir: Path
    splits_path: Path
    vocab_path: Path
    seed: int = 42
    image_height: int = 32
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    rnn_hidden_size: int = 128
    rnn_num_layers: int = 2
    dropout: float = 0.25
    include_err: bool = False
    unknown_character_policy: str = "replace_with_unk"
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    pad_value: float = 0.0
    num_workers: int = 0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ProjectConfig":
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        project = raw.get("project", {})
        paths = raw.get("paths", {})
        preprocess = raw.get("preprocessing", {})
        training = raw.get("training", {})
        model = raw.get("model", {})
        annotation = raw.get("annotation", {})
        split_cfg = raw.get("split", {})

        root_dir = Path(paths.get("root_dir", ".")).resolve()
        if not root_dir.exists():
            raise FileNotFoundError(f"Configured root_dir does not exist: {root_dir}")

        annotation_file = (root_dir / paths.get("annotation_file", "data/raw/iam_words/words.txt")).resolve()
        image_root = (root_dir / paths.get("image_root", "data/raw/iam_words/words")).resolve()
        processed_dir = (root_dir / paths.get("processed_dir", "data/processed")).resolve()
        splits_path = (root_dir / paths.get("splits_path", "data/processed/splits.json")).resolve()
        vocab_path = (root_dir / paths.get("vocab_path", "data/processed/vocabulary.json")).resolve()

        if not annotation_file.exists():
            fallback = root_dir / "data/raw/iam_words/words_new.txt"
            if fallback.exists():
                raise FileNotFoundError(
                    "The required IAM annotation file was not found at "
                    f"{annotation_file}. The actual project data currently contains "
                    f"{fallback}. Rename or re-link the annotation file to "
                    f"{annotation_file} without duplicating the dataset."
                )
            raise FileNotFoundError(f"IAM annotation file not found: {annotation_file}")

        return cls(
            root_dir=root_dir,
            annotation_file=annotation_file,
            image_root=image_root,
            processed_dir=processed_dir,
            splits_path=splits_path,
            vocab_path=vocab_path,
            seed=int(project.get("seed", split_cfg.get("seed", 42))),
            image_height=int(preprocess.get("image_height", 32)),
            batch_size=int(training.get("batch_size", 8)),
            epochs=int(training.get("epochs", 20)),
            learning_rate=float(training.get("learning_rate", 3e-4)),
            weight_decay=float(training.get("weight_decay", 1e-5)),
            cnn_channels=tuple(int(v) for v in model.get("cnn_channels", [32, 64, 128])),
            rnn_hidden_size=int(model.get("rnn_hidden_size", 128)),
            rnn_num_layers=int(model.get("rnn_num_layers", 2)),
            dropout=float(model.get("dropout", 0.25)),
            include_err=bool(annotation.get("include_err", False)),
            unknown_character_policy=str(annotation.get("unknown_character_policy", "replace_with_unk")),
            normalize_mean=float(preprocess.get("normalize_mean", 0.5)),
            normalize_std=float(preprocess.get("normalize_std", 0.5)),
            pad_value=float(preprocess.get("pad_value", 0.0)),
            num_workers=int(training.get("num_workers", 0)),
            train_ratio=float(split_cfg.get("train_ratio", 0.8)),
            val_ratio=float(split_cfg.get("val_ratio", 0.1)),
            test_ratio=float(split_cfg.get("test_ratio", 0.1)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "annotation_file": str(self.annotation_file),
            "image_root": str(self.image_root),
            "processed_dir": str(self.processed_dir),
            "splits_path": str(self.splits_path),
            "vocab_path": str(self.vocab_path),
            "seed": self.seed,
            "image_height": self.image_height,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "cnn_channels": list(self.cnn_channels),
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_num_layers": self.rnn_num_layers,
            "dropout": self.dropout,
            "include_err": self.include_err,
            "unknown_character_policy": self.unknown_character_policy,
            "normalize_mean": self.normalize_mean,
            "normalize_std": self.normalize_std,
            "pad_value": self.pad_value,
            "num_workers": self.num_workers,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.annotation_file = Path(self.annotation_file)
        self.image_root = Path(self.image_root)
        self.processed_dir = Path(self.processed_dir)
        self.splits_path = Path(self.splits_path)
        self.vocab_path = Path(self.vocab_path)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        if self.unknown_character_policy != "replace_with_unk":
            raise ValueError(
                "Only annotation.unknown_character_policy='replace_with_unk' is currently supported."
            )
        if self.normalize_std <= 0:
            raise ValueError("preprocessing.normalize_std must be greater than zero.")


def get_project_config(config_path: str | Path | None = None) -> ProjectConfig:
    """Load a YAML config file, falling back to the default project config."""

    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Project config file not found: {config_path}")
    return ProjectConfig.from_yaml(config_path)


def get_env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)
