from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.config import get_project_config
from src.model import CNNBiLSTMCTC
from src.train import build_data_loaders, run_epoch


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    blank_index: int,
    device: torch.device,
) -> float:
    criterion = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    return run_epoch(model, loader, criterion, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained recognizer on a held-out split.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    config = get_project_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = config.root_dir / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    loaders, vocabulary, _ = build_data_loaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["vocabulary"] != vocabulary:
        raise ValueError(
            "Checkpoint vocabulary does not match the current training split vocabulary."
        )

    model = CNNBiLSTMCTC(
        image_height=config.image_height,
        cnn_channels=config.cnn_channels,
        rnn_hidden_size=config.rnn_hidden_size,
        rnn_num_layers=config.rnn_num_layers,
        vocab_size=len(vocabulary),
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss = evaluate(model, loaders[args.split], vocabulary.index("<blank>"), device)
    print(f"{args.split} CTC loss: {loss:.4f}")


if __name__ == "__main__":
    main()
