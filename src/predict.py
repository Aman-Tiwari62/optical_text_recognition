from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.config import get_project_config
from src.decoder import decode_token_ids, greedy_ctc_decode


def predict_one(model: torch.nn.Module, image_path: str | Path, index_to_char: dict[int, str], blank_index: int = 0) -> str:
    image = Image.open(image_path).convert("L")
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    image_tensor = (image_tensor / 255.0 - 0.5) / 0.5
    with torch.no_grad():
        logits = model(image_tensor)
    tokens = greedy_ctc_decode(logits.squeeze(1), blank_index=blank_index)
    return decode_token_ids(tokens, index_to_char, blank_index=blank_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one handwritten word image.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--image", type=str, required=True, help="Image path to classify.")
    args = parser.parse_args()
    config = get_project_config(args.config)
    print(f"Prediction path: {args.image}")
    print(f"Configured root directory: {config.root_dir}")


if __name__ == "__main__":
    main()
