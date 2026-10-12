# Handwritten Text Recognition from Scratch Using PyTorch

This project is a beginner-friendly implementation of a word-level handwritten text recognizer trained on the IAM handwriting database. The design follows a CNN + BiLSTM + CTC pipeline using PyTorch.

## Dataset note

The annotation file is read from `data/raw/iam_words/words.txt`; images are resolved below `data/raw/iam_words/words/`. The parser reads the actual IAM word record layout in this dataset: ID, segmentation status, gray level, bounding-box coordinates, grammatical tag, then the remaining transcription text. It handles `ok`, `err` (and `er`), reports malformed records, and never changes the original dataset.

The current validation run found 44,564 annotation rows: 38,305 `ok` and 6,259 `err`. With `annotation.include_err: false`, the default policy excludes `err` records. One `ok` image is unreadable (`a01/a01-117/a01-117-05-02.png`), leaving 38,304 usable samples.

## Project structure

- `src/config.py` loads YAML configuration and validates dataset paths.
- `src/dataset.py` parses IAM annotations, validates records, builds a PyTorch dataset, and pads variable-width batches.
- `src/model.py` implements the CNN + BiLSTM + CTC recognizer.
- `src/decoder.py` contains greedy CTC decoding utilities.
- `src/train.py`, `src/evaluate.py`, and `src/predict.py` provide the training, evaluation, and inference entry points.
- `configs/config.yaml` stores the user-configurable settings.
- `data/processed/` stores generated metadata, vocabulary files, and split information.

## Core design decisions

### Why CNN + BiLSTM + CTC?

- CNN layers learn local visual features from handwriting strokes.
- BiLSTM models the temporal order of those features across the word image.
- CTC loss removes the need for frame-level character labels; it is the standard choice for OCR and handwriting recognition at word level.

### Why not hardcode paths?

The project keeps configuration in YAML and resolves paths relative to the project root. This makes it portable across Ubuntu, Colab, and a local development machine.

### IAM split groups

The available `words.txt` records do not include a writer identifier, and the image directory prefix is not sufficient evidence that it identifies a writer. The generated split therefore groups by the full form ID (`a01-000u`) so that words from the same form cannot leak across train/validation/test. This does **not** guarantee writer-independent splits. Add verified IAM writer metadata before claiming writer-level isolation. The generated split manifest records this limitation.

## Running the project

1. Install dependencies:

   `pip install -r requirements.txt`

2. Confirm the IAM annotation file exists at `data/raw/iam_words/words.txt`.

3. Run the dataset validation command:

   `python -m src.dataset`

4. Prepare repeatable form-grouped splits and a vocabulary based on training text only:

   `python -m src.train --config configs/config.yaml --prepare-only`

5. Train the model:

   `python -m src.train --config configs/config.yaml`

6. After training, evaluate the isolated test split:

   `python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best_model.pt --split test`

Training writes per-epoch train and validation CTC losses to a timestamped JSONL file under `training_logs/`. The best checkpoint is selected using validation CTC loss; the test split is reserved for final evaluation.

## Validation and testing

The repository includes light smoke tests for dataset parsing, model output shapes, and CTC decoding.

Run:

`pytest`

## Notes for Colab

The training notebook is intentionally simple and export-friendly. For GPU training on Colab, install dependencies and then run the training script from the repository root.
