import torch

from src.dataset import collate_iam_batch
from src.model import CNNBiLSTMCTC


def test_model_output_shape_and_sequence_length():
    model = CNNBiLSTMCTC(
        image_height=32,
        cnn_channels=(8, 16),
        rnn_hidden_size=12,
        rnn_num_layers=1,
        vocab_size=20,
        dropout=0.0,
    )
    images = torch.randn(2, 1, 32, 48)
    logits = model(images)

    assert logits.shape[0] == 24
    assert logits.shape[1] == 2
    assert logits.shape[2] == 20
    assert model.compute_output_lengths([48, 64]).tolist() == [24, 32]


def test_horizontal_downsampling_preserves_ctc_frames_for_short_words():
    model = CNNBiLSTMCTC(
        image_height=32,
        cnn_channels=(8, 16, 32),
        rnn_hidden_size=12,
        rnn_num_layers=1,
        vocab_size=20,
        dropout=0.0,
    )

    assert model.compute_output_lengths([28]).tolist() == [14]
    assert model.compute_output_lengths([47]).tolist() == [24]


def test_collate_lengths_match_model_for_odd_and_even_widths():
    model = CNNBiLSTMCTC(
        image_height=32,
        cnn_channels=(8, 16, 32),
        rnn_hidden_size=12,
        rnn_num_layers=1,
        vocab_size=20,
        dropout=0.0,
    )
    batch = collate_iam_batch(
        [
            {
                "image": torch.zeros(1, 32, 28),
                "target": torch.tensor([1, 2, 2, 3, 4, 5, 6, 7]),
                "target_text": "herring",
                "image_path": "herring.png",
                "word_id": "sample-1",
            },
            {
                "image": torch.zeros(1, 32, 47),
                "target": torch.tensor([1, 2]),
                "target_text": "go",
                "image_path": "go.png",
                "word_id": "sample-2",
            },
        ],
        blank_index=0,
    )

    logits = model(batch["images"])
    model_lengths = model.compute_output_lengths(batch["image_widths"])

    assert logits.shape == (24, 2, 20)
    assert model_lengths.tolist() == [14, 24]
    assert torch.equal(model_lengths, batch["input_lengths"])
    assert int(batch["input_lengths"][0]) >= 9
