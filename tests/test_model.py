import torch

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

    assert logits.shape[0] == 12
    assert logits.shape[1] == 2
    assert logits.shape[2] == 20
    assert model.compute_output_lengths([48, 64]).tolist() == [12, 16]
