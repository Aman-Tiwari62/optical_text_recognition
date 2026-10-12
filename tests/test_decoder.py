import torch

from src.decoder import decode_token_ids, greedy_ctc_decode


def test_greedy_ctc_decode_collapses_repeats_and_removes_blanks():
    logits = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    tokens = greedy_ctc_decode(logits, blank_index=0)
    index_to_char = {1: "A", 2: "B"}

    assert tokens == [1, 2]
    assert decode_token_ids(tokens, index_to_char, blank_index=0) == "AB"
