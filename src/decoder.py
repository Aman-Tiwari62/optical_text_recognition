from __future__ import annotations

import torch


def greedy_ctc_decode(logits: torch.Tensor, blank_index: int = 0) -> list[int]:
    """Collapse repeated symbols according to CTC rules and remove blanks."""

    if logits.dim() == 3:
        if logits.shape[0] != 1 and logits.shape[1] != 1:
            logits = logits.argmax(dim=-1)
            return _collapse_ctc_tokens(logits.tolist(), blank_index)
        logits = logits.squeeze(0)
    if logits.dim() == 2:
        best = logits.argmax(dim=-1)
        return _collapse_ctc_tokens(best.tolist(), blank_index)
    raise ValueError(f"Expected logits shaped as (T, C) or (T, B, C); got {tuple(logits.shape)}")


def _collapse_ctc_tokens(token_ids: list[int] | torch.Tensor, blank_index: int = 0) -> list[int]:
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in token_ids:
        if isinstance(token_id, torch.Tensor):
            token_id = int(token_id.item())
        if token_id == blank_index:
            previous = None
            continue
        if previous is not None and token_id == previous:
            continue
        collapsed.append(int(token_id))
        previous = int(token_id)
    return collapsed


def decode_token_ids(token_ids: list[int], index_to_char: dict[int, str], blank_index: int = 0) -> str:
    characters: list[str] = []
    for token_id in _collapse_ctc_tokens(token_ids, blank_index=blank_index):
        characters.append(index_to_char.get(token_id, "?"))
    return "".join(characters)
