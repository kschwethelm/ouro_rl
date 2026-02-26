"""Sequence packing for efficient RL forward passes.

Bin-packs prompt+response pairs into packed rows so multiple sequences share a single
forward pass. Combined with FlashAttention2's varlen support, this avoids wasting FLOPs
on padding tokens.

Requires: flash-attn >= 2.x and model loaded with attn_implementation="flash_attention_2".
FlashAttention auto-detects packing from position_ids that reset at sequence boundaries
(via transformers' _is_packed_sequence → flash_attn_varlen_func).
"""

from dataclasses import dataclass

import torch


@dataclass
class SequenceInfo:
    """Metadata for one original sequence within a packed row."""

    seq_idx: int  # original index in the flat sequence list
    row_idx: int  # which packed row this sequence landed in
    offset: int  # start position within the packed row
    total_len: int  # prompt_len + response_len
    resp_offset: int  # response start position within the packed row


@dataclass
class PackedBatch:
    """Packed rows ready for model forward passes.

    Each row is a 1-D tensor of token IDs (no tail padding) containing one or more
    concatenated sequences.  ``position_ids`` reset to 0 at each sequence boundary
    so FlashAttention can infer ``cu_seqlens`` for block-diagonal attention.

    Row length is bounded by ``pack_len`` (not ``max_model_len``): individual
    sequence positions stay within the model's context window, but a row can
    concatenate many sequences to improve GPU utilization.
    """

    # Per-row tensors (variable length — processed one at a time with batch_size=1).
    row_ids: list[torch.Tensor]  # token IDs for each packed row
    row_position_ids: list[torch.Tensor]  # position IDs (reset per sequence)

    # Per-sequence metadata (sorted by seq_idx).
    seq_infos: list[SequenceInfo]

    num_sequences: int
    num_rows: int


def pack_sequences(
    prompt_ids_list: list[list[int]],
    response_ids_list: list[list[int]],
    max_pack_len: int,
    max_seq_len: int | None = None,
) -> PackedBatch:
    """Bin-pack prompt+response pairs into packed rows using first-fit-decreasing.

    Each packed row's total token count is ≤ ``max_pack_len``.  Sequences are never
    split across rows.  Individual sequences are truncated to ``max_seq_len``
    (defaults to ``max_pack_len`` for backward compatibility).

    Args:
        prompt_ids_list: Token IDs for each prompt.
        response_ids_list: Token IDs for each response.
        max_pack_len: Maximum tokens per packed row (controls GPU saturation).
        max_seq_len: Maximum tokens per individual sequence (model context window).

    Returns:
        PackedBatch with packed rows and per-sequence metadata.
    """
    if max_seq_len is None:
        max_seq_len = max_pack_len

    n = len(prompt_ids_list)
    assert n == len(response_ids_list)

    # Compute total lengths and sort descending for better packing.
    lengths = [len(prompt_ids_list[i]) + len(response_ids_list[i]) for i in range(n)]
    sorted_indices = sorted(range(n), key=lambda i: lengths[i], reverse=True)

    # First-fit-decreasing bin packing.
    # Each bin: (current_length, list of (seq_idx, offset_in_row))
    bins: list[tuple[int, list[tuple[int, int]]]] = []

    for si in sorted_indices:
        seq_len = lengths[si]
        if seq_len > max_seq_len:
            # Truncate response to fit within model context window.
            seq_len = max_seq_len

        # Find first bin that fits.
        placed = False
        for bin_idx, (used, items) in enumerate(bins):
            if used + seq_len <= max_pack_len:
                items.append((si, used))
                bins[bin_idx] = (used + seq_len, items)
                placed = True
                break
        if not placed:
            bins.append((seq_len, [(si, 0)]))

    # Build packed row tensors and sequence metadata.
    row_ids_list: list[torch.Tensor] = []
    row_pos_list: list[torch.Tensor] = []
    seq_infos: list[SequenceInfo] = [None] * n  # type: ignore[list-item]

    for row_idx, (_used, items) in enumerate(bins):
        tokens: list[int] = []
        positions: list[int] = []

        for seq_idx, offset in items:
            prompt = prompt_ids_list[seq_idx]
            response = response_ids_list[seq_idx]
            total_len = len(prompt) + len(response)

            # Truncate if needed (same policy as pad_token_id_pairs).
            if total_len > max_seq_len:
                max_resp = max_seq_len - len(prompt)
                if max_resp <= 0:
                    prompt = prompt[-(max_seq_len - 1) :]
                    response = response[:1]
                else:
                    response = response[:max_resp]
                total_len = len(prompt) + len(response)

            tokens.extend(prompt)
            tokens.extend(response)
            positions.extend(range(total_len))

            seq_infos[seq_idx] = SequenceInfo(
                seq_idx=seq_idx,
                row_idx=row_idx,
                offset=offset,
                total_len=total_len,
                resp_offset=offset + len(prompt),
            )

        row_ids_list.append(torch.tensor(tokens, dtype=torch.long))
        row_pos_list.append(torch.tensor(positions, dtype=torch.long))

    assert all(info is not None for info in seq_infos)

    return PackedBatch(
        row_ids=row_ids_list,
        row_position_ids=row_pos_list,
        seq_infos=seq_infos,
        num_sequences=n,
        num_rows=len(bins),
    )
