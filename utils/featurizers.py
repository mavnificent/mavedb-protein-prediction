import torch
import math
import random
import torch.nn.functional as F


def mutate_protein(wt_seq: str, mutation_position: int, new_aa: str) -> str:
    """
    Mutate a protein sequence at a given 0-indexed mutation_positionition.

    Args:
        wt_seq (str): Wild-type protein sequence.
        mutation_position (int): Position to mutate.
        new_aa (str): New amino acid; '*' represents stop codon.

    Raises:
        ValueError: If `mutation_position` is outside the sequence bounds.

    Returns:
        str: Mutated protein sequence.
    """
    if mutation_position < 0 or mutation_position >= len(wt_seq):
        raise ValueError(f"Position {mutation_position+1} is out of bounds for sequence of length {len(wt_seq)}")

    if new_aa == '*':
        return wt_seq[:mutation_position]
    
    return wt_seq[:mutation_position] + new_aa + wt_seq[mutation_position + 1:]



def get_sinusoidal_positional_encoding(seq_len: int, d_model: int = 20) -> torch.Tensor:
    """
    Sinusoidal positional encodings (seq_len, d_model).
    d_model must match one-hot size if we want to add directly.
    """
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # even dims
    pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
    return pe


def center_protein_on_mutation(
    seq: str,
    mutation_position: int,
    segment_length: int = 500,
    max_jitter: int = 0,
    positional_encoding: torch.Tensor | None = None
):
    """
    Extract a fixed-length segment of protein sequence centered on a mutation.
    Pads with spaces (" ") if needed. Optionally slices positional encodings.
    """
    if len(seq) < segment_length + 2 * max_jitter:
        pad_needed = (segment_length + 2 * max_jitter) - len(seq)
        left_pad = math.ceil(pad_needed / 2)
        right_pad = pad_needed - left_pad
        seq = " " * left_pad + seq + " " * right_pad
        mutation_position += left_pad
        if positional_encoding is not None:
            pad_pe = torch.zeros((left_pad + right_pad, positional_encoding.size(1)))
            positional_encoding = torch.cat(
                [pad_pe[:left_pad], positional_encoding, pad_pe[left_pad:]], dim=0
            )

    jitter = random.randint(-max_jitter, max_jitter)
    mut_with_jitter = mutation_position + jitter

    half_left = math.ceil(segment_length / 2)
    half_right = segment_length - half_left
    start = mut_with_jitter - half_left
    end = mut_with_jitter + half_right

    if start < 0:
        end -= start
        start = 0
    if end > len(seq):
        start -= (end - len(seq))
        end = len(seq)

    segment = seq[start:end]
    segment_pe = positional_encoding[start:end] if positional_encoding is not None else None
    return segment, segment_pe


def one_hot_encode(seq: str, positional_encoding: torch.Tensor | None = None) -> torch.Tensor:
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    idxs = torch.tensor([AA_TO_IDX.get(aa, len(AMINO_ACIDS)) for aa in seq])
    # Extra class is for unknown amino acids that gets turned into all 0s after splicing
    one_hot_matrix = F.one_hot(idxs, num_classes=len(AMINO_ACIDS)+1)[:, :20].float()
    if positional_encoding is not None:
        if positional_encoding.shape != one_hot_matrix.shape:
            raise ValueError(f"Shape mismatch: one-hot {one_hot_matrix.shape}, PE {positional_encoding.shape}")
        one_hot_matrix = one_hot_matrix + positional_encoding
    return one_hot_matrix.flatten()