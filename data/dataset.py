import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Literal
from pathlib import Path
import torch.nn.functional as F
import random
import math

def mutate_protein(wt_seq: str, mutation_position: int, new_aa: str) -> str:
    """
    Mutate a protein sequence at a given 0-indexed mutation_positionition.

    Args:
        wt_seq (str): Wild-type protein sequence.
        mutation_position (int): Position to mutate.
        new_aa (str): New amino acid, '*' represents stop codon.

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


# OLD ONE_HOT ENCODING THAT STORES IN E.G. 500 SEQUENCE LENGTH BLOCKS AND AVERAGING
# def one_hot_encode(seq: str, max_seq_length=500) -> torch.Tensor:
#     # encode proteins into a fixed length of `max_seq_length` * 20
#     # pad the sequence to fit within `max_seq_length` blocks (e.g. 5 -> 500, 525 -> 1000)
#     seq = seq + ' ' * ((max_seq_length - (len(seq) % max_seq_length)) % max_seq_length)
    
#     AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
#     AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    
#     idxs = torch.tensor([AA_TO_IDX.get(aa, len(AMINO_ACIDS)) for aa in seq])
#     # Anything mapped to 20 (unknowns) will create a 21-dim vector
#     one_hot_matrix = F.one_hot(idxs, num_classes=len(AMINO_ACIDS)+1)[:, :20].float()
#     flattened = one_hot_matrix.flatten()
#     split = flattened.split(max_seq_length * 20)
    
#     return torch.stack(split).mean(dim=0)

def center_protein_on_mutation(seq: str, mutation_position: int, segment_length: int = 500, max_jitter: int = 0) -> str:
    """
    Extract a fixed-length segment of protein sequence centered on a mutation.
    
    Args:
        seq (str): Protein sequence
        mutation_position (int): Index of the mutated residue (0-based)
        segment_length (int): Desired segment length
        max_jitter (int): Maximum random shift left/right
    
    Returns:
        str: A substring of length `segment_length`, padded with spaces if needed
    """
    
    # if sequence is shorter than needed, pad it
    if len(seq) < segment_length + 2 * max_jitter:
        pad_needed = (segment_length + 2 * max_jitter) - len(seq)
        left_pad = math.ceil(pad_needed / 2)
        right_pad = pad_needed - left_pad
        seq = " " * left_pad + seq + " " * right_pad
        mutation_position += left_pad  # shift index by left padding
    
    # Apply jitter
    jitter = random.randint(-max_jitter, max_jitter)
    mut_with_jitter = mutation_position + jitter
    
    # Desired start/end
    half_left = math.ceil(segment_length / 2)
    half_right = segment_length - half_left
    start = mut_with_jitter - half_left
    end = mut_with_jitter + half_right
    
    # Clamp to sequence boundaries
    if start < 0:
        end -= start  # shift window right
        start = 0
    if end > len(seq):
        start -= (end - len(seq))  # shift window left
        end = len(seq)

    segment = seq[start:end]
    return segment


def one_hot_encode(seq: str, mutation_position: int, positional_encoding: bool=False, max_seq_length: int=500) -> torch.Tensor:
    seq = seq + (" " * max(0, max_seq_length - len(seq)))
    mutation_position = mutation_position - 1 # reindex to 0-based
    
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    
    idxs = torch.tensor([AA_TO_IDX.get(aa, len(AMINO_ACIDS)) for aa in seq])
    # Anything mapped to 20 (unknowns) will create a 21-dim vector
    one_hot_matrix = F.one_hot(idxs, num_classes=len(AMINO_ACIDS)+1)[:, :20].float()
    flattened = one_hot_matrix.flatten()
    split = flattened.split(max_seq_length * 20)
    
    return torch.stack(split).mean(dim=0)


class ProteinDataset(Dataset):
    def __init__(self, split: Literal['train', 'test'], encoding: Literal['one-hot'] | None = None):
        file_dir = Path(__file__).resolve().parent
        self.split = split
        self.encoding = encoding
        self.variants = pd.read_csv(file_dir/f'{split}.csv')
        self.Sequences = pd.read_csv(file_dir/'Sequences.csv', index_col='ensp')
    
    def __len__(self):
        return self.variants.shape[0]
    
    def __getitem__(self, index):
        variant = self.variants.iloc[index]
        wt_seq = self.Sequences.loc[variant['ensp']]['wt_seq']
        mutation_position = variant['pos'] - 1 # -1 to reindex to 0
        variant_seq = mutate_protein(wt_seq, mutation_position, variant['alt_short'])
        
        if self.encoding == "one-hot":
            variant_seq = one_hot_encode(variant_seq)

        return variant_seq, variant['score'].item()
