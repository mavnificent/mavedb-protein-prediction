import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Literal
from pathlib import Path
import torch.nn.functional as F


def mutate_protein(wt_seq: str, pos: int, new_aa: str) -> str:
    """
    Mutate a protein sequence at a given 1-indexed position.

    Args:
        wt_seq (str): Wild-type protein sequence.
        pos (int): Position to mutate (1-indexed).
        new_aa (str): New amino acid, '*' represents stop codon.

    Raises:
        ValueError: If `pos` is outside the sequence bounds.

    Returns:
        str: Mutated protein sequence.
    """
    if pos < 1 or pos > len(wt_seq):
        raise ValueError(f"Position {pos} is out of bounds for sequence of length {len(wt_seq)}")

    if new_aa == '*':
        return wt_seq[:pos-1]
    
    return wt_seq[:pos-1] + new_aa + wt_seq[pos:]


def one_hot_encode(seq: str) -> torch.Tensor:
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    
    idxs = torch.tensor([AA_TO_IDX.get(aa, len(AMINO_ACIDS)) for aa in seq])
    # Anything mapped to 20 (unknowns) will create a 21-dim vector
    one_hot_matrix = F.one_hot(idxs, num_classes=len(AMINO_ACIDS)+1)[:, :20].float()
    return one_hot_matrix.flatten()


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
        variant_seq = mutate_protein(wt_seq, variant['pos'], variant['alt_short'])
        
        if self.encoding == "one-hot":
            variant_seq = one_hot_encode(variant_seq)

        return variant_seq, variant['score'].item()
