import torch
from torch.utils.data import Dataset
from typing import Literal
from utils.featurizers import (
    mutate_protein,
    one_hot_encode,
    get_sinusoidal_positional_encoding,
    center_protein_on_mutation,
)


class ProteinDataset(Dataset):
    def __init__(self, 
                 split: Literal['train', 'test'], 
                 variants, 
                 Sequences, 
                 encoding: Literal['one-hot', 'one-hot-segment'] | None = None):
        self.split = split
        self.encoding = encoding        
        self.variants = variants
        self.Sequences = Sequences
        
    def __len__(self):
        return self.variants.shape[0]
    
    def __getitem__(self, index):
        variant = self.variants.iloc[index]
        wt_seq = self.Sequences.loc[variant['ensp']]['wt_seq']
        mutation_position = variant['pos'] - 1 # -1 to reindex to 0
        variant_seq = mutate_protein(wt_seq, mutation_position, variant['alt_short'])
        
        if self.encoding == "one-hot-segment":
            variant_pe = get_sinusoidal_positional_encoding(len(variant_seq), 20)
            variant_segment, variant_segment_pe = center_protein_on_mutation(seq=variant_seq, 
                                                                             mutation_position= mutation_position,
                                                                             segment_length=500,
                                                                             max_jitter=15,
                                                                             positional_encoding=variant_pe)
            variant_seq = one_hot_encode(seq=variant_segment, positional_encoding=variant_segment_pe) 
        
        elif self.encoding == "one-hot":
            variant_seq = variant_seq + " " * max(0, 4834 - len(variant_seq))
            variant_seq = one_hot_encode(seq=variant_seq)            

        return {
            "input": variant_seq, 
            "accession": variant['accession'],
            "score": torch.tensor(variant['score'], dtype=torch.float)
        }