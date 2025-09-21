from data import sequence_builder
from data.dataset import ProteinDataset
from pathlib import Path

if not Path('data/Sequences.csv').exists():
    sequence_builder.build_sequence_csv()

one_hot_dataset = ProteinDataset('train', encoding='one-hot')

