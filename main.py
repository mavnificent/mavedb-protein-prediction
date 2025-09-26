from data import sequence_builder
from data.dataset import ProteinDataset
from pathlib import Path
from torch.utils.data import DataLoader
from models.linear_regression import train_linear_model
import pandas as pd


if __name__ == '__main__':
    if not Path('data/Sequences.csv').exists():
        sequence_builder.build_sequence_csv()
        
        
    data_dir=Path('./data')
    variants = pd.read_csv(data_dir/f'{'train'}.csv')
    Sequences = pd.read_csv(data_dir/'Sequences.csv', index_col='ensp')

    train_dataset = ProteinDataset(split="train", variants=variants, Sequences=Sequences, encoding="one-hot")

    train_linear_model(train_dataset=train_dataset, batch_size=2048)
