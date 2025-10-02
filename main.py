from data import sequence_builder, scoresets_builder
from data.dataset import ProteinDataset
from pathlib import Path
from torch.utils.data import DataLoader
from models.linear_regression import train_model
import pandas as pd


if __name__ == '__main__':
    if not Path('data/Sequences.csv').exists():
        sequence_builder.build_sequence_csv()
    
    if not Path('data/Info.csv').exists():
        scoresets_builder.build_scoreset_csv()
        
    data_dir=Path('./data')
    variants = pd.read_csv(data_dir/f'{'train'}.csv')
    Sequences = pd.read_csv(data_dir/'Sequences.csv', index_col='ensp')

    train_dataset = ProteinDataset(split="train", variants=variants, Sequences=Sequences, encoding="one-hot")

    train_model(train_dataset=train_dataset, model_name="LinearRegression", save_name="one-hot-LinearRegression.pt", batch_size=2048)
