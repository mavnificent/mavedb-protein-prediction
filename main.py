from data import sequence_builder
from pathlib import Path

if not Path('data/train_seq.csv').exists():
    sequence_builder.build_sequence_csv('train')
    
if not Path('data/test_seq.csv').exists():
    sequence_builder.build_sequence_csv('test')

