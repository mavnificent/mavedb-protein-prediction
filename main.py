from data import sequence_builder
from pathlib import Path

if not Path('data/Sequences.csv').exists():
    sequence_builder.build_sequence_csv()

