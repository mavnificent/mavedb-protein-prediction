import argparse
import torch
from pathlib import Path
import pandas as pd
from utils.torch import get_torch_device
from utils.data import ProteinDataset
from utils.models import ESMRegressor
from utils.train_eval import train_model
from utils.model_io import save_model

def train(
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 4,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_name: str = "esmregressor.pt",
    data_dir: Path | str = "data",
    weight_dir: Path | str = "utils/weights",
):
    # Paths relative to project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = Path(data_dir) if isinstance(data_dir, Path) else PROJECT_ROOT / data_dir
    WEIGHT_DIR = Path(weight_dir) if isinstance(weight_dir, Path) else PROJECT_ROOT / weight_dir
    WEIGHT_DIR.mkdir(exist_ok=True)

    TRAIN_CSV = DATA_DIR / "train.csv"
    SEQUENCES_CSV = DATA_DIR / "Sequences.csv"
    MODEL_SAVE_PATH = WEIGHT_DIR / save_name

    device = get_torch_device()

    # Load data
    variants = pd.read_csv(TRAIN_CSV)
    Sequences = pd.read_csv(SEQUENCES_CSV, index_col="ensp")
    train_dataset = ProteinDataset(split="train", variants=variants, Sequences=Sequences, encoding=None)

    # Initialize model
    model = ESMRegressor(model_name=model_name).to(device)

    # Train
    trained_model = train_model(
        model=model,
        dataset=train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        device=device,
    )

    # Save model
    save_model(trained_model, MODEL_SAVE_PATH)
    return trained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train ESMRegressor on protein dataset.")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D", help="ESM model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers in dataloader")
    parser.add_argument("--save_name", type=str, default="esmregressor.pt", help="Name of file to save model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing CSV data files")
    parser.add_argument("--weight_dir", type=str, default="weights", help="Directory to save model weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        save_name=args.save_name,
        data_dir=args.data_dir,
        weight_dir=args.weight_dir,
    )
