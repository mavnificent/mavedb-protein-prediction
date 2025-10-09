from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from typing import Dict, Type, Literal, Optional

from utils.torch import get_torch_device
from utils.data import ProteinDataset
from utils.model_io import save_model
from utils.models import MODEL_REGISTRY, ESM_CHOICES


# -------------------------------
# Helper: build dataset and model
# -------------------------------
def build_model_and_dataset(
    model_name: str,
    data_dir: Path,
    split: Literal["train", "test"],
    esm_model: ESM_CHOICES | None = None,
    dataset_encoding: Literal["one-hot", "one-hot-segment"] | None = None
) -> tuple[nn.Module, ProteinDataset]:
    """
    Build dataset + model depending on type.

    For OneHot models:
      - Use encoding = 'one-hot-segment'
      - Pass inferred in_dim to model
    For ESM models:
      - Use encoding = None
      - Pass esm_model (defaults to 'facebook/esm2_t33_650M_UR50D')
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    TRAIN_CSV = data_dir / "train.csv"
    SEQUENCES_CSV = data_dir / "Sequences.csv"
    variants = pd.read_csv(TRAIN_CSV)
    Sequences = pd.read_csv(SEQUENCES_CSV, index_col="ensp")

    # Case 1: OneHot models
    if model_name in {"OneHotRegressor", "OneHotMLP"}:
        if dataset_encoding not in ["one-hot", "one-hot-segment"]:
            raise ValueError('A OneHot model must have dataset_encoding = either "one-hot" or "one-hot-segment").')
        
        dataset = ProteinDataset(split=split, variants=variants, Sequences=Sequences, encoding=dataset_encoding)

        # Infer input dimension from one sample
        sample = dataset[0]["input"]
        if isinstance(sample, torch.Tensor):
            in_dim = sample.shape[-1]
        else:
            raise TypeError("Dataset did not return tensor input for OneHot model.")

        model_cls = MODEL_REGISTRY[model_name]
        model = model_cls(in_dim=in_dim)

    # Case 2: Sequence-based models
    else:
        esm_model_options = ["facebook/esm2_t6_8M_UR50D","facebook/esm2_t12_35M_UR50D","facebook/esm2_t30_150M_UR50D","facebook/esm2_t33_650M_UR50D","facebook/esm2_t36_3B_UR50D"]
        if esm_model not in esm_model_options:
            raise ValueError(f"{esm_model} is not a valid ESM model-type. Options: {esm_model_options}")
            
        dataset = ProteinDataset(split=split, variants=variants, Sequences=Sequences, encoding=None)

        model_cls = MODEL_REGISTRY[model_name]
        model = model_cls(model_name=esm_model)
    
    return model, dataset


# -------------------------------
# Main train_model
# -------------------------------
def train_model(
    model_name: str,
    batch_size: int = 1,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    data_dir: Path | str = "data",
    weight_dir: Path | str = "utils/weights",
    save_name: str = "trained_model.pt",
    device: str | torch.device | None = None,
    esm_model: Optional[ESM_CHOICES] = None,
    dataset_encoding: Literal["one-hot", "one-hot-segment"] | None = None
):
    """
    Unified training entrypoint.
    """
    DATA_DIR = Path(data_dir)
    WEIGHT_DIR = Path(weight_dir)
    MODEL_SAVE_PATH = WEIGHT_DIR / save_name

    device = device or get_torch_device()

    # -------------------------------
    # Build model + dataset
    # -------------------------------
    model, dataset = build_model_and_dataset(model_name=model_name, 
                                             split="train",
                                             data_dir=DATA_DIR, 
                                             esm_model=esm_model, 
                                             dataset_encoding=dataset_encoding)
    model = model.to(device)

    # -------------------------------
    # Training setup
    # -------------------------------
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    sample_x = dataset[0]["input"]
    tensor_input = isinstance(sample_x, torch.Tensor)

    # -------------------------------
    # Train loop
    # -------------------------------
    model.train()
    pbar = tqdm(total=epochs * len(loader))
    total_loss, total_samples = 0, 0

    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            X = batch["input"]
            y = batch["score"].to(device)

            if tensor_input:
                X = X.to(device)

            preds = model(X).view(-1)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_samples += len(y)
            pbar.set_description(f"Epoch {epoch+1}/{epochs} - Avg MSE: {total_loss / total_samples:.4f}")
            pbar.update()

    pbar.close()

    save_model(model, MODEL_SAVE_PATH)
    return model
