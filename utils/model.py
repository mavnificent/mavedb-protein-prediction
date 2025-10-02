import torch
import torch.nn as nn
from pathlib import Path

def save_model(model: nn.Module, path: str | Path) -> None:
    """Save model state_dict to the given path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved trained model to {path}.")

def load_model(model: nn.Module, path: str | Path, device: str | torch.device) -> nn.Module:
    """Load state_dict into a model from the given path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file found at {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded pretrained model at {path}.")
    return model
