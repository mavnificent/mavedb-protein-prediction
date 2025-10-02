import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Type, Dict
from tqdm import tqdm
from utils import get_torch_device
from utils.model import save_model, load_model
from pathlib import Path


DEVICE = get_torch_device()
WEIGHT_DIR = Path(__file__).resolve().parent / Path('weights')
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(cls: Type[nn.Module]) -> Type[nn.Module]:
    """Decorator to register a model by class name."""
    MODEL_REGISTRY[cls.__name__] = cls
    return cls


@register_model
class LinearRegression(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # shape (batch,)


@register_model
class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape (batch,)


def train_model(train_dataset,
                model_name: str,
                save_name: str,
                batch_size: int = 64,
                epochs: int = 10,
                lr: float = 1e-3,
                weight_decay: float = 1e-2,
                num_workers: int = 4,
                device: str | torch.device = DEVICE,
                force_retrain: bool = False,
                **model_kwargs) -> nn.Module:
    """
    Train a model selected from the registry.
    Additional kwargs are passed to the model constructor.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    assert save_name.endswith('.pt')
    
    # infer input dimension from one example
    sample_x, _ = train_dataset[0]
    in_dim = sample_x.numel()
    model = MODEL_REGISTRY[model_name](in_dim, **model_kwargs).to(device)
    
    if (WEIGHT_DIR / Path(save_name)).exists() and not force_retrain:
        print(f"Loading pretrained {model_name} from {WEIGHT_DIR / Path(save_name)}")
        return load_model(model=model, path=WEIGHT_DIR / Path(save_name), device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    pbar = tqdm(total=epochs * len(train_loader))
    total_loss, total_samples = 0, 0

    for epoch in range(epochs):
        for X, y in train_loader:
            X = X.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

            pbar.set_description(f"Epoch {epoch+1}/{epochs} - Avg MSE: {total_loss / total_samples:.4f}")
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            pbar.update()
    save_model(model, WEIGHT_DIR / Path(save_name))
    return model


def test_linear_model(test_dataset, model, batch_size, device=DEVICE):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        preds, true = [], []
        for X, y in test_loader:
            X = X.float().to(device)
            y = y.float().to(device)
            p = model(X)
            preds.append(p)
            true.append(y)
        preds = torch.cat(preds).cpu().numpy()
        true  = torch.cat(true).cpu().numpy()