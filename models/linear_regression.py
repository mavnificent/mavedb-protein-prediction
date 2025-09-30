import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Prefer MPS (Apple Silicon) when available, then CUDA, otherwise CPU.
# torch.backends.mps.* checks ensure PyTorch was built with MPS support.
if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class LinearRegression(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # (batch,) instead of (batch,1)
    
    
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1, bias=True)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)  # (batch,) instead of (batch,1)


def train_linear_model(train_dataset, batch_size=64, lr=1e-3, weight_decay=1e-2, epochs=5, num_workers=None, device_override=None): #changed epochs from 10 for testing
    """
    weight_decay = L2 regularization strength (like Ridge alpha)
    """
    # Allow overriding device and DataLoader workers. Choose a reasonable default for num_workers
    # based on available CPUs; on macOS setting num_workers>0 may improve throughput but also
    # increase memory usage. For MPS/CPU it's often fine to use a small number.
    if device_override is None:
        _device = device
    else:
        _device = torch.device(device_override)

    if num_workers is None:
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1
        num_workers = min(4, max(0, cpu_count - 1))

    # pin_memory only helps for CUDA. It is harmless to leave False otherwise.
    pin_memory = True if _device.type == "cuda" else False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    
    # Figure out feature size from one example
    sample_x, _ = train_dataset[0]
    in_dim = sample_x.numel()
    
    model = SimpleMLP(in_dim).to(_device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    model.train()
    pbar = tqdm(range(epochs),total=epochs * len(train_loader))
    total_loss, total_samples = 0, 1
    for epoch in pbar:
        
        pbar.desc = f"Epoch {epoch+1}/{epochs} - Avg MSE: {total_loss / total_samples:.4f}"
        
        for X, y in train_loader:
            X = X.float().to(_device)
            y = y.float().to(_device)
            
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            
            # accumulate loss
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)
            
            # update progress bar with current batch loss
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            pbar.update()
    
    return model


def test_linear_model(test_dataset, model, batch_size, num_workers=0, device_override=None):
    if device_override is None:
        _device = device
    else:
        _device = torch.device(device_override)

    pin_memory = True if _device.type == "cuda" else False
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    model.eval()
    with torch.no_grad():
        preds, true = [], []
        for X, y in test_loader:
            X = X.float().to(_device)
            y = y.float().to(_device)
            p = model(X)
            preds.append(p)
            true.append(y)
        preds = torch.cat(preds).cpu().numpy()
        true  = torch.cat(true).cpu().numpy()
    
