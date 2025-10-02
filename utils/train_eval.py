import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.torch import get_torch_device

DEVICE = get_torch_device()

def train_model(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    device: str | torch.device = DEVICE,
):
    """
    Train a PyTorch model on the given dataset.

    Args:
        model: nn.Module, must output (batch_size, 1)
        dataset: PyTorch Dataset, returns either tensor or string
        batch_size: batch size for training
        epochs: number of epochs
        lr: learning rate
        weight_decay: weight decay for optimizer
        device: torch device
    Returns:
        trained model
    """
    model = model.to(device)
    model.train()
    
    # detect if dataset returns tensors or strings
    sample_x = dataset[0]["input"]
    tensor_input = isinstance(sample_x, torch.Tensor)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    pbar = tqdm(total=epochs * len(loader))
    total_loss, total_samples = 0, 0
    
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            X = batch["input"]
            y = batch["score"].to(device)
            if tensor_input:
                X.to(device)
                
            preds = model(X)
            
            loss = loss_fn(preds.squeeze(-1), y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            total_samples += len(y)
            pbar.set_description(f"Epoch {epoch+1}/{epochs} - Avg MSE: {total_loss / total_samples:.4f}")
            pbar.update()
    
    pbar.close()
    return model



# def test_linear_model(test_dataset, model, batch_size, device=DEVICE):
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     model.eval()
#     with torch.no_grad():
#         preds, true = [], []
#         for X, y in test_loader:
#             X = X.float().to(device)
#             y = y.float().to(device)
#             p = model(X)
#             preds.append(p)
#             true.append(y)
#         preds = torch.cat(preds).cpu().numpy()
#         true  = torch.cat(true).cpu().numpy()