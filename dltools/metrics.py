import torch
from torch import tensor


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()


def avg_deviation(out, yb):
    return abs(out-yb).mean()


def rmsle(x: tensor, y: tensor) -> tensor:
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y
    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return torch.sqrt(((torch.log(x + 1) - torch.log(y + 1))**2).mean())
