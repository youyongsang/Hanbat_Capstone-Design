import torch

def kl_div(y: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(y || pred) 평균
    y, pred: (B,N), sum=1
    """
    y = torch.clamp(y, eps, 1.0)
    pred = torch.clamp(pred, eps, 1.0)
    return (y * (y.log() - pred.log())).sum(dim=1).mean()

def mse(y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.mean((y - pred) ** 2)

def mae(y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y - pred))

def jain_fairness(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Jain's fairness index. (B,N) nonnegative
    """
    num = (x.sum(dim=1) ** 2)
    den = x.shape[1] * (x.pow(2).sum(dim=1) + eps)
    return (num / den).mean()

def max_share(x: torch.Tensor) -> torch.Tensor:
    return x.max(dim=1).values.mean()
