import torch

@torch.no_grad()
def proportional_last_step(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_last = x[:, :, -1, 0]  # (B,N) log1p
    bytes_last = torch.expm1(x_last).clamp_min(0.0)
    return bytes_last / (bytes_last.sum(dim=1, keepdim=True) + eps)
