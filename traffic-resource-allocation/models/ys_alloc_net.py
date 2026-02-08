import torch
import torch.nn as nn
import torch.nn.functional as F

class YSAllocNet(nn.Module):
    """
    입력: x (B,N,T,1)  (log1p scaled)
    출력: alloc (B,N)  (softmax distribution, sum=1)
    """
    def __init__(self, window_size: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)  # per-node score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x[..., 0]  # (B,N,T)

        last = xt[:, :, -1]
        mean = xt.mean(dim=2)
        std  = xt.std(dim=2)
        slope = (xt[:, :, -1] - xt[:, :, 0]) / max(self.window_size - 1, 1)

        feats = torch.stack([last, mean, std, slope], dim=-1)  # (B,N,4)
        scores = self.mlp(feats).squeeze(-1)                   # (B,N)
        alloc = F.softmax(scores, dim=1)
        return alloc
