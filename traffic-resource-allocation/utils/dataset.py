import numpy as np
import torch
from torch.utils.data import Dataset
from utils.alloc_labels import make_alloc_label

class TrafficAllocDataset(Dataset):
    """
    npz 안에 x_data (B,N,T,1), y_data (B,N)가 있다고 가정.
    alloc 라벨은 y_data(next_step) 또는 x 마지막스텝(last_step)으로 생성.
    """
    def __init__(self, npz_path: str, label_mode: str = "next_step"):
        data = np.load(npz_path)
        self.x = data["x_data"].astype(np.float32)
        self.y = data["y_data"].astype(np.float32) if "y_data" in data else None
        self.alloc = make_alloc_label(self.x, self.y, mode=label_mode).astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.alloc[idx])
