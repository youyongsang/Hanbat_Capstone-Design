import numpy as np

def alloc_from_log1p(log1p_vals: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    log1p(bytes) (B,N) -> allocation distribution (B,N), sum=1
    """
    bytes_vals = np.expm1(log1p_vals)  # exp(x)-1
    denom = bytes_vals.sum(axis=1, keepdims=True)
    return bytes_vals / (denom + eps)

def make_alloc_label(x_data: np.ndarray, y_data: np.ndarray | None,
                     mode: str = "next_step", eps: float = 1e-8) -> np.ndarray:
    """
    mode:
      - next_step: y_data(다음 시점 트래픽) 기반 alloc 라벨
      - last_step: x_data 마지막 스텝 기반 alloc 라벨 (완전 현재 기반)
    """
    if mode == "next_step":
        if y_data is None:
            raise ValueError("mode=next_step 인데 y_data가 없습니다.")
        base = y_data
    elif mode == "last_step":
        base = x_data[:, :, -1, 0]  # (B,N)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return alloc_from_log1p(base, eps=eps)
