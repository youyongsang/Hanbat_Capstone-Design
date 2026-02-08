# Utils Module

훈련과 평가를 위한 유틸리티 함수와 클래스들을 제공합니다.

## 파일 설명

### alloc_labels.py
리소스 할당 라벨을 생성합니다.

**함수: `alloc_from_log1p(log1p_vals, eps=1e-8)`**

Log1p로 정규화된 값을 할당 분포로 변환합니다.

```python
log1p_vals: (B, N) - log1p(bytes)
→ returns: (B, N) - allocation (sum=1)
```

**과정:**
1. 역 로그 변환: `expm1()` → 원래 바이트 값
2. 정규화: 각 샘플별 합으로 나누기 → 할당 분포

---

**함수: `make_alloc_label(x_data, y_data, mode="next_step", eps=1e-8)`**

훈련용 할당 라벨을 생성합니다.

**입력:**
- `x_data`: (B, N, T, 1) - 시계열 입력 데이터
- `y_data`: (B, N) - 다음 시점 예측값 (mode=next_step일 때 필요)
- `mode`: 라벨 생성 방식
  - **next_step**: y_data (다음 시점 트래픽) 기반 → 미래 트래픽 예측
  - **last_step**: x_data 마지막 스텝 기반 → 현재 트래픽 기반

**반환:**
- (B, N) - 할당 분포 (각 행의 합 = 1)

**예제:**
```python
alloc = make_alloc_label(x_train, y_train, mode="next_step")
# 다음 시점 트래픽에 따른 최적 할당
```

---

### dataset.py
PyTorch 학습용 Dataset 클래스입니다.

**클래스: `TrafficAllocDataset`**

NPZ 파일로부터 데이터를 로드하고 할당 라벨을 생성합니다.

**초기화:**
```python
dataset = TrafficAllocDataset(npz_path, label_mode="next_step")
```

**파라미터:**
- `npz_path`: 처리된 NPZ 파일 경로
  - 필수: `x_data` (B, N, T, 1), `y_data` (B, N)
- `label_mode`: "next_step" 또는 "last_step"

**사용 예제:**
```python
from torch.utils.data import DataLoader
from utils.dataset import TrafficAllocDataset

train_ds = TrafficAllocDataset("data/processed/traffic_data_train.npz")
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

for x, alloc_label in train_loader:
    print(f"x shape: {x.shape}, alloc shape: {alloc_label.shape}")
    # x: (64, 10, 20, 1), alloc_label: (64, 10)
```

**출력:**
- Batch로부터 `(x, alloc_label)` 튜플 반환
- 모두 PyTorch Tensor (float32)

---

### metrics.py
모델 평가 메트릭들을 제공합니다.

**함수 목록:**

#### 1. `kl_div(y, pred, eps=1e-8)`
KL Divergence (Kullback-Leibler Divergence)

$$KL(y \parallel pred) = \sum_i y_i \log\left(\frac{y_i}{pred_i}\right)$$

**의미:** 두 분포 간의 차이를 측정 (값이 작을수록 좋음)

**입력:** (B, N) 확률 분포
**출력:** 스칼라

---

#### 2. `mse(y, pred)`
Mean Squared Error

$$MSE = \frac{1}{BN} \sum (y - pred)^2$$

**의미:** 예측값과 실제값의 편차 제곱 평균

**입력:** (B, N) 임의의 값
**출력:** 스칼라

---

#### 3. `mae(y, pred)`
Mean Absolute Error

$$MAE = \frac{1}{BN} \sum |y - pred|$$

**의미:** 예측값과 실제값의 절대 편차 평균

**입력:** (B, N) 임의의 값
**출력:** 스칼라

---

#### 4. `jain_fairness(x, eps=1e-8)`
Jain's Fairness Index

$$J = \frac{(\sum_i x_i)^2}{N \sum_i x_i^2}$$

**의미:** 리소스 할당의 공정성 측정 (0~1, 1에 가까울수록 공정함)
- 1: 모든 노드에 동등하게 할당
- 0에 가까움: 일부 노드에 편중

**입력:** (B, N) 음이 아닌 값
**출력:** 스칼라 (배치 평균)

---

#### 5. `max_share(x)`
Maximum Share (최대 점유율)

$$MaxShare = max(x)$$

**의미:** 가장 많이 할당된 노드의 점유율 (낮을수록 균형잡힘)
- 0.5 이상: 일부 노드에 편중
- 0.1 이상: 비교적 균형

**입력:** (B, N) 음이 아닌 값
**출력:** 스칼라 (배치 평균)

---

## 메트릭 해석 가이드

| 메트릭 | 목표값 | 설명 |
|--------|--------|------|
| KL Divergence | ↓ 작음 | 분포 예측 정확도 |
| MSE | ↓ 작음 | 값 예측 정확도 |
| MAE | ↓ 작음 | 평균 편차 |
| Jain's Index | ↑ 1에 가까움 | 할당 공정성 |
| Max Share | ↓ 작음 | 할당 균형도 |

---

## 통합 사용 예제

```python
import torch
from torch.utils.data import DataLoader
from utils.dataset import TrafficAllocDataset
from utils.metrics import kl_div, jain_fairness, max_share
from models.ys_alloc_net import YSAllocNet

# 데이터 로드
dataset = TrafficAllocDataset("data/processed/traffic_data_test.npz")
loader = DataLoader(dataset, batch_size=32)

# 모델 로드
model = YSAllocNet(window_size=20, hidden=64)
model.load_state_dict(torch.load("artifacts/ys_alloc_net.pt"))
model.eval()

# 평가
total_kl = 0
total_jain = 0
total_maxshare = 0

with torch.no_grad():
    for x, y in loader:
        pred = model(x)  # (B, N)
        total_kl += kl_div(y, pred).item()
        total_jain += jain_fairness(pred).item()
        total_maxshare += max_share(pred).item()

n_batches = len(loader)
print(f"KL: {total_kl/n_batches:.6f}")
print(f"Jain: {total_jain/n_batches:.6f}")
print(f"MaxShare: {total_maxshare/n_batches:.6f}")
```

