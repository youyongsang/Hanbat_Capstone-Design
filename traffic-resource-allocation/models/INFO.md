# Models Module

트래픽 리소스 할당을 위한 딥러닝 모델과 Rule-based 베이스라인을 포함합니다.

## 파일 설명

### rule_based.py
Rule-based 베이스라인 모델입니다.

**함수: `proportional_last_step(x, eps=1e-8)`**

마지막 시간 단계의 트래픽 량에 비례하여 리소스를 할당합니다.

**입력:**
- `x`: (B, N, T, 1) 텐서 - log1p로 정규화된 트래픽 데이터
  - B: 배치 크기
  - N: 노드 개수
  - T: 시계열 길이
  - 1: 특성 채널

**출력:**
- (B, N) - softmax 정규화된 할당 분포 (각 행의 합 = 1)

**알고리즘:**
1. 마지막 시간 단계 추출: `x[:, :, -1, 0]`
2. 역 log 변환: `expm1()`로 원래 바이트 값 복구
3. Softmax 정규화: 비율로 변환

---

### ys_alloc_net.py
Deep Learning 기반 리소스 할당 모델입니다.

**클래스: `YSAllocNet`**

시계열 패턴을 학습하여 리소스 할당을 최적화하는 신경망입니다.

**초기화 파라미터:**
```python
YSAllocNet(window_size, hidden=64, dropout=0.1)
```
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| window_size | 필수 | 시계열 윈도우 크기 |
| hidden | 64 | MLP 은닉층 크기 |
| dropout | 0.1 | Dropout 비율 |

**입력:**
- `x`: (B, N, T, 1) - log1p 정규화된 트래픽 데이터

**출력:**
- (B, N) - softmax 정규화된 할당 분포

**특성 엔지니어링:**

4개의 시계열 특성을 추출합니다:

1. **Last**: 마지막 값 (`xt[:, :, -1]`)
   - 현재 트래픽 양
2. **Mean**: 평균값 (`xt.mean(dim=2)`)
   - 평균 트래픽 수준
3. **Std**: 표준편차 (`xt.std(dim=2)`)
   - 변동성 (높을수록 불안정)
4. **Slope**: 추세 (`(last - first) / (T-1)`)
   - 증감 추세

**모델 구조:**
```
Input (B, N, 4)
  ↓
Linear(4 → hidden)
  ↓
ReLU
  ↓
Dropout
  ↓
Linear(hidden → 1)
  ↓
Softmax (dim=1)
  ↓
Output (B, N)
```

**설명:**
- 각 노드별로 4개 특성을 입력
- MLP를 통해 할당 점수 계산
- Softmax로 정규화하여 확률 분포 생성

---

## 모델 비교

| 항목 | Rule-based | YSAllocNet |
|------|-----------|-----------|
| 방식 | 휴리스틱 | 신경망 학습 |
| 계산량 | 매우 적음 | 중간 |
| 적응성 | 고정 규칙 | 학습 가능 |
| 해석성 | 높음 | 낮음 |
| 성능 | 기본 수준 | 최적화 가능 |

---

## 사용 방법

**Rule-based:**
```python
from models.rule_based import proportional_last_step
import torch

x = torch.randn(32, 10, 20, 1)  # (B=32, N=10, T=20, F=1)
alloc = proportional_last_step(x)  # (32, 10)
```

**YSAllocNet:**
```python
from models.ys_alloc_net import YSAllocNet
import torch

model = YSAllocNet(window_size=20, hidden=64, dropout=0.1)
x = torch.randn(32, 10, 20, 1)
alloc = model(x)  # (32, 10)
```

---

## 출력 특성

두 모델 모두:
- 각 노드의 할당량을 **확률 분포**로 반환
- 모든 행의 합 = 1.0 (정규화됨)
- 값 범위: [0, 1]
- 높을수록 더 많은 리소스 할당

