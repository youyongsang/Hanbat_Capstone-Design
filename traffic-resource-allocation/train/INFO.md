# Training Module

트래픽 리소스 할당 모델을 훈련하고 평가하는 모듈입니다.

## 파일 설명

### train_ys_alloc.py
YSAllocNet 모델을 훈련합니다.

**기능:**
- 처리된 트래픽 데이터로 신경망 모델 훈련
- Rule-based 베이스라인(proportional_last_step)과 성능 비교
- KL Divergence를 주요 손실함수로 사용
- 여러 평가 메트릭으로 모델 성능 측정
- 최적 모델 자동 저장

**평가 메트릭:**
- **KL**: KL Divergence (분포 차이)
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **Jain**: Jain's Fairness Index (공정성 지표)
- **MaxShare**: 최대 점유율 (불균형 측정)

## 사용 방법

### 기본 실행
```bash
python train_ys_alloc.py
```

### 커맨드라인 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--label_mode` | next_step | 라벨 생성 방식 (next_step 또는 last_step) |
| `--epochs` | 20 | 훈련 에포크 수 |
| `--batch` | 64 | 배치 크기 |
| `--lr` | 1e-3 | 학습률 |
| `--hidden` | 64 | 은닉층 크기 |
| `--dropout` | 0.1 | Dropout 비율 |

### 예제
```bash
python train_ys_alloc.py --epochs 50 --batch 32 --lr 0.0005 --hidden 128
```

## 훈련 과정

1. **데이터 로드**: 처리된 훈련/테스트 NPZ 파일 로드
2. **베이스라인 평가**: Rule-based 모델 성능 측정
3. **모델 훈련**: 
   - Loss: KL Divergence
   - Optimizer: AdamW
   - 최적 모델만 저장
4. **평가**: 매 에포크마다 테스트 셋에서 성능 평가

## 출력

- **콘솔 출력**: 베이스라인 성능 및 매 에포크 훈련/테스트 메트릭
- **저장 파일**: `artifacts/ys_alloc_net.pt` - 최적 모델 가중치

## 실행 요구사항

- 전처리된 데이터 필요:
  - `data/processed/traffic_data_train.npz`
  - `data/processed/traffic_data_test.npz`
- GPU (CUDA) 또는 CPU에서 자동 실행

