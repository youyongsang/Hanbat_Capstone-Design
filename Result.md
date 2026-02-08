# Training Results Interpretation

**요약**
- **모델 성능 개선:** 학습된 `YSAllocNet`은 Rule-based 베이스라인보다 분포 예측 성능이 우수합니다.
- **저장된 모델:** [artifacts/ys_alloc_net.pt](artifacts/ys_alloc_net.pt)

**핵심 수치**
- **Rule-based (proportional_last_step)**: KL=0.052434, MSE=0.001622, MAE=0.016380, Jain=0.714869, MaxShare=0.185604
- **YSAllocNet (best, epoch 17)**: KL=0.028870, MSE=0.000831, MAE=0.011292, Jain=0.752192, MaxShare=0.161482

**훈련 경향 요약**
- 초기 몇 에포크에서 빠르게 손실이 감소하며 안정적으로 수렴했습니다 (특히 epoch 3~10 사이 큰 개선).
- 검증(KL)은 epoch 17에서 최저(0.028870)를 기록하여 해당 시점 모델을 저장했습니다.
- 최종적으로 훈련/검증 간 큰 괴리 없음 — 과적합 징후는 크지 않습니다.

**해석**
- **KL 감소 (0.052→0.0289)**: 모델이 노드별 할당 분포를 더 정확히 예측함을 의미합니다. 분포 차이가 절반 가까이 줄었습니다.
- **MSE/MAE 개선**: 포인트 단위 예측 정확도도 함께 향상되어 실제 트래픽 기반 할당과 근사도가 좋아졌습니다.
- **공정성(Jain) 증가**: Rule-based 대비 Jain 지표가 상승(0.714→0.752), 모델 예측이 더 균형 잡힌 할당을 생성합니다.
- **MaxShare 감소**: 최다 할당 비율이 감소하여 특정 노드 편중이 완화되었습니다.

**권장 다음 단계**
- `train/train_ys_alloc.py`로 하이퍼파라미터(학습률, 은닉크기, 드롭아웃, 에포크) 탐색 후 재학습해 보세요. ([train/train_ys_alloc.py](train/train_ys_alloc.py))
- 검증 곡선(Train/Test KL, MSE, MAE, Jain, MaxShare) 시각화로 수렴 및 안정성 확인을 권장합니다.
- 별도 Hold-out 또는 실전 로그로 추가 평가를 수행하세요(일반화 성능 검증).
- 실서비스용으로는 모델 추론 스크립트 및 입력/출력 포맷 검증을 추가하세요.

**참고**
- 저장된 모델은 [artifacts/ys_alloc_net.pt](artifacts/ys_alloc_net.pt) 입니다.
- 학습 스크립트: [train/train_ys_alloc.py](train/train_ys_alloc.py)
