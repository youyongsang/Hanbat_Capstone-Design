# predictive-resource

LSTM 예측 모델과 Reactive 보정 메커니즘을 결합한 **하이브리드 동적 자원 할당 시스템**.

---

## 1. 프로젝트 개요

기존 `reactive-resource`는 성능 저하가 *발생한 뒤* 자원을 늘리는 방식이라 딜레이와 과부하 구간이 불가피합니다.  
이 프로젝트는 두 가지 레이어를 조합해 그 문제를 해결합니다.

| 레이어 | 방식 | 역할 |
|--------|------|------|
| Layer A | 예측 (LSTM) | CSV 스케줄의 미래 트래픽을 5초 앞서 예측해 Docker 자원을 선제 할당 |
| Layer B | 반응형 보정 | 실시간 /metrics를 2초마다 확인, 예측과 실제의 편차를 발견하면 CPU·컨테이너 보정 |

---

## 2. 디렉토리 구조

```
predictive-resource/
├── app/
│   └── app.py                          # FastAPI 서버 (reactive-resource와 동일)
├── data/
│   ├── input/
│   │   └── sale_event_traffic.csv      # 트래픽 시나리오
│   └── output/
│       ├── predictive_allocation_plan.csv   # 예측 레이어 로그
│       └── hybrid_correction_log.csv        # 보정 레이어 로그
├── results/                            # 시각화 결과 PNG
├── scripts/
│   ├── predictive_allocator.py         # Layer A: LSTM 예측 + 선제 할당
│   ├── hybrid_controller.py            # Layer B: 실시간 보정
│   ├── run_hybrid.py                   # 진입점 (두 레이어 동시 구동)
│   ├── hybrid_load_generator.py        # 로드 생성기 (reactive-resource에서 복사)
│   └── plot_results.py                 # 결과 시각화
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 3. 사전 준비

### 3.1 가상환경 및 패키지

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow requests aiohttp
```

### 3.2 LSTM 모델 준비 (선택)

`Dynamic_Resource_Allocation_Model/lstm_model.h5`와 `scaler.pkl`이 있으면 LSTM 예측을 사용합니다.  
없으면 단순 이동평균(SMA)으로 자동 폴백합니다.

```bash
# 모델 학습이 필요한 경우
cd Dynamic_Resource_Allocation_Model
python preprocess_lstm.py sale_event_traffic.csv
python train_lstm.py
```

### 3.3 Docker 이미지 빌드

```bash
cd predictive-resource
docker build -t reactive-server .
```

---

## 4. 실행 방법

### 4.1 기존 컨테이너 정리

```bash
docker rm -f app_server_1 app_server_2 app_server_3 app_server_4 app_server_5
```

### 4.2 하이브리드 시스템 실행

```bash
# 터미널 1: 하이브리드 시스템
python scripts/run_hybrid.py

# 터미널 2: 로드 생성기 (별도 실행)
python scripts/hybrid_load_generator.py

# 또는 한 번에
python scripts/run_hybrid.py --with-loadgen
```

### 4.3 CSV 경로 직접 지정

```bash
python scripts/run_hybrid.py --csv data/input/sale_event_traffic.csv
```

---

## 5. 핵심 설계

### LOOKAHEAD_SEC = 5
컨테이너 기동에 약 2~3초 걸리므로, 현재 시각 기준 5초 뒤의 트래픽을 미리 예측해 자원을 준비합니다.

### AllocationState 공유 객체
예측 레이어가 설정한 `(cpu, replicas)`를 보정 레이어의 **복귀 기준선**으로 사용합니다.  
보정 레이어는 성능이 나빠지면 올리고, 여유가 생기면 예측 수준까지만 내립니다 (최솟값까지 내리지 않음).

### 보정 우선순위
1. 과부하 → CPU 먼저 증가 → CPU 한계 + 심각 과부하면 컨테이너 추가
2. 여유 → 예측 수준(pred_cpu / pred_replicas)으로 복귀
3. 정상 → HOLD

---

## 6. 출력 파일

| 파일 | 내용 |
|------|------|
| `data/output/predictive_allocation_plan.csv` | 매 초 예측 RPS, 목표 CPU, 목표 replica, 적용 여부 |
| `data/output/hybrid_correction_log.csv` | 보정 시각, 예측 vs 실제 상태, 보정 액션, latency, CPU 사용률 |
| `results/*.png` | 시각화 그래프 |

---

## 7. reactive-resource와의 비교

| 항목 | reactive-resource | predictive-resource |
|------|-------------------|---------------------|
| 자원 할당 시점 | 성능 저하 후 | 트래픽 증가 5초 전 |
| 딜레이 구간 | 발생 | 최소화 |
| 과부하 보호 | 반응형 | 예측 + 반응형 보정 |
| LSTM 모델 사용 | X | O (없으면 SMA 폴백) |
