# predictive-resource

LSTM 예측 모델과 Reactive 보정 메커니즘을 결합한 하이브리드 동적 자원 할당 시스템입니다.

---

## 1. 프로젝트 개요

이 디렉토리는 아래 4단계 흐름으로 동작합니다.

1. 과거 CSV로 LSTM 학습
2. 학습된 모델로 미래 트래픽을 순차 예측
3. 예측 결과를 규칙 기반 자원 정책에 넣어 계획 CSV 생성
4. 실행 시에는 계획 CSV를 따라 선제 할당하고, Reactive 보정 레이어가 실시간으로 오차를 조정

즉 현재 구조는 `예측 -> 계획 생성 -> 계획 실행 -> 실시간 보정`으로 분리되어 있습니다.

---

## 2. 디렉토리 구조

```text
predictive-resource/
├── app/
│   └── app.py
├── data/
│   ├── input/
│   │   └── sale_event_traffic.csv
│   └── output/
│       ├── predicted_traffic.csv
│       ├── resource_allocation_plan.csv
│       ├── predictive_allocation_log.csv
│       ├── hybrid_correction_log.csv
│       └── loadgen_result.csv
├── model/
│   ├── preprocess_lstm.py
│   ├── train_lstm.py
│   ├── evaluate_full.py
│   ├── lstm_model.h5
│   ├── scaler_x.pkl
│   ├── scaler_y.pkl
│   ├── feature_cols.pkl
│   └── metadata.pkl
├── scripts/
│   ├── forecast_and_plan.py
│   ├── predictive_allocator.py
│   ├── hybrid_controller.py
│   ├── hybrid_load_generator.py
│   ├── plot_hybrid_results.py
│   └── run_hybrid.py
├── Dockerfile
├── requirements.txt
└── README.md
```

파일 역할:

- `forecast_and_plan.py`: 미래 트래픽 예측 CSV와 자원 계획 CSV 생성
- `predictive_allocator.py`: 계획 CSV를 읽어 시간축에 맞춰 선제 할당
- `hybrid_controller.py`: 실시간 metrics 기반 보정
- `run_hybrid.py`: 계획 실행과 보정 루프를 함께 구동

---

## 3. 사전 준비

### 3.1 패키지 설치

```bash
pip install -r requirements.txt
```

PowerShell:

```powershell
python -m pip install -r .\requirements.txt
```

### 3.2 LSTM 모델 학습

```bash
cd predictive-resource
python model/preprocess_lstm.py data/input/sale_event_traffic.csv
python model/train_lstm.py
python model/evaluate_full.py data/input/sale_event_traffic.csv
```

PowerShell:

```powershell
Set-Location .\predictive-resource
python .\model\preprocess_lstm.py .\data\input\sale_event_traffic.csv
python .\model\train_lstm.py
python .\model\evaluate_full.py .\data\input\sale_event_traffic.csv
```

현재 `forecast_and_plan.py`는 학습된 `lstm_model.h5`, `scaler_x.pkl`, `scaler_y.pkl`, `feature_cols.pkl`, `metadata.pkl`이 있어야 동작합니다.

### 3.3 Docker 이미지 빌드

```bash
docker build -t reactive-server .
```

PowerShell:

```powershell
docker build -t reactive-server .
```

---

## 4. 실행 방법

### 4.1 기존 컨테이너 정리

```bash
docker rm -f app_server_1 app_server_2 app_server_3 app_server_4 app_server_5 app_server_6 app_server_7 app_server_8
```

PowerShell:

```powershell
docker rm -f app_server_1 app_server_2 app_server_3 app_server_4 app_server_5 app_server_6 app_server_7 app_server_8
```

### 4.2 미래 트래픽 및 자원 계획 생성

```bash
python scripts/forecast_and_plan.py --input-csv data/input/sale_event_traffic.csv
```

PowerShell:

```powershell
python .\scripts\forecast_and_plan.py --input-csv .\data\input\sale_event_traffic.csv
```

선택 옵션:

```bash
python scripts/forecast_and_plan.py --input-csv data/input/sale_event_traffic.csv --observed-points 180
```

`--observed-points`의 기본값은 `180`이며, 이 구간까지는 실제 관측 데이터를 사용하고 이후 구간은 모델이 순차 예측합니다.

현재 구조는 이렇게 생성된 `predicted_traffic.csv`를 바탕으로 전체 `resource_allocation_plan.csv`를 먼저 수립한 뒤, 실행 단계에서 Reactive 보정 레이어가 오차를 보완하는 방식입니다.  
따라서 트래픽 패턴이 특정 날짜나 이벤트 주기에 따라 반복적인 양상을 보일수록 예측 정확도가 높아질 수 있으며, 이에 따라 예측 기반 자원 계획의 신뢰성도 함께 향상될 수 있습니다.

현재 계획 생성 로직은 단순히 `현재 시점 predicted_rps`만 사용하지 않고, `lookahead` 구간의 최대 예측값을 함께 참고하여 피크 구간을 조금 더 일찍 준비하도록 설계되어 있습니다.
실행 단계에서는 이 계획을 그대로 현재 시각에 적용하지 않고, `PredictiveAllocator`가 약 `15초` 앞당겨 적용하여 피크 시작 직전 미리 자원을 확보하도록 동작합니다.

현재 최종 기준선은 `3단계 안정화 튜닝` 버전이며, 핵심 방향은 다음과 같습니다.

- 계획은 `lookahead` 기반으로 생성하고 `15초` 앞당겨 적용
- 피크 구간은 replica floor로 선제 준비
- 실행 중 보정은 CPU-first를 기본으로 하되, 높은 latency와 CPU 압박이 동시에 나타날 때만 replica 추가
- scale-out은 비교적 빠르게, scale-in은 더 느리고 보수적으로 수행
- PredictiveAllocator와 HybridController는 같은 Docker 자원을 제어하므로, lock과 hold를 사용해 충돌과 scale churn을 줄임

생성 파일:

- `data/output/predicted_traffic.csv`
- `data/output/resource_allocation_plan.csv`
- `results/resource_allocation_plan_overview.png`

### 4.3 하이브리드 시스템 실행

실제 성능 실험을 수행하려면 로드 생성기를 함께 실행하는 방식이 기본입니다.

```bash
python scripts/run_hybrid.py --plan-csv data/output/resource_allocation_plan.csv --with-loadgen
```

PowerShell:

```powershell
python .\scripts\run_hybrid.py --plan-csv .\data\output\resource_allocation_plan.csv --with-loadgen
```

이 명령은 아래 작업을 한 번에 수행합니다.

- 예측 기반 자원 계획 적용
- FastAPI 서버 및 Docker 자원 실행
- Load Generator를 통한 실제 트래픽 발생
- Reactive 보정 수행
- 결과 로그 및 성능 분석용 CSV 생성

실험 시 Load Generator 기준은 다음과 같습니다.

| 항목 | 값 |
|---|---:|
| `SLA_LATENCY_MS` | `1000` |
| `REQUEST_TIMEOUT` | `5` |

로드 생성기 없이 자원 계획 및 제어 로직만 확인하려면 아래처럼 실행할 수 있습니다.

```bash
python scripts/run_hybrid.py --plan-csv data/output/resource_allocation_plan.csv
```

PowerShell:

```powershell
python .\scripts\run_hybrid.py --plan-csv .\data\output\resource_allocation_plan.csv
```

실제 로드용 CSV를 직접 지정하려면:

```bash
python scripts/run_hybrid.py --plan-csv data/output/resource_allocation_plan.csv --actual-csv data/input/sale_event_traffic.csv
```

### 4.4 결과 분석 그래프 생성

하이브리드 실행이 끝난 뒤 결과 CSV를 기반으로 분석 그래프를 다시 생성하려면 아래 명령을 실행합니다.

```bash
python scripts/plot_hybrid_results.py
```

PowerShell:

```powershell
python .\scripts\plot_hybrid_results.py
```

생성 결과:

- `results/predicted_vs_actual_rps.png`
- 실제 트래픽(`actual_rps`)과 예측 트래픽(`predicted_rps`)의 시간축 비교 그래프
- `results/prediction_error.png`
- 예측 오차 분포와 시간대별 오차 크기를 확인하는 그래프
- `results/resource_plan.png`
- 예측된 트래픽을 기준으로 생성된 CPU 및 Replica 계획 분포 그래프
- `results/plan_vs_actual_resource.png`
- 계획 자원과 실제 실행 중 적용된 CPU/Replica 변화를 비교하는 그래프
- `results/correction_actions.png`
- Reactive 보정 액션(`HOLD`, `CPU_UP` 등)과 상태 변화 추이를 보여주는 그래프
- `results/loadgen_performance.png`
- 목표 요청량, 성공/실패 수, SLA 위반, 평균/p95 응답시간을 함께 보여주는 성능 분석 그래프

---

## 5. 핵심 설계

### 5.1 예측과 실행 분리

예측 모델은 `model/` 폴더에서 트래픽 예측만 담당하고, 자원 할당은 `scripts/forecast_and_plan.py`가 담당합니다.  
즉 모델과 자원 정책을 분리한 구조입니다.

### 5.2 자원 정책

예측된 `predicted_rps`는 아래 규칙으로 자원 계획으로 변환됩니다.

| 항목 | 값 |
|---|---:|
| `SAFETY_MARGIN` | `1.60` |
| `MIN_REPLICAS` | `3` |
| `MAX_REPLICAS` | `7` |
| `MIN_CPU` | `2.0` |
| `MAX_CPU` | `6.0` |

현재 정책은 **기본적으로 CPU-first**로 동작하며, `lookahead` 구간 안에서 큰 피크가 곧 시작될 것으로 예측되는 경우에만 **selective replica-first**를 적용합니다.

또한 피크 구간에서는 `lookahead_peak` 수준에 따라 최소 replica floor를 적용하여, CPU만 높이는 대신 피크 시작 전 분산 여유를 더 적극적으로 확보합니다. 현재 기준선은 최대 replica 수를 `7`로 제한하여 마지막 단계 확장에서 반복되던 오류 가능성을 줄이고 있습니다.

`lookahead_peak` 기준 최소 replica floor:

| `lookahead_peak` | 최소 replica |
|---|---:|
| `>= 570` | `7` |
| `>= 520` | `6` |
| `>= 460` | `5` |
| `>= 400` | `4` |

CPU 단계별 1개 Replica 기준 처리량:

- `2.0 CPU` : `55 RPS`
- `3.0 CPU` : `80 RPS`
- `4.0 CPU` : `105 RPS`
- `5.0 CPU` : `130 RPS`
- `6.0 CPU` : `155 RPS`

정책 흐름:

1. 예측값에 `SAFETY_MARGIN`을 곱해 안전 여유를 적용
2. 현재 시점 예측값이 아니라 `lookahead` 구간의 최대 예측값을 기준으로 선제 계획 수립
3. 기본적으로 `MIN_REPLICAS`를 유지한 상태에서 감당 가능한 최소 CPU 단계를 먼저 선택
4. 다만 큰 피크가 임박한 구간은 `lookahead_peak`에 따른 최소 replica floor를 적용하여, replica를 1~2단계 이상 선제 증가시키는 selective replica-first 적용
5. CPU 최대치에서도 부족하면 그때만 replica를 추가 증가
6. 작은 CPU 변화는 무시하는 hysteresis를 적용해 `5.0 ↔ 6.0` 같은 흔들림을 줄임

### 5.3 Reactive 보정

Reactive 보정 레이어는 `AllocationState`에 저장된 계획 기준선을 바탕으로 동작하며, 단일 컨테이너가 아니라 현재 활성화된 전체 컨테이너의 `/metrics`를 집계한 결과를 기준으로 emergency correction을 수행합니다.
또한 내부 `/metrics`만으로는 실제 사용자 체감 지연을 놓칠 수 있기 때문에, 최근 `loadgen_result.csv`의 avg/p95 latency, fail 수, SLA 위반 수를 함께 참고하여 replica 증가 여부를 판단합니다.

핵심 보정 파라미터:

| 구분 | 항목 | 값 |
|---|---|---:|
| 과부하 기준 | `LATENCY_WARN_MS` | `500.0` |
| 과부하 기준 | `P95_WARN_MS` | `750.0` |
| 과부하 기준 | `CPU_USAGE_WARN` | `60.0` |
| replica 추가 기준 | `LAT_REPLICA_UP_MS` | `600.0` |
| replica 추가 기준 | `P95_REPLICA_UP_MS` | `850.0` |
| replica 추가 기준 | `CPU_CONTAINER_OUT_THRESHOLD` | `70.0` |
| replica 추가 기준 | `MAX_CPU_NEAR_THRESHOLD` | `5.0` |
| 여유 기준 | `LATENCY_EASY_MS` | `250.0` |
| 여유 기준 | `P95_EASY_MS` | `500.0` |
| 여유 기준 | `CPU_USAGE_EASY` | `30.0` |

운영/안정화 파라미터:

| 항목 | 값 |
|---|---:|
| `CHECK_INTERVAL` | `2` |
| `CPU_SCALE_OUT_COOLDOWN` | `5` |
| `REPLICA_SCALE_OUT_COOLDOWN` | `10` |
| `SCALE_IN_COOLDOWN` | `30` |
| `OVERRIDE_HOLD_SEC` | `30` |
| `SCALE_DOWN_HOLD_SEC` | `60` |
| `CPU_SCALE_IN_CONFIRMATIONS` | `3` |
| `REPLICA_SCALE_IN_CONFIRMATIONS` | `5` |
| `PLAN_ADVANCE_SEC` | `15` |
| `REPLICA_SCALE_DOWN_CONFIRMATIONS` | `3` |

동작 방식은 다음과 같습니다.

- 과부하: 평균 latency `500ms`, p95 latency `750ms`, CPU 사용률 `60%` 기준을 넘기면 CPU를 우선 증가
- 평균 CPU뿐 아니라 **활성 컨테이너 중 최대 CPU 사용률**도 함께 확인하여, 특정 컨테이너에 병목이 몰리는 경우를 더 빨리 감지
- CPU가 `5.0` 이상으로 높아진 상태에서 활성 컨테이너 중 최대 CPU 사용률이 높고, 평균 latency `600ms` 또는 p95 latency `850ms` 이상으로 악화되면 replica를 추가 증가
- 여유 상태: 여러 번 연속 여유 상태가 확인될 때만 천천히 scale-in
- 정상 상태: 유지

또한 Reactive가 CPU나 replica를 증가시킨 직후에는 `override hold` 기간을 두어, 다음 Predictive 계획 tick이 방금 올린 자원을 바로 낮추지 않도록 보호합니다.
추가로 PredictiveAllocator와 HybridController는 동일한 Docker 자원을 제어하므로, 내부적으로 동기화 lock과 `60초` scale-down hold를 사용하여 동시 제어 충돌과 과도한 scale churn을 줄이도록 구성되어 있습니다. replica scale-out은 CPU scale-out보다 더 긴 cooldown을 사용하며, replica scale-down은 더 많은 연속 확인과 더 긴 cooldown 이후에만 반영되도록 하여 흔들림을 줄였습니다.

즉 Reactive baseline과 Predictive-Hybrid는 동일한 보정 기준 위에서 비교되며, 두 방식의 핵심 차이는 **보정 임계값**이 아니라 **피크 전에 얼마나 빨리 준비하느냐**에 있습니다.

---

## 6. 출력 파일

| 파일 | 내용 |
|------|------|
| `data/output/predicted_traffic.csv` | 시간축별 예측 트래픽 |
| `data/output/resource_allocation_plan.csv` | 예측 트래픽을 자원 계획으로 변환한 결과 |
| `data/output/predictive_allocation_log.csv` | 계획 실행 로그 |
| `data/output/hybrid_correction_log.csv` | 실시간 보정 로그 |
| `data/output/loadgen_result.csv` | 실제 부하 발생 결과 |
| `results/*.png` | 시각화 결과 |

---

## 7. reactive-resource와의 비교

| 항목 | reactive-resource | predictive-resource |
|------|-------------------|---------------------|
| 자원 할당 시점 | 성능 저하 후 | 계획 기반 선제 할당 후 보정 |
| 과부하 보호 | 반응형 | 예측 + 반응형 보정 |
| 계획 CSV 생성 | X | O |
| LSTM 모델 사용 | X | O |
