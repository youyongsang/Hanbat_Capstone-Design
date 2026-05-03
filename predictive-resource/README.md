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
docker rm -f app_server_1 app_server_2 app_server_3 app_server_4 app_server_5
```

PowerShell:

```powershell
docker rm -f app_server_1 app_server_2 app_server_3 app_server_4 app_server_5
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

- `SAFETY_MARGIN = 1.60`
- `MIN_REPLICAS = 3`, `MAX_REPLICAS = 8`
- `MIN_CPU = 2.0`, `MAX_CPU = 6.0`

현재 정책은 `CPU 먼저, 필요 시 Replica 증가` 방식으로 동작합니다.

CPU 단계별 1개 Replica 기준 처리량:

- `2.0 CPU` : `70 RPS`
- `3.0 CPU` : `105 RPS`
- `4.0 CPU` : `140 RPS`
- `5.0 CPU` : `175 RPS`
- `6.0 CPU` : `210 RPS`

정책 흐름:

1. 예측값에 `SAFETY_MARGIN`을 곱해 안전 여유를 적용
2. `MIN_REPLICAS`부터 시작해서 가능한 가장 적은 Replica 수를 탐색
3. 그 Replica 수 안에서 감당 가능한 최소 CPU 단계를 선택
4. 모든 단계로도 부족하면 `MAX_CPU`, `MAX_REPLICAS`를 사용

### 5.3 Reactive 보정

Reactive 보정 레이어는 `AllocationState`에 저장된 계획 기준선을 바탕으로 동작합니다.

- 과부하: CPU 먼저 증가, 부족하면 replica 증가
- 여유 상태: 계획 수준으로 복귀
- 정상 상태: 유지

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
