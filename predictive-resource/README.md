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
│   ├── lstm_model.h5
│   └── scaler.pkl
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
python scripts/forecast_and_plan.py --observed-points 60
```

생성 파일:

- `data/output/predicted_traffic.csv`
- `data/output/resource_allocation_plan.csv`

### 4.3 하이브리드 시스템 실행

```bash
python scripts/run_hybrid.py --plan-csv data/output/resource_allocation_plan.csv
```

PowerShell:

```powershell
python .\scripts\run_hybrid.py --plan-csv .\data\output\resource_allocation_plan.csv
```

로드 생성기를 함께 실행하려면:

```bash
python scripts/run_hybrid.py --plan-csv data/output/resource_allocation_plan.csv --with-loadgen
```

PowerShell:

```powershell
python .\scripts\run_hybrid.py --plan-csv .\data\output\resource_allocation_plan.csv --with-loadgen
```

실제 로드용 CSV를 직접 지정하려면:

```bash
python scripts/run_hybrid.py --plan-csv data/output/resource_allocation_plan.csv --actual-csv data/input/sale_event_traffic.csv
```

---

## 5. 핵심 설계

### 5.1 예측과 실행 분리

예측 모델은 `model/` 폴더에서 트래픽 예측만 담당하고, 자원 할당은 `scripts/forecast_and_plan.py`가 담당합니다.  
즉 모델과 자원 정책을 분리한 구조입니다.

### 5.2 자원 정책

예측된 `predicted_rps`는 아래 규칙으로 자원 계획으로 변환됩니다.

- `SAFETY_MARGIN = 1.2`
- `CONTAINER_CAPACITY = 80`
- `MIN_REPLICAS = 1`, `MAX_REPLICAS = 5`
- `MIN_CPU = 0.5`, `MAX_CPU = 3.0`

CPU 정책:

- 컨테이너당 40 RPS 이하: `0.5 CPU`
- 컨테이너당 80 RPS 이하: `1.0 CPU`
- 컨테이너당 120 RPS 이하: `2.0 CPU`
- 그 이상: `3.0 CPU`

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
