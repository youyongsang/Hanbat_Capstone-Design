# Predictive Resource Allocation Project Guide

이 문서는 **Sales Event 트래픽 상황**을 가정하여,  
**LSTM 예측 모델**과 **Smart Allocation 자원 할당 정책**이 유기적으로 상호작용하는 구조를 설명합니다.

---

## 1. 프로젝트 파일 구조 및 역할

| 분류 | 파일명 | 역할 및 상세 설명 |
|------|--------|------------------|
| Data | `sales_event_traffic.csv` | 원본 데이터: 세일 이벤트 시 발생하는 시계열 트래픽 데이터 (RPS) |
| Pre-process | `preprocess_lstm.py` | CSV 데이터를 학습용(`.npy`)과 정규화 파일(`.pkl`)로 변환 |
| Training | `train_lstm.py` | LSTM 모델 학습 및 저장 |
| Strategy | `resource_policy.py` | 예측된 RPS 기반 CPU 및 Replica 계산 로직 |
| Execution | `predict_and_allocate.py` | 모델 로드 후 실시간 예측 및 자원 할당 |
| Evaluation | `evaluate_predictive.py` | MAE 등 예측 성능 평가 |
| Main | `allocate_resources.py` | 전체 실행 및 결과 그래프 생성 |

---

## 2. 코드 간 상호작용 흐름 (Workflow)

### Step 1. 데이터 정제 (Data Preparation)

- **Input**
  - `sales_event_traffic.csv`

- **Process**
  - `preprocess_lstm.py` 실행
  - 데이터 0~1 정규화
  - 12초 단위 Window 생성

- **Output**
  - `scaler.pkl`
  - `X_train.npy`
  - `y_train.npy`

---

### Step 2. 모델 학습 (Model Training)

- **Input**
  - `X_train.npy`, `y_train.npy`

- **Process**
  - `train_lstm.py`에서 LSTM 학습 수행

- **Output**
  - `lstm_model.h5`

---

### Step 3. 예측 및 정책 적용 (Prediction & Policy)

- **Input**
  - `lstm_model.h5`
  - 최신 12초 트래픽 데이터

- **Process**
  - `predict_and_allocate.py` → 미래 트래픽(RPS) 예측
  - 예측값을 `resource_policy.py`에 전달
  - 자원 할당 정책 적용

- **Output**
  - CPU 할당량
  - Replica 개수

---

### Step 4. 최종 실행 및 결과 (Action & Result)

- **Process**
  - `allocate_resources.py` 실행

- **Output**
  - `result/` 폴더에 그래프 생성
  - 자원 할당 안정성 확인

---

## 3. 핵심 상호작용: Smart Allocation

### 핵심 구성
- `predict_and_allocate.py`
- `resource_policy.py`

---

### 동작 구조

- **LSTM (예측)**
  - 과거 패턴 기반 미래 트래픽 예측

- **Smart Allocation (대응)**
  - 예측 결과 기반 자원 할당 수행

---

결과적으로 시스템 가용성과 비용 효율성을 동시에 확보
