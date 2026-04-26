# 🚀 AI 기반 지능형 서버 자원 할당 시스템 (LSTM RPS Predictor)

본 프로젝트는 딥러닝(Bidirectional LSTM)을 활용하여 웹 서비스의 트래픽(RPS)을 예측하고, 이를 기반으로 서버 자원(Container Replicas)을 선제적으로 자동 할당하는 지능형 오토스케일링 엔진을 구현한 프로젝트입니다.

---

## 📌 1. 프로젝트 개요

단순한 수치 예측을 넘어, 실제 운영 환경에서 치명적인 **트래픽 피크(Peak) 상황**에 유연하게 대응할 수 있도록 설계되었습니다.
특히 **Peak-Weighted Loss(피크 가중 손실 함수)**를 도입하여, 트래픽 급증 시의 예측 정확도를 높이고 서비스 가용성을 극대화한 것이 특징입니다.

---

## 🛠️ 2. 환경 설정 (Prerequisites)

### 가상 환경 생성 및 활성화

```bash
# 가상 환경 폴더 생성
python -m venv venv

# 가상 환경 활성화 (Windows 기준)
venv\Scripts\activate
```

> ✅ 활성화 성공 시 터미널 앞에 `(venv)`가 표시됩니다.

### 필수 라이브러리 설치

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

---

## 📂 3. 실행 방법 (How to Run)

### 1) 파일 구성 확인

프로젝트 루트 폴더에 아래 파일들이 포함되어 있는지 확인합니다:

* `allocate_resources.py`
* `schema_adapter.py`
* `robust_scaler.py`
* 기타 관련 모듈

---

### 2) 실행 명령어

```bash
# 기본 데이터셋 (week_traffic.csv) 실행
python allocate_resources.py

# 특정 트래픽 데이터셋으로 실행
python allocate_resources.py your_data.csv
```

---

### 3) 자동화 파이프라인

실행 시 다음 과정이 자동으로 수행됩니다:

```
데이터 스키마 변환
→ OOD 탐지 및 스케일링
→ LSTM 학습 (Max 150 Epochs)
→ 자원 할당 시뮬레이션
```

---

### 4) 결과 확인

실행 완료 후:

* `plots/` 폴더에 시각화 리포트 생성

---

## 📊 4. 결과 분석 가이드

그래프 구성 요소:

* **Actual Traffic (회색 실선)**: 실제 트래픽
* **LSTM Predicted (파란 점선)**: 모델 예측값
* **Resource Allocation (빨간 실선)**: 할당된 서버 자원

---

### 🔍 핵심 특징

* **Peak Sensitivity**
  트래픽 급증 구간에서 자원이 선제적으로 증가하여 장애를 방지

* **OOD Alert**
  학습 범위를 벗어난 데이터 감지 시 경고 발생

---

## 🎓 5. 기술 상세 (Technical Details)

### 📌 Data Normalization

* `RobustScaler` 사용 (Outlier 영향 최소화)
* OOD(Out-of-Distribution) 탐지 로직 내장

---

### 📌 Deep Learning Architecture

* **Model**: Bidirectional LSTM
* **Input Window**: 60
* **구조**: 2-Layer Bi-LSTM + Dropout
* **Optimizer**: Adam (LR = 0.001)

---

### 📌 Advanced Loss Function

* **Peak-Weighted Loss**

  * 상위 35% 피크 데이터 → 가중치 3.0 적용
* 일반 MSE 대비 피크 추종 성능 향상

---

### 📌 Resource Policy

* 기본 정책: **100 RPS → 1 Unit**
* 운영 범위:

  * 최소: 2 Units
  * 최대: 20 Units

---

### 📌 Training Strategy

* **EarlyStopping**

  * 기준: Validation Loss
  * Patience: 10
  * 최적 모델 자동 저장

---

## ✅ Summary

이 시스템은 단순 예측 모델을 넘어,
**실제 운영 환경에서 안정성과 비용 효율을 동시에 만족시키는 AI 기반 오토스케일링 솔루션**입니다.
