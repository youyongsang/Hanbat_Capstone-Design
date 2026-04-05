# 🚀 Production-Ready LSTM Auto-Scaling System

이 프로젝트는 딥러닝(LSTM)을 활용하여 시계열 트래픽 데이터를 예측하고, 실제 운영 환경에서 가용성과 비용 효율성을 동시에 확보할 수 있는 지능형 자동 자원 할당(Auto-Scaling) 모델을 구현한 프로젝트입니다.

---

## 📌 1. 프로젝트 개요

본 모델은 단순한 트래픽 예측을 넘어, 실제 인프라 제어 시 발생하는 진동 현상(Flapping)을 방지하기 위해 **Hysteresis(이력 제어)**와 **Exponential Smoothing(지수 평활법)** 로직을 결합한 프로덕션급 엔진입니다.

---

## 🛠️ 2. 환경 설정 (Prerequisites)

## 가상 환경 생성 및 활성화
```bash
# 가상 환경 폴더 생성
python -m venv venv

# 가상 환경 활성화 (Windows 기준)
venv\Scripts\activate

# ※ 활성화 성공 시 터미널 앞에 (venv)라고 표시됩니다.

아래 명령어를 복사하여 필요한 라이브러리를 설치하세요.

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

---

## 📂 3. 실행 방법 (How to Run)

1. 파일 확인: 프로젝트 루트 폴더에 `allocate_resources.py` 파일이 있는지 확인합니다.
2. 명령어 실행: 터미널(CMD)에서 아래 명령어를 실행합니다.

```bash
python allocate_resources.py
```

3. 학습 진행: 모델이 50,000개의 대규모 합성 데이터를 바탕으로 20회(Epochs) 반복 학습을 수행합니다.
4. 결과 확인: 실행이 완료되면 `plots/` 폴더 내에 `real_lstm_production_final.png` 시각화 결과물이 생성됩니다.

---

## 📊 4. 결과 분석 가이드

최종 결과 그래프(`real_lstm_production_final.png`)는 다음 세 가지 선으로 구성됩니다.

* **Actual Traffic (회색 실선)**: 인프라에 유입되는 원본 트래픽 데이터
* **LSTM Predicted (파란 점선)**: AI 모델이 학습을 통해 도출한 예측값 (MAE 지표로 정확도 확인 가능)
* **Smart Allocation (빨간 실선)**: 예측치를 바탕으로 실제 할당된 서버 용량

특징:

* 예측값보다 항상 상단에 위치하여 안정성을 보장
* 트래픽 노이즈에 민감하게 반응하지 않고 부드럽게 자원을 조절

---

## 🎓 5. 가이드라인 준수 사항 (Technical Details)

* **Data Normalization**: MinMaxScaler를 사용하여 데이터를 [0, 1] 범위로 정규화

* **Deep Learning Architecture**

  * Input Layer: Window Size 12 (과거 12개 시점 참조)
  * Hidden Layer: LSTM (64 Units)
  * Dense Layer: 32 Units (ReLU Activation)
  * Output Layer: 1 Unit (Linear)

* **Evaluation Metrics**:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)

* **Production Logic**:

  * ceil(p / C) 수식을 기반으로 자원 할당
  * 컨테이너당 용량: 80 RPS
  * 최소 5대 ~ 최대 35대 운영 범위 준수
