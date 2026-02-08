# Preprocessing Module

트래픽 데이터 전처리 및 데이터셋 생성을 담당하는 모듈입니다.

## 파일 설명

### generator.py
합성 트래픽 데이터를 생성합니다.

**기능:**
- 지정된 개수의 노드에서 시계열 트래픽 데이터 생성
- 기본 패턴(사인함수), 노드별 가중치, 랜덤 노이즈, 버스트 트래픽 추가
- CSV 형식으로 raw 데이터 저장

**실행:**
```bash
python generator.py
```

**출력:**
- `data/raw/traffic_log.csv` - 원본 트래픽 데이터

### preprocessor.py
CSV 데이터를 모델 학습용 numpy 배열로 변환합니다.

**기능:**
- CSV를 읽어 node별로 pivot
- 로그 변환 (log1p)을 통한 정규화
- 시계열 윈도우 슬라이딩으로 (X, Y) 쌍 생성
- 훈련/테스트 데이터 분할
- NPZ 형식으로 저장

**데이터 형태:**
- Input (X): (S, N, T, F) - Sample, Node, Time, Feature
- Output (Y): (S, N) - Sample, Node별 예측값

**실행:**
```bash
python preprocessor.py
```

**출력:**
- `data/processed/traffic_data_train.npz` - 훈련 데이터
- `data/processed/traffic_data_test.npz` - 테스트 데이터

## 실행 순서

1. `generator.py` 실행 → raw 데이터 생성
2. `preprocessor.py` 실행 → 처리된 dataset 생성

## 설정

`config.py`에서 다음 파라미터 조정:
- `TOTAL_SAMPLES`: 생성할 샘플 수
- `NUM_NODES`: 노드 개수
- `WINDOW_SIZE`: 시계열 윈도우 크기
- `PRED_HORIZON`: 예측 범위
- `TRAIN_RATIO`: 훈련/테스트 비율

