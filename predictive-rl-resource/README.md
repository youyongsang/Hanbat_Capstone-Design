# predictive-rl-resource

LSTM 기반 트래픽 예측 결과를 입력으로 사용하고, 강화학습(RL)으로 CPU / Replica 정책을 학습하는 실험 디렉토리입니다.

이 디렉토리는 **`predictive-resource`를 대체하는 완성형 실행 시스템**이라기보다,
우선 다음 목표를 달성하기 위한 **정책 학습용 확장판**입니다.

1. 규칙 기반 계획 정책을 RL 정책으로 대체/보완
2. `Reactive보다 무조건 품질 우위`가 아니라
   `비슷한 품질을 유지하면서 자원을 덜 쓰는 정책` 탐색
3. 이후 실제 Docker 실행 환경에 연결 가능한 형태의 RL 정책기 골격 확보

---

## 1. 현재 구조

```text
과거 CSV
→ LSTM 예측 (predictive-resource)
→ predicted_traffic.csv
→ RL 환경(state/action/reward)
→ baseline imitation warm-start
→ PPO 학습
→ 학습된 정책으로 RL 계획 CSV 생성
→ 나중에 실제 Docker 실행 환경에 연결 가능
```

즉 현재 구조에서:

- **LSTM**: 미래 RPS 예측 담당
- **RL**: CPU / Replica 행동 결정 담당

RL은 LSTM을 대체하는 것이 아니라, **예측 결과를 자원 정책으로 바꾸는 계층**입니다.
현재 학습은 가능한 경우 기존 rule-based Predictive baseline을 teacher로 삼아
**모방 학습(behavior cloning warm-start)** 을 먼저 수행한 뒤 PPO로 미세 조정합니다.

---

## 2. predictive-resource에서 가져와야 하는 것

이 디렉토리는 이제 독립적으로 실험할 수 있도록, `predictive-resource`의 핵심 CSV를
`predictive-rl-resource/data/` 아래로 복사해 두는 구조를 사용합니다.

### 2.1 필수 입력 파일

아래 파일은 `data/input/`에 복사해 두고 사용합니다.

| 원본 경로 | RL 디렉토리 내 복사 위치 | 용도 |
|---|---|---|
| `predictive-resource/data/input/sale_event_traffic.csv` | `data/input/sale_event_traffic.csv` | 실제 트래픽 시나리오 |
| `predictive-resource/data/output/predicted_traffic.csv` | `data/input/predicted_traffic.csv` | LSTM 예측 결과 |

### 2.2 권장 참고 파일

아래 파일은 `data/reference/`에 복사해 두면 규칙 기반 baseline 비교와
시뮬레이터 보정에 활용하기 좋습니다.

| 원본 경로 | RL 디렉토리 내 복사 위치 | 용도 |
|---|---|---|
| `predictive-resource/data/output/resource_allocation_plan.csv` | `data/reference/resource_allocation_plan.csv` | 기존 규칙 기반 plan 비교 |
| `predictive-resource/data/output/loadgen_result.csv` | `data/reference/loadgen_result.csv` | 기존 Predictive 실행 결과 비교 |
| `predictive-resource/data/output/hybrid_correction_log.csv` | `data/reference/hybrid_correction_log.csv` | 보정 패턴 참고 |

### 2.3 나중에 실제 예측을 다시 만들고 싶을 때

추후 예측 결과를 새로 만들고 싶다면 아래 model 자산도 참고할 수 있습니다.

| 경로 | 용도 |
|---|---|
| `predictive-resource/model/lstm_model.h5` | LSTM 모델 |
| `predictive-resource/model/scaler_x.pkl` | 입력 스케일러 |
| `predictive-resource/model/scaler_y.pkl` | 출력 스케일러 |
| `predictive-resource/model/feature_cols.pkl` | 입력 피처 구성 |
| `predictive-resource/model/metadata.pkl` | window size 등 메타데이터 |

---

## 3. 디렉토리 구조

```text
predictive-rl-resource/
├── README.md
├── requirements.txt
├── data/
│   ├── input/
│   │   ├── sale_event_traffic.csv
│   │   └── predicted_traffic.csv
│   ├── output/
│   │   ├── rl_eval_summary.json
│   │   └── rl_resource_allocation_plan.csv
│   └── reference/
│       ├── resource_allocation_plan.csv
│       ├── loadgen_result.csv
│       └── hybrid_correction_log.csv
├── artifacts/
│   └── ppo_resource_agent.zip
├── results/
│   ├── rl_training_curve.png
│   ├── rl_eval_plots.png
│   └── rl_analysis_report.txt
└── scripts/
    ├── config.py
    ├── data_utils.py
    ├── simulator.py
    ├── resource_env.py
    ├── train_rl.py
    ├── evaluate_rl.py
    ├── export_rl_plan.py
    ├── run_rl_hybrid.py
    └── plot_rl_results.py
```

---

## 4. 상태 / 행동 / 보상 설계

### 4.1 상태(state)

RL 에이전트는 아래 상태를 입력받습니다.

- 현재 실제 RPS
- 예측 RPS
- lookahead peak RPS
- 현재 CPU 사용률
- 현재 avg latency
- 현재 p95 latency
- 현재 fail count
- 현재 SLA violation count
- 현재 CPU 할당량
- 현재 replica 수
- 예측/실제 부하 차이
- RPS 증가 추세
- latency / p95 변화 추세

### 4.2 행동(action)

이산 행동 6개이며, **절대 자원값을 직접 올리고 내리는 것이 아니라 baseline 계획 대비 offset**을 선택합니다.

- `0 = baseline 그대로 유지`
- `1 = baseline CPU +0.5`
- `2 = baseline CPU -0.5`
- `3 = baseline replica +1`
- `4 = baseline replica -1`
- `5 = baseline CPU +0.5, replica +1`

### 4.3 보상(reward)

기본 방향:

- SLA 위반은 큰 패널티
- 실패 요청은 더 큰 패널티
- p95 / avg latency도 패널티
- CPU / replica 사용량은 비용 패널티
- 피크 구간과 비피크 구간의 자원 비용을 다르게 적용
- 비피크 구간에서 과도한 고자원 유지와 불필요한 scale-up에는 추가 패널티
- 자원을 줄였는데도 품질이 유지되면 scale-down 보너스
- 기존 rule-based baseline plan에 너무 멀어지면 추가 패널티
- baseline plan 근처에서 안정적으로 동작하면 alignment 보너스
- 피크 구간에서 안정적으로 SLA를 지키면 보너스

즉 **품질 + 효율**을 동시에 고려합니다.

---

## 5. 설치

```bash
pip install -r predictive-rl-resource/requirements.txt
```

PowerShell:

```powershell
python -m pip install -r .\predictive-rl-resource\requirements.txt
```

---

## 6. 전체 실행 순서

현재 RL 디렉토리의 전체 실행 흐름은 아래 순서를 따릅니다.

### 6.1 입력 데이터 준비

먼저 아래 입력 파일이 준비되어 있어야 합니다.

- `data/input/sale_event_traffic.csv`
- `data/input/predicted_traffic.csv`

필요 시 비교용 reference 파일도 함께 둡니다.

- `data/reference/resource_allocation_plan.csv`
- `data/reference/loadgen_result.csv`
- `data/reference/hybrid_correction_log.csv`

특히 `resource_allocation_plan.csv`와 `loadgen_result.csv`는 teacher imitation warm-start에 직접 사용됩니다.

### 6.2 RL 학습

기본 학습 순서:

1. reference plan / loadgen 결과에서 teacher dataset 생성
2. sparse한 teacher action 분포를 고려한 weighted imitation warm-start 수행
3. correction log를 teacher signal에 함께 반영해 `REP_UP` 계열 행동을 더 직접적으로 학습
4. curriculum(`easy → peak → full`) 기반 PPO fine-tuning

기본 명령:

```powershell
python .\predictive-rl-resource\scripts\train_rl.py
```

imitation warm-start 없이 PPO만 바로 학습하려면:

```powershell
python .\predictive-rl-resource\scripts\train_rl.py --no-imitation
```

생성 결과:

- `artifacts/ppo_resource_agent.zip`

### 6.3 RL 정책 평가

```powershell
python .\predictive-rl-resource\scripts\evaluate_rl.py
```

생성 결과:

- `data/output/rl_eval_summary.json`

### 6.4 RL 기반 계획 CSV 생성

```powershell
python .\predictive-rl-resource\scripts\export_rl_plan.py
```

생성 결과:

- `data/output/rl_resource_allocation_plan.csv`

### 6.5 실제 Docker 환경 검증

```powershell
python .\predictive-rl-resource\scripts\run_rl_hybrid.py
```

이 단계에서는:

1. RL 계획 CSV를 읽음
2. 기존 `predictive-resource/scripts/run_hybrid.py`를 호출
3. 실제 Docker 컨테이너와 load generator로 검증
4. 실행 로그를 RL 디렉토리로 복사

생성/복사 결과:

- `data/output/rl_predictive_allocation_log.csv`
- `data/output/rl_hybrid_correction_log.csv`
- `data/output/rl_loadgen_result.csv`

### 6.6 RL 전용 결과 그래프 생성

```powershell
python .\predictive-rl-resource\scripts\plot_rl_results.py
```

생성 결과:

- `results/rl_predicted_vs_actual_rps.png`
- `results/rl_resource_plan.png`
- `results/rl_plan_vs_actual_replicas.png`
- `results/rl_loadgen_performance.png`
- `results/rl_correction_actions.png`

### 6.7 한 번에 보면

```powershell
python -m pip install -r .\predictive-rl-resource\requirements.txt
python .\predictive-rl-resource\scripts\train_rl.py
python .\predictive-rl-resource\scripts\evaluate_rl.py
python .\predictive-rl-resource\scripts\export_rl_plan.py
python .\predictive-rl-resource\scripts\run_rl_hybrid.py
python .\predictive-rl-resource\scripts\plot_rl_results.py
```

---

## 7. 산출물 요약

전체 실행 순서에서 생성되는 핵심 산출물은 아래와 같습니다.

| 단계 | 주요 산출물 | 설명 |
|---|---|---|
| 학습 | `artifacts/ppo_resource_agent.zip` | PPO 학습 완료 모델 |
| 평가 | `data/output/rl_eval_summary.json` | reward, avg/p95 latency, fail, SLA, replica-time 요약 |
| 계획 생성 | `data/output/rl_resource_allocation_plan.csv` | RL 정책이 생성한 자원 계획 CSV |
| Docker 검증 | `data/output/rl_predictive_allocation_log.csv` | RL 계획 기반 실제 적용 로그 |
| Docker 검증 | `data/output/rl_hybrid_correction_log.csv` | RL 계획 위에 동작한 보정 로그 |
| Docker 검증 | `data/output/rl_loadgen_result.csv` | 실제 load generator 성능 결과 |
| 시각화 | `results/*.png` | RL 결과 그래프 |

---

## 8. 현재 한계

현재 RL 버전은 **실제 Docker 환경을 직접 학습하지 않고**, 로그/시뮬레이션 기반 환경에서 학습합니다.

즉:

- 장점:
  - 빠른 반복 학습 가능
  - 토큰 비용 없음
  - 로컬 연산 자원만 사용
- 한계:
  - 실제 Docker 실행과 차이 존재
  - 시뮬레이터 정확도에 따라 결과 차이 발생 가능

따라서 현재 단계에서는:

1. 시뮬레이터 기반 RL 정책 학습
2. 기존 rule-based plan과 비교
3. 나중에 실제 Docker 실행 환경으로 연결

순서로 가는 것이 적절합니다.

---

## 9. 한 줄 요약

이 디렉토리는 **Predictive의 규칙 기반 자원 정책을 RL 기반 정책으로 확장하기 위한 실험용 골격**이며,
`predictive-resource`의 예측 결과와 로그를 그대로 활용해
`비슷한 품질을 유지하면서 더 적은 자원으로 운영하는 정책`을 학습하는 것을 목표로 합니다.
