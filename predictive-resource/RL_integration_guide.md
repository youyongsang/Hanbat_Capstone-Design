# Reinforcement Learning Integration Guide

## 1. 목적

현재 `predictive-resource`는 아래 흐름으로 동작한다.

1. 과거 트래픽 데이터를 전처리한다.
2. LSTM 모델이 미래 트래픽(RPS)을 예측한다.
3. 고정 규칙 기반 정책이 예측된 RPS를 CPU / Replica 계획으로 변환한다.
4. 실행 중에는 reactive controller가 실시간으로 보정한다.

즉 현재 구조는 `예측 모델 + 규칙 기반 자원 정책 + 반응형 보정`이다.

강화학습(RL)은 여기서 **LSTM을 대체하는 역할이 아니라**,  
**고정 규칙 기반 자원 정책을 대체하거나 보완하는 정책 결정기**로 들어가는 것이 가장 자연스럽다.

---

## 2. 강화학습이 들어갈 위치

현재 구조:

```text
과거 트래픽 데이터
→ LSTM 예측 모델
→ 고정 자원 할당 정책
→ CPU / Replica 계획 생성
→ 실제 실행
→ Reactive 보정
```

강화학습 적용 구조:

```text
과거 트래픽 데이터
→ LSTM 예측 모델
→ RL 기반 자원 할당 정책
→ CPU / Replica 계획 생성
→ 실제 실행
→ Reactive 보정
```

즉 강화학습은 **예측 결과를 실제 자원 의사결정으로 연결하는 중간 정책 계층**이다.

---

## 3. 왜 LSTM과 RL을 같이 쓰는가

### 3.1 LSTM의 역할

LSTM은 미래 트래픽을 예측한다.

- 입력: 과거 트래픽 시계열
- 출력: 미래 예측 RPS

즉 질문은 이거다.

> "앞으로 트래픽이 얼마나 들어올까?"

### 3.2 RL의 역할

강화학습은 현재 상태와 예측 결과를 보고 자원 할당 행동을 결정한다.

- 입력: 예측 RPS + 현재 시스템 상태
- 출력: 자원 할당 행동

즉 질문은 이거다.

> "그러면 지금 CPU와 Replica를 얼마로 둘까?"

### 3.3 결론

- LSTM: 미래 부하를 추정하는 예측기
- RL: 예측 결과를 바탕으로 최적 자원 할당을 선택하는 정책기

따라서 RL은 LSTM의 대체재가 아니라 **정책 최적화 계층**으로 이해하는 것이 맞다.

---

## 4. RL 설계 방향

강화학습을 붙일 때 가장 중요한 것은 아래 4가지다.

1. 상태(State)를 무엇으로 정의할지
2. 행동(Action)을 무엇으로 정의할지
3. 보상(Reward)을 무엇으로 정의할지
4. 학습 환경(Environment)을 어떻게 구성할지

---

## 5. 상태(State) 정의

강화학습 에이전트가 매 시점에 관측하는 정보다.

현재 프로젝트 기준으로는 아래 값들이 상태 후보가 된다.

- 현재 예측 RPS (`predicted_rps`)
- 현재 실제 RPS (`current_rps`)
- 현재 CPU 사용률 (`cpu_usage_pct`)
- 평균 응답시간 (`avg_latency_ms`)
- p95 응답시간 (`p95_latency_ms`)
- 실패 요청 수 (`fail_count`)
- 현재 CPU 할당량 (`current_cpu`)
- 현재 Replica 수 (`current_replicas`)

추천 상태 구성 예시:

```text
state = [
  predicted_rps,
  current_rps,
  cpu_usage_pct,
  avg_latency_ms,
  p95_latency_ms,
  fail_count,
  current_cpu,
  current_replicas
]
```

추가로 넣을 수 있는 값:

- 직전 시점 대비 RPS 변화량
- SLA 위반 수
- 트래픽 구간 정보(상승/피크/하강)

---

## 6. 행동(Action) 정의

강화학습 에이전트가 선택하는 자원 제어 행동이다.

가장 단순한 방식은 이산 행동(discrete action)이다.

예시:

```text
0 = HOLD
1 = CPU + 0.5
2 = CPU - 0.5
3 = REPLICA + 1
4 = REPLICA - 1
```

또는 더 단순하게:

```text
0 = SCALE_IN
1 = HOLD
2 = SCALE_OUT
```

하지만 현재 프로젝트처럼 CPU와 Replica를 따로 다루는 구조에서는  
아래 방식이 더 실용적이다.

추천 행동 집합:

```text
0 = HOLD
1 = CPU_UP
2 = CPU_DOWN
3 = REP_UP
4 = REP_DOWN
```

이렇게 하면 현재 `hybrid_controller.py` 구조와도 자연스럽게 연결된다.

---

## 7. 보상(Reward) 정의

보상은 강화학습이 "좋은 자원 할당"을 배우게 만드는 핵심이다.

현재 프로젝트에서는 아래 목표를 동시에 만족해야 한다.

- 실패율 줄이기
- SLA 위반 줄이기
- 응답시간 줄이기
- 불필요한 CPU/Replica 과할당 줄이기

즉 보상 함수는 다음 두 가지를 함께 반영해야 한다.

1. 서비스 품질
2. 자원 효율성

예시 보상 함수:

```text
reward =
  + (success_count 비중)
  - (fail_count * 큰 패널티)
  - (sla_violation_count * 패널티)
  - (avg_latency_ms * 가중치)
  - (cpu_allocation * 비용 가중치)
  - (replica_count * 비용 가중치)
```

간단한 예시:

```text
reward = 
  - 10 * fail_count
  - 2 * sla_violation_count
  - 0.01 * avg_latency_ms
  - 0.2 * current_cpu
  - 0.5 * current_replicas
```

핵심은:

- 실패와 SLA 위반은 강하게 벌점
- CPU/Replica 과할당도 벌점
- 즉 "안정성 + 효율성"을 동시에 학습하게 설계

---

## 8. 학습 환경(Environment) 구성

강화학습은 직접 실제 서버에서 처음부터 학습시키는 것보다  
**시뮬레이션 또는 로그 기반 환경**에서 먼저 학습시키는 것이 좋다.

### 8.1 추천 1단계: 로그/시뮬레이션 기반 학습

현재 이미 가진 데이터:

- `predicted_traffic.csv`
- `resource_allocation_plan.csv`
- `predictive_allocation_log.csv`
- `hybrid_correction_log.csv`
- `loadgen_result.csv`

이 데이터로 상태-행동-결과 관계를 근사한 환경을 만들 수 있다.

즉:

- 상태: 예측값 + 현재 시스템 상태
- 행동: CPU_UP, CPU_DOWN, REP_UP, REP_DOWN, HOLD
- 결과: latency, fail, SLA, cpu cost, replica cost

이걸 이용해 offline RL 또는 simulator-based RL을 시도할 수 있다.

### 8.2 추천 2단계: 실제 실행 환경 검증

시뮬레이터에서 어느 정도 학습한 뒤,

- `run_hybrid.py`
- `predictive_allocator.py`
- `hybrid_controller.py`

구조 안에 RL 정책기를 넣어서 실제 Docker 환경에서 검증한다.

즉:

1. 시뮬레이터에서 정책 학습
2. 실제 실행 환경에서 policy inference만 수행

이 순서가 가장 안정적이다.

---

## 9. 현재 프로젝트에 붙이는 방법

### 9.1 가장 쉬운 방식

현재 `forecast_and_plan.py` 안의 고정 정책 부분을 별도 정책 클래스로 분리한다.

예:

```text
ResourcePolicyBase
├── RuleBasedPolicy
└── RLPolicy
```

즉 현재는:

- `RuleBasedPolicy` 사용

향후에는:

- `RLPolicy` 사용

이 구조로 교체 가능하게 만드는 것이다.

### 9.2 실제 연결 흐름

```text
predicted_rps 생성
→ 현재 상태 수집
→ RLPolicy가 action 선택
→ action을 CPU / Replica 변경으로 변환
→ resource_allocation_plan.csv 또는 실시간 실행 로직에 반영
```

### 9.3 적용 위치

후보는 두 가지다.

#### 방법 A. 계획 단계 대체

`forecast_and_plan.py`에서  
기존 규칙 기반 `compute_allocation()` 대신 RL 정책 사용

장점:

- 예측 기반 전체 계획을 RL로 만들 수 있음
- 현재 구조와 가장 잘 맞음

단점:

- 장기 계획의 품질이 RL 정책 품질에 크게 좌우됨

#### 방법 B. reactive 보정 단계 대체/보완

`hybrid_controller.py`에서  
기존 규칙 기반 CPU_UP/HOLD/REP_UP 판단 대신 RL 정책 사용

장점:

- 실시간 상태를 더 직접 반영 가능
- reactive 제어 최적화에 적합

단점:

- 처음부터 실제 실행에서 바로 검증하면 위험함

### 9.4 추천

중간보고서/캡스톤 진행 순서상 아래가 가장 자연스럽다.

1. 현재 구조 유지
2. RL 정책기를 `forecast_and_plan.py`의 자원 계획 단계에 먼저 붙임
3. 이후 필요하면 `hybrid_controller.py`의 reactive 보정 단계까지 확장

즉 처음에는 **계획 정책 대체용 RL**로 시작하는 것이 좋다.

---

## 10. 구현 우선순위 제안

### 1단계

- 상태, 행동, 보상 정의
- 기존 실행 로그 기반 학습 데이터셋 정리

### 2단계

- `RuleBasedPolicy`와 `RLPolicy` 인터페이스 분리
- `forecast_and_plan.py`에서 정책 교체 가능 구조로 수정

### 3단계

- 시뮬레이터 또는 로그 기반 환경에서 RL 학습

### 4단계

- RL 정책으로 `resource_allocation_plan.csv` 생성
- 기존 고정 정책과 성능 비교

### 5단계

- 필요 시 `hybrid_controller.py` 단계까지 RL 확장

---

## 11. 중간보고서용 정리 문장

강화학습은 본 시스템에서 LSTM 예측 모델을 대체하는 역할이 아니라, 예측된 트래픽과 현재 시스템 상태를 바탕으로 CPU 및 Replica 할당 정책을 최적화하는 의사결정 계층으로 활용될 수 있다. 즉 LSTM은 미래 부하를 예측하고, 강화학습은 그 예측 결과와 실시간 상태를 입력으로 받아 자원 증감 행동을 선택하는 구조이다. 현재 구현 단계에서는 고정 규칙 기반 자원 할당 정책을 사용하고 있으나, 향후에는 강화학습 기반 정책 모듈을 도입하여 자원 효율성과 서비스 안정성을 동시에 고려하는 최적 정책을 학습하도록 확장할 계획이다.

---

## 12. 한 줄 요약

> 강화학습은 LSTM을 대체하는 것이 아니라, LSTM이 예측한 트래픽과 현재 시스템 상태를 바탕으로 CPU / Replica 자원 할당 행동을 최적화하는 정책 모듈로 사용하는 것이 가장 적절하다.
