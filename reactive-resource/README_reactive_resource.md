# Reactive Resource Allocation README

## 1. 프로젝트 개요

이 프로젝트는 **Reactive 기반 동적 자원 할당 시스템**을 구현하기 위한 실험 환경이다.  
기본 아이디어는 서버의 실시간 상태를 확인하고, 지연시간(latency), 실패 수(fail count), CPU 사용률 등을 바탕으로 **CPU 자원 및 컨테이너 수를 동적으로 조절**하는 것이다.

이 프로젝트의 목적은 다음과 같다.

- 고정 자원 할당 방식과 Reactive 방식의 차이 비교
- CPU scaling만으로 충분한지 확인
- 필요할 경우 컨테이너 수까지 늘리는 hybrid scaling 실험
- 이후 LSTM 기반 Predictive 자원 할당으로 확장하기 위한 기반 마련

---

## 2. Reactive 모델 설명

### 2.1 기본 개념

Reactive 방식은 **현재 시스템 상태를 보고 반응하는 방식**이다.

즉, 다음과 같은 흐름으로 동작한다.

1. 서버가 요청을 처리한다.
2. `/metrics` API를 통해 최근 요청 상태를 수집한다.
3. 평균 지연시간, p95 지연시간, 실패 수, CPU 사용률 등을 분석한다.
4. 상태가 나빠졌다고 판단되면 CPU를 늘리거나 컨테이너를 추가한다.
5. 상태가 안정되면 다시 자원을 줄인다.

즉, Predictive 방식처럼 미래를 예측하는 것이 아니라, **이미 관찰된 결과를 보고 대응**한다.

---

### 2.2 본 프로젝트의 Reactive 정책

본 프로젝트에서는 두 가지 수준의 자원 제어를 사용한다.

#### (1) CPU Scaling
먼저 단일 컨테이너 내부에서 CPU 제한을 증가시킨다.

예:
- 0.5 → 1.0 → 1.5 → 2.0 → 3.0

이 방식은 **vertical scaling**에 해당한다.

---

#### (2) Container Scaling
CPU를 최대까지 올렸는데도 여전히 성능이 부족하면, 컨테이너 수를 증가시킨다.

예:
- 1개 → 2개 → 3개 → 4개 → 5개

이 방식은 **horizontal scaling**에 해당한다.

---

### 2.3 Hybrid Reactive 전략

최종 Reactive 구조는 다음과 같이 설계하였다.

- 먼저 CPU를 늘린다.
- CPU가 최대치에 도달했는데도 지연시간이 높거나 CPU 사용률이 높으면 컨테이너를 늘린다.
- 반대로 부하가 줄어들면 먼저 컨테이너를 줄이고, 이후 CPU도 줄인다.

이 전략을 통해 단순 CPU 조절의 한계를 보완하고, burst 트래픽 환경에서 더 안정적인 처리를 목표로 한다.

---

### 2.4 SLA 및 Reactive 반응 기준

실험에서 Load Generator는 요청 단위 응답 시간이 SLA 기준을 넘는지 확인한다.

현재 기준:

```text
SLA_LATENCY_MS = 1000
REQUEST_TIMEOUT = 5
```

즉 HTTP 요청이 성공했더라도 end-to-end latency가 1000ms를 초과하면 `sla_violation_count`로 기록된다.
`fail_count`는 HTTP 오류 또는 요청 예외처럼 요청 자체가 실패한 경우를 의미하며, 느리지만 성공한 요청은 `fail_count`가 아니라 `sla_violation_count`로 분리된다.

Controller는 서버의 `/metrics` 값과 Load Generator의 `data/output/loadgen_result.csv` 최신 값을 함께 사용한다.
서버 `/metrics`는 FastAPI handler 내부 처리시간을 기준으로 하므로, 요청 대기열과 네트워크 왕복 시간을 포함한 end-to-end latency보다 작게 나올 수 있다.
따라서 Hybrid Reactive Controller는 `loadgen_result.csv`의 최신 `avg_latency_ms`, `p95_latency_ms`, `sla_violation_count`를 함께 반영해 실제 사용자 기준 SLA 위반을 놓치지 않도록 한다.

Hybrid Reactive Controller의 scale-out 기준은 SLA보다 너무 이른 시점에 반응하지 않도록 다음과 같이 설정하였다.

```text
MAX_CPU = 6.0
MAX_REPLICAS = 8
LATENCY_UP_THRESHOLD = 500
P95_UP_THRESHOLD = 750
CPU_UP_USAGE_THRESHOLD = 60
CPU_CONTAINER_OUT_THRESHOLD = 70
```

`MAX_CPU`와 `MAX_REPLICAS`는 Predictive 방식과 동일하게 맞췄다.
따라서 비교의 핵심은 자원 상한 차이가 아니라, Reactive가 성능 저하를 관측한 뒤 확장하는지 Predictive가 피크 전에 미리 확장하는지에 있다.

또한 Load Generator 기준 `sla_violation_count`가 1 이상이면 과부하로 판단하여 scale-out 후보로 본다.

이전처럼 평균 80ms, p95 200ms 수준에서 바로 scale-out하면 SLA 1000ms/2000ms에 도달하기 전에 Reactive가 과도하게 빠르게 대응하므로, burst 구간에서 Reactive의 사후 대응 한계가 그래프에 잘 드러나지 않는다.
반대로 SLA 직전인 평균 700ms, p95 900ms 수준에서만 반응하면 피크 구간에서도 컨테이너가 충분히 늘지 않아 Reactive를 과도하게 불리하게 만들 수 있다.
따라서 현재 설정은 Reactive가 명확한 성능 저하를 관측한 뒤 대응하되, 피크 구간에서는 CPU와 컨테이너를 충분히 확장할 수 있도록 맞춘 실험용 기준이다.

Predictive 방식과 비교할 때는 양쪽 Load Generator의 SLA 기준을 동일하게 유지해야 한다.

---

## 3. 프로젝트 디렉토리 구조 예시

```text
reactive-resource/
├── app/
│   └── app.py
├── data/
│   ├── input/
│   │   └── sale_event_traffic.csv
│   └── output/
│       ├── loadgen_result.csv
│       └── hybrid_reactive_result.csv
├── results/
├── scripts/
│   ├── hybrid_load_generator.py
│   ├── reactive_controller.py
│   ├── hybrid_reactive_controller.py
│   └── plot_hybrid_results.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 4. 주요 파일 설명

### `app/app.py`
FastAPI 서버 코드이다.

주요 API:
- `/work`: CPU 부하를 발생시키는 요청 처리 API
- `/metrics`: 최근 요청 상태(latency, success/fail, RPS 등) 제공

---

### `scripts/hybrid_load_generator.py`
CSV 트래픽 시나리오를 읽어서 요청을 발생시키는 코드이다.

역할:
- `sale_event_traffic.csv`의 `target_rps`에 맞춰 요청 전송
- 여러 컨테이너에 round-robin 방식으로 요청 분산
- 결과를 `loadgen_result.csv`에 저장

---

### `scripts/reactive_controller.py`
CPU 자원만 동적으로 조절하는 기본 Reactive Controller이다.

---

### `scripts/hybrid_reactive_controller.py`
CPU + 컨테이너 수를 함께 조절하는 하이브리드 Reactive Controller이다.

---

### `scripts/plot_hybrid_results.py`
실험 결과를 그래프로 시각화하는 코드이다.

---

## 5. 실행 전 준비

### 5.1 Python 가상환경 활성화

예시:

```bash
C:\Users\PC\anaconda3\Scripts\activate
conda activate capstone
```

---

### 5.2 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

---

### 5.3 Docker Desktop 실행

Reactive 실험에서는 Docker 기반 컨테이너를 사용하므로, 반드시 Docker Desktop이 실행 중이어야 한다.

확인 명령:

```bash
docker version
docker ps
```

---

## 6. Docker 이미지 빌드

프로젝트 루트에서 다음 명령 실행:

```bash
docker build -t reactive-server .
```

빌드 완료 후 다음 명령으로 확인 가능:

```bash
docker images
```

---

## 7. 실행 방법

## 7.1 단일 컨테이너 서버 직접 실행

고정 자원 방식 또는 단일 서버 테스트용:

```bash
docker run -d --name app_server --cpus="0.5" -p 8000:8000 reactive-server
```

서버 확인:

```bash
curl http://localhost:8000/
curl http://localhost:8000/work
curl http://localhost:8000/metrics
```

---

## 7.2 CPU Reactive Controller 실행

이 방식은 하나의 컨테이너에 대해 CPU만 동적으로 조절한다.

### 1단계. 서버 실행
```bash
docker run -d --name app_server --cpus="0.5" -p 8000:8000 reactive-server
```

### 2단계. 다른 CMD에서 Reactive Controller 실행
```bash
python scripts/reactive_controller.py
```

### 3단계. 또 다른 CMD에서 Load Generator 실행
```bash
python scripts/hybrid_load_generator.py
```

---

## 7.3 Hybrid Reactive Controller 실행

이 방식은 **CPU + 컨테이너 수**를 모두 동적으로 조절한다.

### 1단계. 기존 컨테이너 정리

Windows CMD 기준:

```bash
docker rm -f app_server_1
docker rm -f app_server_2
docker rm -f app_server_3
docker rm -f app_server_4
docker rm -f app_server_5
docker rm -f app_server_6
docker rm -f app_server_7
docker rm -f app_server_8
docker rm -f app_server
```

컨테이너가 없으면 에러가 나와도 무시 가능하다.

---

### 2단계. Hybrid Controller 실행

CMD 하나에서 Controller와 Load Generator를 함께 실행하려면 다음 명령을 사용한다.

```bash
python scripts/run_hybrid_reactive.py
```

이 스크립트는 `hybrid_reactive_controller.py`를 먼저 실행하고, 컨테이너 준비 시간을 위해 5초 대기한 뒤 `hybrid_load_generator.py`를 실행한다. Load Generator가 종료되면 Controller도 함께 종료한다.

Controller와 Load Generator를 직접 분리해서 실행하려면 아래처럼 두 CMD를 사용한다.

CMD 1:

```bash
python scripts/hybrid_reactive_controller.py
```

실행되면 다음과 같이 `app_server_1`, `app_server_2` 등의 컨테이너가 자동으로 생성될 수 있다.

예:
- `app_server_1` → `8001:8000`
- `app_server_2` → `8002:8000`
- `app_server_3` → `8003:8000`
- `app_server_4` → `8004:8000`
- `app_server_5` → `8005:8000`
- `app_server_6` → `8006:8000`
- `app_server_7` → `8007:8000`
- `app_server_8` → `8008:8000`

---

### 3단계. 다른 CMD에서 Load Generator 실행

CMD 2:

```bash
python scripts/hybrid_load_generator.py
```

이때 `hybrid_load_generator.py`는 활성 컨테이너 포트를 확인하여 요청을 round-robin 방식으로 분산한다.

---

### 4단계. Docker 상태 확인

```bash
docker ps
```

또는 Docker Desktop에서 확인 가능하다.

확인 포인트:
- 컨테이너 수 증가/감소 여부
- 각 컨테이너 CPU 사용률
- 포트 매핑 상태
- Reactive와 Predictive 비교 시 둘 다 최대 `CPU 6.0`, `replica 8` 기준으로 실행되는지

---

## 8. CMD 실행 순서 정리

### 고정 자원 방식
CMD 1:
```bash
docker run -d --name app_server --cpus="0.5" -p 8000:8000 reactive-server
```

CMD 2:
```bash
python scripts/hybrid_load_generator.py
```

---

### CPU Reactive 방식
CMD 1:
```bash
docker run -d --name app_server --cpus="0.5" -p 8000:8000 reactive-server
```

CMD 2:
```bash
python scripts/reactive_controller.py
```

CMD 3:
```bash
python scripts/hybrid_load_generator.py
```

---

### Hybrid Reactive 방식
CMD 1개로 실행:
```bash
python scripts/run_hybrid_reactive.py
```

또는 Controller와 Load Generator를 분리해서 실행:

CMD 1:
```bash
python scripts/hybrid_reactive_controller.py
```

CMD 2:
```bash
python scripts/hybrid_load_generator.py
```

---

## 9. 결과 파일

실험 후 주요 결과 파일은 다음 위치에 저장된다.

### Load Generator 결과
```text
data/output/loadgen_result.csv
```

포함 항목 예:
- time_sec
- target_rps
- actual_sent
- success_count
- fail_count
- sla_violation_count
- avg_latency_ms
- p95_latency_ms

---

### Hybrid Reactive 결과
```text
data/output/hybrid_reactive_result.csv
```

포함 항목 예:
- elapsed_sec
- current_cpu
- current_replicas
- action
- cpu_usage_percent
- avg_latency_ms
- p95_latency_ms

---

## 10. 그래프 생성

```bash
python scripts/plot_hybrid_results.py
```

생성 파일 예:
- `results/loadgen_performance.png`
- `results/hybrid_loadgen_count.png`
- `results/hybrid_loadgen_latency.png`
- `results/hybrid_resource_scaling.png`
- `results/hybrid_performance.png`

---

## 11. 결과 해석 포인트

### `loadgen_performance.png`
- 상단 그래프에서 `target_rps`, `success_count`, `fail_count`, `sla_violation_count`를 함께 확인 가능
- 하단 그래프에서 `avg_latency_ms`와 `p95_latency_ms`를 한 번에 비교 가능
- Predictive 결과의 `loadgen_performance.png`와 동일한 형태로 Reactive 결과를 비교할 때 사용 가능

---

### `hybrid_loadgen_count.png`
- success_count와 target_rps가 얼마나 비슷한지
- fail_count가 발생하는지
- sla_violation_count가 피크 구간에서 증가하는지
- `fail_count`와 `sla_violation_count`는 다른 의미이다. 요청 자체가 실패하면 `fail_count`, 요청은 성공했지만 1000ms를 초과하면 `sla_violation_count`로 본다.

---

### `hybrid_loadgen_latency.png`
- avg_latency와 p95_latency가 피크 구간에서 얼마나 상승하는지
- 컨테이너 증가 후 latency가 감소하는지
- p95가 1000ms 근처 또는 그 이상으로 올라가는 구간은 SLA 위반 가능성이 큰 구간이다.

---

### `hybrid_resource_scaling.png`
- CPU가 먼저 증가하는지
- CPU가 충분치 않을 때 컨테이너 수가 증가하는지
- CPU usage가 높은 구간에서 scaling이 발생하는지
- Reactive가 너무 빨리 scale-out하면 SLA 위반이 거의 보이지 않을 수 있으므로, scale-out 시점과 SLA 위반 시점을 함께 확인한다.

---

### `hybrid_performance.png`
- request_count와 latency 관계
- scaling 이후 성능이 안정되는지

---

## 12. 주의사항

### 12.1 Docker 이름
단일 컨테이너 CPU Reactive 실험에서는 `app_server` 이름을 사용한다.

Hybrid Reactive 실험에서는 `app_server_1`, `app_server_2` 등의 이름을 사용한다.

이름이 다르면 controller가 컨테이너를 찾지 못해 에러가 날 수 있다.
현재 Hybrid Reactive 실험은 최대 `app_server_8`까지 사용할 수 있다.

---

### 12.2 Docker Desktop 리소스
컨테이너 수가 많아지면 로컬 PC의 CPU와 메모리를 많이 사용할 수 있다.

과도하게 늘리면:
- PC가 느려질 수 있음
- 실험 결과가 왜곡될 수 있음

따라서 로컬 실험에서는 적절한 `MAX_REPLICAS`와 CPU 상한을 설정하는 것이 좋다.

---

### 12.3 Load Balancing
컨테이너를 여러 개 띄우기만 해서는 충분하지 않다. 요청이 실제로 여러 컨테이너에 분산되어야 의미가 있다.

본 프로젝트에서는 `hybrid_load_generator.py`에서 round-robin 방식으로 요청을 분산한다.

---

## 13. 요약

이 프로젝트의 Reactive 자원 할당 구조는 다음과 같이 요약할 수 있다.

- **고정 자원 방식**: 자원이 고정되어 있고 부하 변화에 대응하지 않음
- **CPU Reactive 방식**: 현재 상태를 보고 CPU 자원을 동적으로 조절
- **Hybrid Reactive 방식**: CPU 조절로 부족할 경우 컨테이너 수도 함께 조절

이를 통해 단순 vertical scaling의 한계를 보완하고, burst 트래픽 환경에서 보다 안정적인 처리 성능을 확보하는 것을 목표로 한다.
