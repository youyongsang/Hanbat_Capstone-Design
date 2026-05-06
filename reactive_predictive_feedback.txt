# Reactive vs Predictive 실험 피드백

## 1. 현재 실험 결과 해석

현재 그래프와 로그만 보면 Predictive 방식이 Reactive 방식보다 명확히 우수하다고 말하기 어렵다.

Reactive 방식은 같은 트래픽 시나리오에서 다음과 같은 결과를 보였다.

- `fail_count`가 거의 0에 가까움
- `sla_violation_count`도 거의 0에 가까움
- `success_count`가 `target_rps`를 대부분 따라감
- 평균 지연시간과 p95 지연시간도 Predictive보다 낮게 나타남

따라서 현재 조건에서는 Reactive Hybrid가 매우 강한 baseline으로 동작하고 있으며, 단순히 성공률이나 SLA 위반만으로는 Predictive 방식으로 전환해야 하는 이유가 약하다.

## 2. SLA 기준 문제

두 방식 모두 `sla_violation_count` 기준은 동일하다.

```python
SLA_LATENCY_MS = 2000
sla_violation = http_ok and (latency_ms > SLA_LATENCY_MS)
```

즉 HTTP 요청이 성공했고, 응답 지연시간이 2000ms를 초과할 때만 SLA 위반으로 계산된다.

이 기준은 현재 실험에서는 다소 느슨하다. Reactive 방식의 p95 지연시간이 1200~1800ms까지 올라가도 2000ms를 넘지 않으면 SLA 위반으로 잡히지 않는다. 따라서 `sla_violation_count`가 0에 가깝다는 사실만으로 서비스 품질이 충분히 좋다고 단정하기는 어렵다.

더 엄격한 서비스 품질을 평가하려면 다음 기준도 함께 실험하는 것이 좋다.

- SLA = 1000ms
- SLA = 800ms
- SLA = 500ms

이렇게 기준을 낮추면 Reactive 방식의 지연시간 문제가 더 명확히 드러날 수 있다.

## 3. 입력 CSV에 대한 해석

Reactive와 Predictive 모두 실제 부하 생성에는 같은 CSV를 사용한다.

```text
data/input/sale_event_traffic.csv
```

Reactive 방식은 이 CSV를 기반으로 실제 요청을 발생시키고, 실행 중 `/metrics`를 보고 자원을 조절한다.

Predictive 방식도 실제 요청은 같은 CSV로 발생시킨다. 다만 자원 할당은 예측 결과로 생성된 `resource_allocation_plan.csv`를 따라간다.

정리하면 다음과 같다.

```text
실제 테스트 부하:
- Reactive: sale_event_traffic.csv
- Predictive: sale_event_traffic.csv

자원 조절 기준:
- Reactive: 실시간 metrics
- Predictive: 예측 기반 resource_allocation_plan.csv + Reactive 보정
```

따라서 현재 비교는 같은 트래픽 조건에서 수행되고 있지만, Predictive 방식은 순수 예측 방식이 아니라 "예측 기반 선제 할당 + Reactive 보정" 구조에 가깝다.

## 4. Predictive 방식의 현재 문제점

현재 Predictive 방식의 지연시간이 큰 이유는 입력 데이터 차이보다는 자원 계획과 실행 정책 문제일 가능성이 크다.

### 4.1 Scale-out 시점이 늦음

피크 트래픽은 약 400초 구간부터 급격히 증가한다. 그러나 Predictive 계획에서는 피크 초반에도 replica 수가 충분히 빨리 증가하지 않는다.

관찰된 흐름은 대략 다음과 같다.

```text
400초 부근: 3 replicas
417초 부근: 4 replicas
459초 부근: 5 replicas
```

즉 피크가 시작된 뒤에야 replica가 증가하는 구조라서, Predictive의 핵심 장점인 "미리 준비"가 충분히 드러나지 않는다.

### 4.2 CPU 계획이 흔들림

피크 구간에서 `planned_cpu`가 5.0과 6.0 사이를 반복적으로 오가는 현상이 있다.

```text
5.0 -> 6.0 -> 5.0 -> 6.0
```

이런 반복은 작은 예측 오차가 Docker 자원 변경으로 과하게 증폭되는 현상이다. 피크 구간에서 Docker `update`가 자주 발생하면 오히려 시스템 overhead가 생기고 latency가 증가할 수 있다.

### 4.3 Reactive 보정이 계획 레이어에 덮일 수 있음

Predictive 방식에는 `HybridController`가 있어 실제 지연시간, 실패 수, CPU 사용률을 보고 보정한다.

하지만 `PredictiveAllocator`가 다음 계획 tick에서 다시 예측 계획 기준으로 상태를 갱신하면, 보정 레이어가 올린 자원이 계속 계획값에 끌려갈 수 있다.

즉 보정이 누적되어 안정화되기보다는, 예측 계획과 보정 정책이 서로 간섭할 가능성이 있다.

## 5. 발표 자료 수정 방향

현재 발표 자료의 핵심 논리는 다음과 같다.

```text
Reactive는 사후 대응이라 부족하다.
따라서 Predictive가 필요하다.
Predictive가 더 좋은 결과를 냈다.
```

그러나 현재 그래프와 로그는 이 주장과 완전히 일치하지 않는다.

따라서 발표 논리는 다음처럼 수정하는 것이 안전하다.

```text
Reactive Hybrid는 현재 SLA=2000ms 조건에서는 매우 강한 baseline이었다.
실패율과 SLA 위반을 거의 0에 가깝게 유지했다.
따라서 단순 안정성 지표만으로는 Predictive 방식의 우위를 확인하기 어렵다.

하지만 Reactive는 본질적으로 부하가 관측된 뒤에 반응하는 구조다.
더 엄격한 SLA, 더 급격한 burst, 컨테이너 시작 지연이 큰 환경에서는 한계가 나타날 수 있다.

Predictive 방식은 이러한 한계를 줄이기 위한 선제 할당 구조로 설계되었다.
현재 실험에서는 Predictive 계획의 lookahead, smoothing, scale-out 정책 튜닝이 중요하다는 점을 확인했다.
```

즉 "Predictive가 무조건 이겼다"가 아니라, "Reactive baseline이 강했고 Predictive-Hybrid 구조는 추가 튜닝이 필요하다"는 방향이 더 설득력 있다.

## 6. 슬라이드별 수정 제안

### 6.1 3페이지: Reactive 방식만으로 부족한 이유

현재 문구는 Reactive가 이미 실패한다는 전제를 강하게 깔고 있다.

수정 방향:

```text
Reactive Hybrid는 고정 자원 방식 대비 큰 성능 개선을 보였다.
현재 SLA=2000ms 조건에서는 실패율과 SLA 위반을 거의 0으로 유지했다.
하지만 이는 자원 확장이 충분히 빠르고 SLA 기준이 비교적 완화된 조건에서의 결과다.
Reactive는 구조적으로 부하를 관측한 뒤 반응하기 때문에, 더 엄격한 SLA 또는 더 급격한 burst에서는 초기 지연 spike가 발생할 수 있다.
```

### 6.2 7페이지: 예측값을 자원 계획으로 변환

현재는 `HOLD 85%`, `CPU_UP 15%`를 예측이 충분히 정확했다는 근거로 쓰고 있다.

하지만 실제 latency 그래프를 보면 보정이 적었다고 해서 성능이 좋았다고 말하기 어렵다.

수정 방향:

```text
보정 액션이 적다는 것은 계획이 안정적으로 유지되었다는 의미일 수 있지만,
반드시 latency 최적화를 의미하지는 않는다.
따라서 보정 액션 분포는 latency, SLA 위반, 자원 사용량과 함께 해석해야 한다.
```

### 6.3 8페이지: 실제 결과

현재 표에서 Predictive가 Reactive보다 명확히 우수한 것처럼 보이면, 최신 실험 결과와 충돌할 수 있다.

수정 방향:

```text
현재 실험에서는 Reactive Hybrid가 강한 baseline으로 확인되었다.
Predictive-Hybrid 방식은 예측 기반 선제 할당 구조를 구현했지만,
피크 구간에서 scale-out timing과 CPU 계획 흔들림으로 인해 latency spike가 발생했다.
향후 lookahead 기반 선제 증설과 계획 smoothing을 적용해 개선할 예정이다.
```

## 7. 코드 및 실험 개선 제안

### 7.1 SLA 기준 재실험

현재 `SLA_LATENCY_MS = 2000`은 Reactive의 한계를 드러내기에는 다소 완화된 기준이다.

다음 기준으로 재실험을 권장한다.

```text
SLA_LATENCY_MS = 1000
SLA_LATENCY_MS = 800
SLA_LATENCY_MS = 500
```

### 7.2 Predictive lookahead 적용

예측값을 해당 시점에 바로 적용하지 말고, 피크 예상 시점보다 30~60초 먼저 적용한다.

예:

```text
현재:
time_sec = 400의 계획을 400초에 적용

개선:
time_sec = 400의 계획을 340~370초에 미리 적용
```

이렇게 해야 Predictive의 "선제 할당" 장점이 실제 latency 그래프에 드러난다.

### 7.3 CPU hysteresis 적용

작은 예측 변화 때문에 CPU가 5.0과 6.0 사이를 반복하지 않도록 한다.

예:

```text
CPU 증가:
- 예측 RPS가 기준보다 10~20% 이상 증가할 때만 적용

CPU 감소:
- 낮은 부하가 일정 시간 이상 지속될 때만 적용
```

핵심은 scale-out은 빠르게, scale-in은 느리게 하는 것이다.

### 7.4 자원 비용 그래프 추가

Predictive의 장점은 latency만으로 드러나지 않을 수 있다. 따라서 다음 지표를 추가해야 한다.

```text
CPU-time = CPU 할당량 × 유지 시간
Replica-time = replica 수 × 유지 시간
평균 CPU 사용률
피크 구간 전후 자원 낭비량
```

Reactive가 latency는 낮지만 자원을 더 많이 사용했다면, Predictive는 자원 효율성 측면에서 의미를 가질 수 있다.

반대로 Predictive가 latency도 높고 자원도 많이 쓴다면, 현재 정책은 개선 필요성이 크다고 해석해야 한다.

## 8. 보고서용 결론 문장 예시

현재 결과를 정직하게 반영하면 다음 문장이 적절하다.

```text
본 실험에서 Reactive Hybrid 방식은 고정 자원 방식 대비 큰 성능 개선을 보였으며,
SLA=2000ms 조건에서는 실패율과 SLA 위반을 거의 0에 가깝게 유지하였다.
이는 Reactive 방식이 단순 baseline이 아니라 매우 강한 비교 대상임을 의미한다.

반면 Predictive-Hybrid 방식은 예측 기반 선제 할당 구조를 구현했으나,
피크 구간에서 scale-out 시점이 충분히 선제적이지 않았고 CPU 계획이 반복적으로 흔들리면서
일부 latency spike가 발생하였다.

따라서 현재 결과는 Predictive 방식의 우월성을 단정하기보다는,
예측 기반 자원 계획에서 lookahead, smoothing, scale-out 정책 튜닝이 성능에 결정적이라는 점을 보여준다.
향후 더 엄격한 SLA 조건과 급격한 burst 시나리오에서 Predictive-Hybrid 방식의 효과를 재검증할 필요가 있다.
```

## 9. 추천 결론

현재 실험을 그대로 사용한다면 발표의 메시지는 다음처럼 잡는 것이 가장 안전하다.

```text
Reactive는 현재 조건에서 매우 잘 동작했다.
Predictive는 구조적으로 필요한 방향이지만, 현재 구현은 아직 튜닝이 부족하다.
이번 실험은 Predictive 방식의 완성된 우수성보다,
Predictive-Hybrid 시스템을 안정화하기 위해 필요한 조건을 발견한 결과다.
```

이 방향이 결과와 가장 잘 맞고, 질문 방어에도 유리하다.
