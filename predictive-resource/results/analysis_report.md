# Predictive Resource Allocation Analysis Report

## 1. 분석 대상

이번 분석은 아래 결과 파일을 기준으로 수행했다.

- [predicted_traffic.csv](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/data/output/predicted_traffic.csv)
- [resource_allocation_plan.csv](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/data/output/resource_allocation_plan.csv)
- [predictive_allocation_log.csv](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/data/output/predictive_allocation_log.csv)
- [hybrid_correction_log.csv](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/data/output/hybrid_correction_log.csv)
- [loadgen_result.csv](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/data/output/loadgen_result.csv)

생성 그래프:

- [predicted_vs_actual_rps.png](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/predicted_vs_actual_rps.png)
- [prediction_error.png](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/prediction_error.png)
- [resource_plan.png](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/resource_plan.png)
- [plan_vs_actual_resource.png](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/plan_vs_actual_resource.png)
- [correction_actions.png](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/correction_actions.png)
- [loadgen_performance.png](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/loadgen_performance.png)

---

## 2. 전체 요약

이번 실행에서는 서비스 안정성은 비교적 잘 유지되었지만, 예측 모델 성능은 충분히 좋지 않았다.

- 전체 요청 수: `309,795`
- 전체 성공 수: `309,795`
- 전체 실패 수: `0`
- SLA 위반 수: `21,994`
- 평균 latency: `706.734ms`
- 최대 p95 latency: `2553.2ms`

즉, 시스템은 요청 실패 없이 동작했지만 지연시간 측면에서는 아직 개선 여지가 크다.

---

## 3. 예측 성능 분석

`predicted_traffic.csv` 기준 모델 예측 구간 성능:

- Forecast rows: `941`
- MAE: `203.438`
- RMSE: `276.359`
- MAPE: `53.336%`
- Bias: `-172.122`
- Max absolute error: `547.245`

### 해석

- MAPE가 약 `53.3%`로 높아 예측 정확도는 좋은 편이 아니다.
- Bias가 음수이므로, 모델이 전체적으로 실제보다 낮게 예측하는 경향이 강하다.
- 후반 구간에서 `predicted_rps`가 약 `152.763` 근처로 평탄하게 유지되는 모습이 확인되며, 이는 미래 시계열 변화를 충분히 따라가지 못한 것으로 해석된다.

### 의미

현재 Predictive 레이어는 “정확한 미래 트래픽 추정기”라기보다, 대략적인 부하 추세를 반영하는 수준에 가깝다.

![Predicted vs Actual](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/predicted_vs_actual_rps.png)

![Prediction Error](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/prediction_error.png)

---

## 4. 자원 계획 분석

`resource_allocation_plan.csv` 기준 계획 분포:

- `planned_cpu = 1.0`: `1001`회
- `planned_replicas = 2`: `85`회
- `planned_replicas = 3`: `916`회
- 최대 예측 RPS: `152.763`

### 해석

- CPU 계획이 전 구간 `1.0`으로 고정되어 있어 정책이 충분히 세밀하게 반응하지 못했다.
- 대부분 구간에서 replica 3개를 유지했기 때문에, 자원 계획 자체가 사실상 보수적인 고정 할당처럼 동작했다.
- 예측 모델 출력이 넓게 변하지 않다 보니 정책 함수도 거의 같은 결과를 반복 생성했다.

### 의미

현재 정책은 “예측 기반 동적 계획”이라기보다 “약간 보수적인 정적 계획 + 일부 조정”에 가까운 모습이다.

![Resource Plan](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/resource_plan.png)

---

## 5. 실시간 보정 분석

`hybrid_correction_log.csv` 기준 보정 요약:

- `HOLD`: `281`회
- `CPU_UP`: `4`회
- 평균 `avg_latency_ms`: `73.646`
- 최대 `avg_latency_ms`: `109.814`
- 평균 `p95_latency_ms`: `247.928`
- 최대 `p95_latency_ms`: `380.402`
- 평균 `cpu_usage_pct`: `76.445`
- 최대 `cpu_usage_pct`: `106.77`
- 평균 `current_rps`: `71.339`
- 최대 `current_rps`: `94.0`
- `fail_count`: 항상 `0`

### 해석

- 보정 액션은 대부분 `HOLD`였고, 실제 scale-up은 CPU 증가 4회만 발생했다.
- 이는 계획 자체가 어느 정도 버틸 수 있는 수준이었음을 의미하지만, 동시에 reactive 레이어가 실제 안정성을 많이 보완했을 가능성도 시사한다.
- CPU 사용률이 최대 `106.77%`까지 올라간 구간이 있었지만 요청 실패는 발생하지 않았다.

### 주의점

로그상 `curr_replicas`와 실제 활성 포트 수가 완전히 일치하지 않는 구간이 보여, replica 추적 로직은 추가 검증이 필요하다.

![Plan vs Actual Resource](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/plan_vs_actual_resource.png)

![Correction Actions](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/correction_actions.png)

---

## 6. 실제 부하 처리 성능 분석

`loadgen_result.csv` 기준:

- 총 요청 대비 실패는 `0`
- 평균 latency는 `706.734ms`
- 최대 p95 latency는 `2553.2ms`
- SLA 위반 수는 `21,994`

### 해석

- 서비스는 무너지지 않았고, 처리 자체는 안정적으로 수행되었다.
- 하지만 p95 latency가 순간적으로 `2.5초`를 넘는 구간이 있어 응답 품질은 완전히 안정적이라고 보기 어렵다.
- SLA 위반이 적지 않게 발생했으므로, “성공 처리율”은 높았지만 “사용자 체감 성능”은 여전히 개선이 필요하다.

![Loadgen Performance](C:/Users/PC/Desktop/Capston%20Design/Hanbat_Capstone-Design/predictive-resource/results/loadgen_performance.png)

---

## 7. 종합 결론

이번 실험의 핵심 결론은 다음과 같다.

1. 하이브리드 구조 자체는 안정적으로 동작했다.
2. 실패 없이 전체 요청을 처리했다는 점에서 시스템 안정성은 확보되었다.
3. 그러나 예측 모델 정확도가 낮아 Predictive 레이어의 품질은 아직 부족하다.
4. 실제 안정성은 예측 자체보다는 보수적인 계획과 Reactive 보정이 뒷받침한 결과일 가능성이 높다.

즉, 이번 결과는 “하이브리드 구조의 기본 동작 검증”에는 성공했지만, “정확한 예측 기반 선제 자원 할당의 효과 입증” 단계까지는 아직 도달하지 못했다고 볼 수 있다.

---

## 8. 개선 방향

### 8.1 예측 모델 개선

- 현재 LSTM이 실제보다 낮게 예측하는 경향이 강하다.
- 멀티스텝 예측 구조 또는 feature 확장이 필요하다.
- 필요 시 모델 재학습과 입력 윈도우 조정도 검토할 수 있다.

### 8.2 자원 정책 개선

- `SAFETY_MARGIN`을 높여 더 보수적으로 계획할 수 있다.
- `CONTAINER_CAPACITY`를 낮춰 replica가 더 빨리 증가하도록 조정할 수 있다.
- CPU 단계 기준도 더 공격적으로 조정 가능하다.

### 8.3 로그 정합성 개선

- `curr_replicas`와 실제 컨테이너 상태가 일치하도록 보정 로직 검증이 필요하다.
- 분석 신뢰도를 높이기 위해 실제 Docker 상태 기반 기록을 추가하는 것이 좋다.

---

## 9. 한 줄 요약

> 이번 결과는 “예측 모델은 아직 부정확하지만, 계획 기반 선제 할당과 Reactive 보정을 결합한 하이브리드 구조는 안정적으로 동작한다”는 점을 보여준다.
