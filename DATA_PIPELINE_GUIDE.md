# 트래픽 CSV 파일 생성 규격

## 1. 개요

본 문서는 시나리오 기반 트래픽 데이터를 CSV 파일 형태로 생성하기 위한 규격을 정의한다.  
해당 CSV 파일은 load generator, Reactive 방식 실험, LSTM 모델 학습 등에 공통 입력으로 사용된다.

---

## 2. 파일 기본 구조

트래픽 데이터는 CSV 형식으로 생성하며, 각 행은 1초 단위의 트래픽 정보를 나타낸다.

### 기본 컬럼

time_sec,target_rps,scenario,phase

---

## 3. 컬럼 정의

### time_sec
- 실험 시작 기준 시간 (초)
- 0부터 시작하는 정수
- 1씩 증가

### target_rps
- 해당 시점의 요청 수 (Requests Per Second)
- 정수값 사용

### scenario
- 시나리오 이름
- 파일 전체에서 동일

예: sale_event

### phase
- 현재 구간 상태

가능 값:
normal, ramp_up, peak, cool_down, stable

---

## 4. 예시 CSV

time_sec,target_rps,scenario,phase
0,92,sale_event,normal
1,88,sale_event,normal
2,95,sale_event,normal
300,110,sale_event,ramp_up
480,620,sale_event,peak
700,210,sale_event,cool_down
1000,98,sale_event,stable

---

## 5. 생성 규칙

### 시간 단위
- 1초 단위
- 반드시 연속

### 데이터 길이
- 900 ~ 1200초 권장

#### 근거

- 본 실험은 단일 시나리오 내에서 **트래픽 증가, 피크, 감소, 안정화 구간을 모두 포함**하는 것을 목표로 한다  
- 최소 15분 이상의 데이터는 이러한 전체 흐름을 포함하기에 충분하다  
- Reactive 방식의 경우 트래픽 변화 이후의 **지연된 자원 할당 반응을 관찰할 수 있는 시간 확보가 필요하다**  
- 또한 실험 반복 수행 및 다양한 시나리오 비교를 고려할 때, 과도하게 긴 데이터는 실험 효율을 저하시킬 수 있다  

따라서 본 연구에서는 **실험 효율성과 시계열 패턴 학습 가능성을 동시에 고려하여 900~1200초 범위를 설정하였다**

### 값 변화
- 시나리오에 따라 변화
- 구간 특징 유지

---

## 6. 타임세일 시나리오 기준

normal: 80~100 RPS  
→ 평상시 사용자 트래픽 수준으로, 서버에 큰 부하가 없는 상태

ramp_up: 100~250 RPS  
→ 이벤트 시작 전 사용자 유입이 증가하는 구간으로, 트래픽이 점진적으로 상승하는 단계

peak: 500~700 RPS  
→ 타임세일 시작 직후 요청이 집중되는 구간으로, 최대 부하가 발생하는 핵심 구간

cool_down: 150~300 RPS  
→ 이벤트 이후 트래픽이 감소하는 구간으로, 사용자 이탈에 따라 점진적으로 부하가 줄어드는 단계

stable: 90~110 RPS  
→ 이벤트 종료 후 다시 평상 수준으로 돌아온 상태

---

## 7. 주의사항

### 노이즈 포함
값은 완전히 일정하면 안 됨

예:
90 → 88 → 95 → 91

### 흐름 유지
ramp_up → 증가  
cool_down → 감소  

### 급격한 변화 포함
peak 구간에서 큰 변화 필요

---

## 8. 파일 명명 규칙

{scenario_name}_traffic.csv

예:
sale_event_traffic.csv  
periodic_load_traffic.csv  
burst_load_traffic.csv  

---

## 9. 사용 방식

- load generator → 요청 발생 기준  
- Reactive 방식 → 실험 입력  
- LSTM 모델 → 학습 데이터  

---

## 10. 요약

- 1초 단위 시계열 데이터  
- RPS 기반 표현  
- scenario, phase 포함  
- CSV 형식  
- 시나리오 패턴 유지  
