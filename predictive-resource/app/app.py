from fastapi import FastAPI
from collections import deque
from threading import Lock
from typing import Deque, Dict, Any
import time
import math

# FastAPI 애플리케이션 생성
# → HTTP 요청을 받아 처리하는 서버 역할
app = FastAPI()


# ================================
# 요청 로그 저장 구조
# ================================

# 최근 요청들의 정보를 저장하는 큐 (고정 길이)
# → maxlen=10000: 최신 10,000개 요청만 유지 (메모리 보호)
# → 오래된 데이터는 자동 삭제됨
request_logs: Deque[Dict[str, Any]] = deque(maxlen=10000)

# 멀티스레드 환경에서 데이터 충돌 방지
# → 여러 요청이 동시에 request_logs를 수정할 수 있기 때문
log_lock = Lock()


# ================================
# 기본 확인용 엔드포인트
# ================================

@app.get("/")
def root():
    """
    서버 정상 동작 확인용 API
    → 브라우저에서 접속 시 서버가 살아있는지 체크
    """
    return {"message": "Reactive baseline-compatible server running"}


# ================================
# 실제 부하 발생 엔드포인트
# ================================

@app.get("/work")
def work():
    """
    실제 트래픽 부하를 발생시키는 핵심 API

    역할:
    - CPU 연산을 수행하여 서버에 부하를 줌
    - 요청 처리 시간(latency)을 측정
    - 결과를 로그(request_logs)에 저장
    """

    # 요청 시작 시간 기록
    start = time.time()

    # 기본 성공 상태
    success = True
    error_msg = None

    try:
        # ============================
        # CPU 부하 발생 구간
        # ============================
        total = 0
        for i in range(300000):
            total += i * i
        # → 반복 연산을 통해 CPU 사용률 증가
        # → 실제 네트워크/DB 대신 CPU-bound 작업을 시뮬레이션

    except Exception as e:
        # 예외 발생 시 실패 처리
        success = False
        error_msg = str(e)
        total = None

    # latency 계산 (ms 단위)
    latency_ms = (time.time() - start) * 1000.0

    # ============================
    # 요청 로그 기록
    # ============================
    with log_lock:
        request_logs.append({
            "timestamp": time.time(),   # 요청 종료 시각
            "latency_ms": latency_ms,   # 처리 시간
            "success": success          # 성공 여부
        })

    # 클라이언트에게 결과 반환
    return {
        "success": success,
        "latency_ms": latency_ms,
        "value": total,
        "error": error_msg
    }


# ================================
# 성능 메트릭 조회 API
# ================================

@app.get("/metrics")
def get_metrics(window_sec: int = 5):
    """
    최근 일정 시간(window_sec) 동안의 성능 지표를 계산

    주요 목적:
    - Reactive Controller가 참고할 지표 제공
    - 시스템 상태를 실시간으로 모니터링

    반환 지표:
    - 요청 수
    - 성공/실패 수
    - 평균 latency
    - p95 latency (상위 5% 지연)
    - 현재 RPS
    """

    now = time.time()

    # ============================
    # 최근 window_sec 구간 데이터 필터링
    # ============================
    with log_lock:
        recent = [x for x in request_logs if now - x["timestamp"] <= window_sec]

    total_count = len(recent)

    # 성공/실패 개수 계산
    success_count = sum(1 for x in recent if x["success"])
    fail_count = total_count - success_count

    # ============================
    # latency 계산
    # ============================

    # latency 값 정렬 (p95 계산을 위해 필요)
    latencies = sorted(x["latency_ms"] for x in recent)

    # 평균 latency
    avg_latency_ms = sum(latencies) / total_count if total_count > 0 else 0.0

    # ============================
    # p95 latency 계산
    # ============================
    if total_count == 0:
        p95_latency_ms = 0.0
    else:
        # 상위 95% 위치 인덱스 계산
        idx = max(
            0,
            min(len(latencies) - 1,
                math.ceil(len(latencies) * 0.95) - 1)
        )
        p95_latency_ms = latencies[idx]

    # ============================
    # 현재 처리량 (RPS)
    # ============================
    current_rps = total_count / window_sec if window_sec > 0 else 0.0

    # ============================
    # 결과 반환
    # ============================
    return {
        "window_sec": window_sec,
        "request_count": total_count,
        "success_count": success_count,
        "fail_count": fail_count,
        "avg_latency_ms": round(avg_latency_ms, 3),
        "p95_latency_ms": round(p95_latency_ms, 3),
        "current_rps": round(current_rps, 3)
    }