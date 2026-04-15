def allocate_resource(pred_rps, current_rps=None):
    """
    예측된 RPS를 기반으로 (CPU, Replicas)를 결정하는 정책 함수.
    안전 계수(Buffer) 및 급증 대응 로직(Panic Mode) 포함.
    """
    # 1. 안전 계수 적용 (예측 오차를 대비해 15% 넉넉하게 할당)
    safe_rps = pred_rps * 1.15
    
    # 2. 급증 대응 (Panic Mode)
    # 현재 RPS가 주어졌고, 1초 전보다 50 이상 급증했다면 선제적 자원 투입
    if current_rps is not None and (pred_rps - current_rps) > 50:
        safe_rps = max(safe_rps, 450)
        
    # 3. 자원 할당 기준 (넉넉하게 상향 조정)
    if safe_rps < 150:
        return 0.5, 1
    elif safe_rps < 280:
        return 1.0, 2
    elif safe_rps < 450:
        return 2.0, 4  # 중간 피크 변동성에 대비해 Replicas 증가 (3->4)
    else:
        return 4.0, 6  # 극단적 피크 구간 대비 (3.0->4.0, 5->6)
