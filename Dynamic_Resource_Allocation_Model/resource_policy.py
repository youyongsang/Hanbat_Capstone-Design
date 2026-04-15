def allocate_resource(pred_rps):
    """
    예측된 RPS를 기반으로 (CPU, Replicas)를 결정하는 정책 함수.
    """
    if pred_rps < 150:
        return 0.5, 1
    elif pred_rps < 250:
        return 1.0, 2
    elif pred_rps < 400:
        return 2.0, 3
    else:
        return 3.0, 5
