# -*- coding: utf-8 -*-
import numpy as np

def allocate_resource(pred_rps: float):
    """
    예측된 RPS를 기반으로 레플리카와 CPU를 계산.
    100 RPS당 1개 + 여유분 1개, 최소 2대 최대 15대.
    """
    target_replicas = int(np.ceil(pred_rps / 100)) + 1
    replicas = min(max(2, target_replicas), 15)
    cpu = round(replicas * 0.5, 1)
    return cpu, replicas
