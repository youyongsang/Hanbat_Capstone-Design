def allocate_resource(pred_rps, current_rps=None):
    safe_rps = pred_rps * 1.15
    
    if current_rps is not None and (pred_rps - current_rps) > 50:
        safe_rps = max(safe_rps, 450)
        
    if safe_rps < 150:
        return 0.5, 1
    elif safe_rps < 280:
        return 1.0, 2
    elif safe_rps < 450:
        return 2.0, 4
    else:
        return 4.0, 6
