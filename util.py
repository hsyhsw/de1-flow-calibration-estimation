def eq_within(v1, v2, margin=0.005):
    return v2 - margin < v1 < v2 + margin
