def build_mask(mask):
    if mask is None:
        return lambda x: x
    return lambda x: x[mask]
