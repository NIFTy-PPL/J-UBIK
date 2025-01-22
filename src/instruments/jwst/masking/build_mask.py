def build_mask(data_mask):
    if data_mask is None:
        return lambda x: x
    else:
        return lambda x: x[data_mask]
