def build_mask(mask):
    """Builds a masking function from a mask."""
    if mask is None:
        return lambda x: x
    return lambda x: x[mask]
