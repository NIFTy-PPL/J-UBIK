
def build_sum_integration(
    high_res_shape: tuple,
    reduction_factor: int
):
    if (high_res_shape[0] % reduction_factor != 0) or (high_res_shape[1] % reduction_factor != 0):
        raise ValueError(
            "The reduction factor must evenly divide both dimensions")

    new_shape = (high_res_shape[0] // reduction_factor, reduction_factor,
                 high_res_shape[1] // reduction_factor, reduction_factor)

    return lambda x: x.reshape(new_shape).sum(axis=(1, 3))


def build_sum_integration_old(
    shape: tuple,
    reduction_factor: int
):
    assert shape[0] == reduction_factor ** 2

    return lambda x: x.sum(axis=0)
