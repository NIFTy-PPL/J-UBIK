import pytest

import numpy as np
from ..sum_integration import build_sum_integration


def test_sum_integration():
    # Setup
    hi_ar = np.arange(16).reshape(4, 4)
    sum_integration = build_sum_integration(hi_ar.shape, 2)

    # test build fail
    with pytest.raises(ValueError):
        build_sum_integration(hi_ar.shape, 3)

    # test apply
    assert np.allclose(np.array(((10, 18), (42, 50))), sum_integration(hi_ar))
    assert not np.allclose(0, sum_integration(hi_ar))
