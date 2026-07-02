# Copyright(C) 2025
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig,

from typing import Any, Union


def check_demands(model_name, kwargs, demands):
    """Check that all demands are provided in kwargs."""
    for key in demands:
        assert key in kwargs, f"{key} not in {model_name}.\nProvide settings for {key}"
