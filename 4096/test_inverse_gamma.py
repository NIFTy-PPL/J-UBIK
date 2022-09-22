#!/usr/bin/env python3

import nifty8 as ift

position_space = ift.RGSpace(15)
inverse_gamma = ift.InverseGammaOperator(position_space, 2, 2, 2)
xi = ift.from_random(inverse_gamma.domain)
ift.extra.check_operator(inverse_gamma, xi)
