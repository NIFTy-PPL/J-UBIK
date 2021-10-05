import nifty8 as ift
import numpy as np
from lib.output import plot_result


if True:
    dct = np.load("varinf_reconstruction.npy", allow_pickle=True).item()
    for name in dct:
        plot_result(dct[name], f"var_{name}.png")
    residual = ift.abs(dct["signal_response"] - dct["data"])
    plot_result(residual, "residual.png")
