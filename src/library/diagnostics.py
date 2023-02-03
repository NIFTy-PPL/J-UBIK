import nifty8 as ift
import pickle


def uncertainty_weighted_residual_image_from_file(sl_path_base, ground_truth_path, sky_model=None):
    sl = ift.ResidualSampleList.load(sl_path_base)
    mean, var = sl.sample_stat(sky_model)
    with open(ground_truth_path, "rb") as f:
        rec = pickle.load(f)
    np.sqrt()
    return ()




