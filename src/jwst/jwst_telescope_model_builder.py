from .integration_model import build_integration


def build_data_model(
        reconstruction_grid,
        data_grid,
        data_mask,
        sky_model,
        data_model_keyword,
        subsample,
        updating=False):

    return build_integration(
        reconstruction_grid,
        data_grid,
        data_mask,
        sky_model,
        data_model_keyword,
        subsample,
        updating)
