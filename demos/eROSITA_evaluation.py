import nifty8 as ift
import xubik0 as xu

if __name__ == "__main__":
    reconstruction_path = "path/to/dir/" #FIXME filepath
    config_filename = "eROSITA_config.yaml"
    sky_op = SkyModel(reconstruction_path + config_filename)
    sl_path_base = reconstruction_path + "pickle/last" # NIFTy dependency
    data_path = reconstruction_path + "data"
    response_path = ""
    output_dir_base = "diagnostics"
    xu.signal_space_uwr_from_file()
    xu.data_space_uwr_from_file()
    xu.signal_space_uwm_from_file()
    xu.weighted_residual_distribution()
