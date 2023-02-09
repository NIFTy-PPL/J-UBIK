import nifty8 as ift
import xubik0 as xu

if __name__ == "__main__":
    config_filename= "eROSITA_config.yaml"
    cfg = xu.get_cfg(config_filename) # FIXME OutputFilename

    sl_path_base = ""
    data_path = ""
    sky_op = SkyModel(config_filename)
    response_path ""
    output_dir_base = ""
    xu.signal_space_uwr_from_file()
    xu.data_space_uwr_from_file()
    xu.signal_space_uwm_from_file()
    xu.weighted_residual_distribution()
