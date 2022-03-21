import nifty8 as ift
import xubik0 as xu

obs_info = xu.get_cfg("obs/obs.yaml")
cfg = xu.get_cfg("config.yaml")

info = xu.ChandraObservationInformation(obs_info["obs4942"])
arr = xu.get_synth_pointsource(info, npix_s, idx_tupel, numrays)
