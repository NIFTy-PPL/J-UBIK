import nifty8 as ift

with open('../models/fit_mf_trans.py', 'r') as fd:
    exec(fd.read())
builder_dct = {"sf_lh": lh, "sky0": sky, "sky1": sky, "trans01": trans}
cfg_file = '../config_mf_trans.cfg'
cfg = ift.OptimizeKLConfig.from_file(cfg_file, builder_dct)
