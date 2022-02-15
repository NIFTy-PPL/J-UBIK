import nifty8 as ift
import numpy as np
from astropy.io import fits
from lib.output import plot_result

diffuse = fits.open("df_rec/diffuse/last_mean.fits")[0].data
diffuse_clip = np.copy(diffuse)
diffuse_clip[diffuse<1e-3] = 0
diffuse_diff = diffuse - diffuse_clip

signal = fits.open("df_rec/signal/last_mean.fits")[0].data
signal_clip = np.copy(signal)
signal_clip[signal<0.3] = 0
diff = signal - signal_clip

sp = ift.RGSpace(signal.shape)

s_field = ift.Field.from_raw(sp, signal)
diff_field = ift.Field.from_raw(sp, diff)
clip_field = ift.Field.from_raw(sp, signal_clip)
diffuse_field = ift.Field.from_raw(sp, diffuse)
diffuse_diff_field = ift.Field.from_raw(sp, diffuse_diff)
diffuse_clip_field = ift.Field.from_raw(sp, diffuse_clip)
plot_result(s_field, "SN1006.png", logscale=True, vmin=0.35)
# plot_result(diff_field, "SN1006_diff.png", logscale=True)
# plot_result(clip_field, "SN1006_clip.png", logscale=True)
# plot_result(diffuse_field, "SN1006_diffuse.png", logscale=False)
# plot_result(diffuse_diff_field, "SN1006_diffuse_diff.png", logscale=False)
# plot_result(diffuse_clip_field, "SN1006_diffuse_clip.png", logscale=False)
