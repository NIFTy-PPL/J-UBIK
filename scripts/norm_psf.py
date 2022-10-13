import nifty8 as ift
import xubik0 as xu

psf_file = np.load("data/npdata/psf_patches/4952_patches_v2.npy", allow_pickle=True).item()

psfs = []
for p in psf_file:
    p_arr = p.val
    p_arr[p_arr<50] = 0
    norm_val = p.integrate().val**-1
    norm = ift.ScalingOperatorl(p.domain, norm_val)
    psf_norm = norm(p)
    psfs.append(p.val)

