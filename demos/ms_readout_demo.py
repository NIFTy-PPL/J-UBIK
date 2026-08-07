# %% [markdown]
# # Data Readout Demo: measurement sets to resolve npz
#
# This demo converts interferometric measurement sets into J-UBIK's resolve
# `Observation` files (`.npz`), which is the format the imaging demos
# (`resolve_demo.py`) and the resolve configs expect as input.
#
# For every (measurement set, spectral window) pair the demo
#
# 1. splits the requested visibility column into a temporary, single-spw,
#    single-field measurement set with CASA `mstransform`,
# 2. optionally recomputes the weights on that split with CASA `statwt`,
# 3. reads the split with `ms2observations` and saves the `Observation` as npz,
# 4. deletes the split again.
#
# Only one split exists at a time, so the scratch usage stays at the size of a
# single spectral window irrespective of how large the parent measurement sets
# are. Existing npz files are skipped, which makes the readout resumable.
#
# All settings live in `demos/configs/ms_readout_demo.yaml`, here a
# [link to the file](https://github.com/NIFTy-PPL/J-UBIK/blob/main/demos/configs/ms_readout_demo.yaml).
#
# ## Requirements
#
# Besides J-UBIK itself the readout needs the CASA Python packages
# `casatasks` and `casatools` (for `mstransform`/`statwt`) and
# `python-casacore` (used by `ms2observations`). They are not J-UBIK
# dependencies; the cheapest way to get them is an ephemeral environment:
#
# ```
# uv run --with casatools --with casatasks --with python-casacore \
#     python demos/ms_readout_demo.py demos/configs/ms_readout_demo.yaml
# ```
#
# On its first run `casatools` downloads the CASA measures data (~1 GB, once)
# into `~/.casa/data`; that directory has to exist and be user-owned.
#
# ## Why the temporary split
#
# `ms2observations` assumes `DATA_DESC_ID == spectral_window == SPECTRAL_WINDOW`
# row, which only holds for a measurement set with a single spectral window.
# `mstransform` additionally moves the selected column into `DATA` and reindexes
# the spectral window to `0`, so the readout call is identical for all datasets
# no matter which column the data originally live in.
#
# ## Which visibility column
#
# The right choice of `datacolumn` depends on the pipeline that produced the
# measurement set and cannot be guessed from the file, hence it is a config
# entry per dataset:
#
# - self-calibration applied (e.g. ALMA pipelines from 2022 on, with the
#   datatype framework): `DATA` holds the calibrated pre-selfcal visibilities
#   and `CORRECTED_DATA` the self-calibrated ones -> use `corrected`.
# - older pipelines: `CORRECTED_DATA` may hold something else entirely, e.g.
#   continuum-subtracted data after `uvcontsub` -> use `data`.
# - no `CORRECTED_DATA` column present -> use `data`.
#
# The demo checks that the requested column exists before touching a
# measurement set, so a wrong guess fails immediately instead of silently
# exporting the wrong visibilities.
#
# ## Weights
#
# The resolve likelihood takes the weights literally as `1 / sigma^2`, whereas
# calibration-propagated weights (from Tsys and `applycal`) are usually only
# *relatively* correct: a global scale error in them does not affect CLEAN, but
# it directly miscalibrates a Bayesian reconstruction. CASA `statwt` discards
# these weights and recomputes them from the scatter of the visibilities
# themselves, which is why `statwt.enabled` defaults to `True` here.
#
# For spectral windows that contain a line, the line would bias the scatter
# estimate. Listing the line window under `statwt.exclude` removes it from the
# estimation sample (`fitspw` with `excludechans=True`) while the resulting
# weights are still applied to all channels, so line imaging profits from the
# data-driven weights as well.

# %%
import argparse
import shutil
import tempfile
from os import makedirs
from os.path import basename, dirname, exists, expanduser, expandvars, isabs
from os.path import join, splitext

import jubik as ju
import jubik.instruments.resolve as rve

# %% [markdown]
# ## Configuration helpers

# %%
DATA_COLUMNS = {"data": "DATA", "corrected": "CORRECTED_DATA", "model": "MODEL_DATA"}


def resolve_path(path, root=None):
    """Expand `~` and environment variables, optionally prepending a root."""
    path = expanduser(expandvars(str(path)))
    if root is not None and not isabs(path):
        path = join(root, path)
    return path


def output_path(cfg, dataset, ms, spw):
    """Build the npz path for one (measurement set, spectral window)."""
    files = cfg["files"]
    root = resolve_path(files["data_root"])
    template = files["output_template"]
    name = template.format(
        dataset=dataset["name"],
        ms=splitext(basename(ms.rstrip("/")))[0],
        field=dataset.get("field") or "",
        spw=spw,
    )
    return join(resolve_path(files["output_root"], root), name)


def check_column(ms, datacolumn):
    """Fail before any splitting if the requested column is not there."""
    from casatools import table

    colname = DATA_COLUMNS[datacolumn]
    tb = table()
    tb.open(ms)
    columns = tb.colnames()
    tb.close()
    if colname not in columns:
        raise RuntimeError(f"{ms}: column {colname} not found (has: {columns})")


# %% [markdown]
# ## Readout of a single spectral window
#
# `split_and_read` performs the three CASA-side steps and returns the
# `Observation`. The temporary split is removed in all cases, also if the
# readout raises.


# %%
def split_and_read(ms, spw, dataset, readout, scratch):
    """Split one spectral window into a temporary MS and read it out.

    Parameters
    ----------
    ms : str
        Path of the parent measurement set.
    spw : int
        Spectral window id as numbered in the parent measurement set.
    dataset : dict
        Dataset block of the config (`field`, `datacolumn`, `statwt`).
    readout : dict
        `readout` block of the config.
    scratch : str
        Directory the temporary split is created in.

    Returns
    -------
    Observation
        The single observation contained in the split.
    """
    from casatasks import mstransform, statwt

    field = dataset.get("field")
    tmpdir = tempfile.mkdtemp(prefix="ms_readout_", dir=scratch)
    try:
        split = join(tmpdir, f"spw{spw}.ms")
        mstransform(
            vis=ms,
            outputvis=split,
            field=field if field is not None else "",
            spw=str(spw),
            datacolumn=dataset["datacolumn"],
            reindex=True,
        )

        statwt_cfg = dataset.get("statwt", {}) or {}
        if statwt_cfg.get("enabled", False):
            # The column selected above now *is* DATA of the split. Weights
            # are recomputed from the visibility scatter, see the docstring.
            # flagbackup=False: no point snapshotting flags of a throwaway MS.
            kwargs = dict(vis=split, datacolumn="data", flagbackup=False)
            exclude = (statwt_cfg.get("exclude") or {}).get(spw)
            if exclude is not None:
                # The split's spectral window is reindexed to 0, hence "0:".
                kwargs.update(fitspw=f"0:{exclude}", excludechans=True)
                print(f"  statwt: excluding {exclude} from the estimate")
            statwt(**kwargs)

        observations = rve.ms2observations(
            ms=split,
            data_column="DATA",
            with_calib_info=readout.get("with_calib_info", True),
            spectral_window=0,
            polarizations=readout.get("polarizations", "stokesi"),
            ignore_flags=readout.get("ignore_flags", False),
            field=field,
        )
        observations = [oo for oo in observations if oo is not None]
        if len(observations) != 1:
            raise RuntimeError(
                f"{ms} spw {spw}: expected one observation, got "
                f"{len(observations)}. Narrow the selection via the "
                f"dataset's `field` entry."
            )
        return observations[0]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# %% [markdown]
# ## Readout of all configured datasets


# %%
def run_readout(cfg):
    """Export every (measurement set, spectral window) listed in the config."""
    files = cfg["files"]
    readout = cfg.get("readout", {}) or {}
    data_root = resolve_path(files["data_root"])
    scratch = resolve_path(files.get("scratch", "/tmp"))
    makedirs(scratch, exist_ok=True)

    from casatasks import casalog

    casalog.setlogfile(join(scratch, "casa-ms-readout.log"))

    written = []
    for dataset in cfg["datasets"]:
        for entry in dataset["measurement_sets"]:
            ms = resolve_path(entry, data_root)
            column_checked = False
            for spw in dataset["spws"]:
                out = output_path(cfg, dataset, ms, spw)
                if exists(out) and not readout.get("overwrite", False):
                    print(f"skip (exists): {out}")
                    continue
                if not column_checked:
                    check_column(ms, dataset["datacolumn"])
                    column_checked = True

                print(
                    f"{dataset['name']}: {basename(ms)} spw {spw} "
                    f"[{dataset['datacolumn']}] -> {out}"
                )
                obs = split_and_read(ms, spw, dataset, readout, scratch)
                makedirs(dirname(out), exist_ok=True)
                obs.save(out, readout.get("compress", False))
                written.append(out)
    return written


# %%
# Parser setup (only for the python interpreter, non ipython/jupyter)
parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    type=str,
    help="Config file (.yaml) for the measurement set readout.",
    nargs="?",
    const=1,
    default="configs/ms_readout_demo.yaml",
)
args = parser.parse_args()
config_path = args.config

# %%
# For ipython / jupyter
# config_path = "configs/ms_readout_demo.yaml"

# %%
cfg = ju.get_config(config_path)
written = run_readout(cfg)
print(f"wrote {len(written)} npz file(s)")

# %% [markdown]
# ## Using the result
#
# The npz files are read back with `Observation.load` and can be fed into the
# imaging demo directly:
#
# ```python
# import jubik.instruments.resolve as rve
#
# obs = rve.Observation.load(written[0])
# obs = rve.data.restrict_to_stokesi(obs)
# obs = rve.data.average_stokesi(obs)
# ```
#
# Since the exports keep XX and YY separate (`polarizations: 'stokesi'`), the
# averaging to Stokes I stays a choice of the reconstruction rather than of the
# readout.
