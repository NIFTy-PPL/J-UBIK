"""Parse `radio_response.calibration`: the priors of the antenna gains.

```yaml
radio_response:
  calibration:
    dt: 20.0                   # seconds, spacing of the gain time grid
    phase:                     # rad, one track per polarization and antenna
      intercept: [0.0, 0.5]    # [mean, std] of the value at the first grid point
      slope: [0.0, 0.0]        # [mean, std] of the linear trend, rad per second
      deviations: [0.5, 0.2]   # [mean, std] of the smooth deviations over the whole observation
    log_amplitude:
      intercept: [0.0, 0.1]
      slope: [0.0, 0.0]
      deviations: [0.1, 0.05]
```

With `calibration` set, every observation gets its own gains: a phase and a
log-amplitude track per polarization and antenna, each an integrated Wiener
process over the time of the observation, see `CalibrationIWP`. The
deviations are lognormal, so their mean and std must be positive. Without
`calibration` the model visibilities have no gains.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TrackPrior:
    """The prior of one kind of gain track, `[mean, std]` pairs of `CalibrationIWP`.

    Parameters
    ----------
    intercept: tuple[float, float]
        The value at the first grid point.
    slope: tuple[float, float]
        The linear trend, per unit of observation time.
    deviations: tuple[float, float]
        The smooth deviations from the trend over the whole observation.
    """

    intercept: tuple[float, float]
    slope: tuple[float, float]
    deviations: tuple[float, float]

    @classmethod
    def from_yaml_dict(cls, name: str, raw: dict) -> "TrackPrior":
        """Parse the entry `name` of `radio_response.calibration`.

        Raises
        ------
        ValueError
            If a key is missing or unknown, a pair is not `[mean, std]` with
            `std >= 0`, or the lognormal `deviations` are not positive.
        """
        _check_keys(f"calibration.{name}", raw, cls)
        pairs = {}
        for key, pair in raw.items():
            pair = tuple(float(v) for v in pair)
            if len(pair) != 2 or pair[1] < 0:
                raise ValueError(
                    f"`calibration.{name}.{key}` must be [mean, std] with std >= 0, "
                    f"got {raw[key]!r}"
                )
            pairs[key] = pair
        # the deviations are a lognormal prior, which nifty rejects at zero
        if min(pairs["deviations"]) <= 0:
            raise ValueError(
                f"`calibration.{name}.deviations` is lognormal: mean and std must be "
                f"positive, got {raw['deviations']!r}"
            )
        return cls(**pairs)


@dataclass(frozen=True)
class CalibrationConfig:
    """The `calibration` entry of `radio_response`.

    Parameters
    ----------
    dt: float
        The spacing of the gain time grid, in the units of `Observation.time`.
    phase: TrackPrior
        The prior of the phase tracks in rad.
    log_amplitude: TrackPrior
        The prior of the log-amplitude tracks.
    """

    dt: float
    phase: TrackPrior
    log_amplitude: TrackPrior

    @classmethod
    def from_yaml_dict(cls, raw: dict | None) -> "CalibrationConfig | None":
        """Parse the `calibration` entry of `radio_response`.

        Parameters
        ----------
        raw: dict | None
            The `calibration` entry; None means no gains.

        Returns
        -------
        CalibrationConfig | None
            None if `raw` is None.

        Raises
        ------
        ValueError
            If a key is missing or unknown, `dt` is not positive, or a track
            prior is invalid, see `TrackPrior.from_yaml_dict`.
        """
        if raw is None:
            return None
        _check_keys("calibration", raw, cls)
        dt = float(raw["dt"])
        if dt <= 0:
            raise ValueError(f"`calibration.dt` must be positive, got {dt}")
        return cls(
            dt=dt,
            phase=TrackPrior.from_yaml_dict("phase", raw["phase"]),
            log_amplitude=TrackPrior.from_yaml_dict(
                "log_amplitude", raw["log_amplitude"]
            ),
        )


def _check_keys(where: str, raw: dict, cls: type) -> None:
    """Raise a `ValueError` naming the keys of `raw` that are not the fields of
    `cls`, or the fields missing from `raw`."""
    expected = [f.name for f in fields(cls)]
    missing = [k for k in expected if k not in raw]
    unknown = [k for k in raw if k not in expected]
    if missing or unknown:
        raise ValueError(
            f"`{where}` needs exactly {expected}: missing {missing}, unknown {unknown}"
        )
