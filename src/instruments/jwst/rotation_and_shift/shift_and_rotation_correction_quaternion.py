"""
JWST small–angle maneuver:  pixel sky-vector after
 • a commanded (ΔV2, ΔV3) boresight translation, and
 • an optional extra roll φ around the *new* boresight.

Implemented in JAX only (no SciPy dependency).

Author: 2025-05-09
"""

import jax.numpy as jnp


# ---------- quaternion helpers ------------------------------------------------
def quat_mul(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Hamilton product of two quaternions, scalar-first convention:
       q = (w, x, y, z).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(q: jnp.ndarray) -> jnp.ndarray:
    """Quaternion conjugate (w, -x, -y, -z)."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(axis: jnp.ndarray, angle_rad: float) -> jnp.ndarray:
    """Unit quaternion for a rotation of `angle_rad` about `axis` (must be unit-norm)."""
    half = 0.5 * angle_rad
    return jnp.concatenate((jnp.array([jnp.cos(half)]), axis * jnp.sin(half)))


def quat_apply(q: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate 3-vector `vec` by quaternion `q` (scalar-first convention).
    """
    vq = jnp.concatenate((jnp.array([0.0]), vec))
    return quat_mul(quat_mul(q, vq), quat_conj(q))[1:]  # drop scalar part


# ------------------------------------------------------------------------------


ARCSEC2RAD = jnp.deg2rad(1.0 / 3600.0)  # 4.848 … × 10⁻⁶


def pixel_direction_after_maneuver(
    pixel_dir_det: jnp.ndarray,
    attitude_quat: jnp.ndarray,
    boresight_shift_arcsec: tuple[float, float],
    extra_roll_deg: float = 0.0,
) -> jnp.ndarray:
    """
    Parameters
    ----------
    pixel_dir_det         : (3,) float32/64
        Unit vector of the detector pixel in the V2/V3 (Ideal) frame.
    attitude_quat         : (4,) float
        Spacecraft attitude quaternion **before** the maneuver, scalar first.
    boresight_shift_arcsec: (ΔV2, ΔV3) tuple[float, float]
        Requested boresight translation (arcsec).  +ΔV2 = left, +ΔV3 = up on detector.
    extra_roll_deg        : float, optional
        Additional rotation *about the (new) boresight* (degrees).  Defaults to 0.

    Returns
    -------
    new_pixel_sky_dir     : (3,) float
        Unit vector of the pixel on the sky after the maneuver.
    """

    # -- 1. convert commanded translation to radians ---------------------------
    dV2_arcsec, dV3_arcsec = boresight_shift_arcsec
    dV2_rad = dV2_arcsec * ARCSEC2RAD
    dV3_rad = dV3_arcsec * ARCSEC2RAD
    delta_rad = jnp.hypot(dV2_rad, dV3_rad)

    # quick exit: no motion requested
    def _no_shift():
        return quat_apply(attitude_quat, pixel_dir_det)

    def _with_shift():
        # -- 2. direction unit-vector of desired boresight displacement in tangent plane
        t_hat = jnp.array([dV2_rad, dV3_rad, 0.0]) / delta_rad

        # -- 3. current boresight on the sky (V1 axis)
        v1_det = jnp.array([0.0, 0.0, 1.0])
        boresight_sky = quat_apply(attitude_quat, v1_det)

        # -- 4. rotation axis that produces the translation
        axis_shift = jnp.cross(boresight_sky, t_hat)
        axis_shift = axis_shift / jnp.linalg.norm(axis_shift)

        # -- 5. quaternion for the pure translation
        q_shift = quat_from_axis_angle(axis_shift, delta_rad)

        # -- 6. optional extra roll about the *new* boresight
        if extra_roll_deg != 0.0:
            phi = jnp.deg2rad(extra_roll_deg)
            # boresight AFTER shift = q_shift ⊗ boresight ⊗ q_shift*
            new_boresight = quat_apply(q_shift, boresight_sky)
            axis_roll = new_boresight / jnp.linalg.norm(new_boresight)
            q_roll = quat_from_axis_angle(axis_roll, phi)
            q_delta = quat_mul(q_roll, q_shift)  # roll after the shift
        else:
            q_delta = q_shift

        # -- 7. new attitude & apply to pixel
        q_new = quat_mul(q_delta, attitude_quat)
        return quat_apply(q_new, pixel_dir_det)

    return jnp.where(delta_rad == 0.0, _no_shift(), _with_shift())
