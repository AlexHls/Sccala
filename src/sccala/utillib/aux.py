import os
import sys
from contextlib import contextmanager

import numpy as np
import scipy.stats as st
import scipy.optimize as op

from sccala.utillib.const import H_ERG, C_AA, C_LIGHT


def calc_single_error(err_low, err_high, mode="mean"):
    """
    Calculates single error from asymmetric errors
    """
    if mode == "mean":
        return (err_low + err_high) / 2
    if mode == "max":
        return max(err_low, err_high)
    if mode == "min":
        return min(err_low, err_high)


def prior_tune(l, u):
    """
    Function to fune tune inverse gamma function
    parameters following the example of
    https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#323_Informative_Prior_Model
    """

    def tail_delta(y, l, u):
        a, b = y
        return (
            st.invgamma.cdf(l, a, scale=b) - 0.01,
            1 - st.invgamma.cdf(u, a, scale=b) - 0.01,
        )

    delta = 1
    a_0 = (delta * (u + l) / (u - l)) ** 2 + 2
    b_0 = ((u + l) / 2) * ((delta * (u + l) / (u - l)) ** 2 + 1)
    log_a_0 = np.log(a_0)
    log_b_0 = np.log(b_0)
    y0 = [log_a_0, log_b_0]

    a, b = op.fsolve(tail_delta, y0, args=(l, u))
    print("a =", a)
    print("b =", b)
    assert all(np.isclose(tail_delta((a, b), l, u), [0.0, 0.0])), "Solver failed"

    return a, b


def velocity_conversion(x, rest=4861):
    """Converts wavelength into velocity with relativistic
    Doppler formula

    Parameters
    ----------
    x : float
        wavelength to convert
    rest : float
        restwavelength w.r.t to which to convert

    Returns
    -------
    vel : float
        velocity in m/s
    """
    return 299792458 * (rest**2 - x**2) / (rest**2 + x**2)


def distmod_kin(z, q0=-0.55, j0=1):
    # Hubble constant free distance modulus d = d_L * H0 in kinematic expansion
    return (C_LIGHT * z) * (1 + (1 - q0) * z / 2 - (1 - q0 - 3 * q0**2 + j0) * z**2 / 6)


def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.
    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.
    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.
    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.
    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These
    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.
    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.
    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            print(len(x), len(weights))
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.array(np.interp(q, cdf, x[idx]).tolist())


def convert_to_flux(data, data_err=None):
    """
    Converts magnitude data from mag to flux
    """
    flux = H_ERG * C_AA * np.power(10, -0.4 * data)
    if data_err is not None:
        return data_err / 2.5 * np.log(10) * flux

    return flux


def convert_to_mag(data):
    """
    Converts flux data from flux to mag
    """
    mag = -2.5 * np.log10(1 / H_ERG / C_AA * data)
    return mag


@contextmanager
def nullify_output(suppress_stdout=True, suppress_stderr=True):
    stdout = sys.stdout
    stderr = sys.stderr
    devnull = open(os.devnull, "w")
    try:
        if suppress_stdout:
            sys.stdout = devnull
        if suppress_stderr:
            sys.stderr = devnull
        yield
    finally:
        if suppress_stdout:
            sys.stdout = stdout
        if suppress_stderr:
            sys.stderr = stderr
        devnull.close()


def split_list(in_list, chunk_size):
    """
    Splits list into chunks for parallelization
    """
    for i in range(0, len(in_list), chunk_size):
        yield in_list[i : i + chunk_size]
