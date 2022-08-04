import numpy as np

from sccala.utillib.const import H_ERG, C_AA, C_LIGHT


def distmod_kin(z, q0=-0.55, j0=1):
    # Hubble constant free distance modulus d = d_L * H0 in kinematic expansion
    return (C_LIGHT * z) * (
        1 + (1 - q0) * z / 2 - (1 - q0 - 3 * q0**2 + j0) * z**2 / 6
    )


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
