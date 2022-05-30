import numpy as np
import pandas as pd

from specutils import Spectrum1D
from specutils.manipulation import gaussian_smooth
import astropy.units as u


def calculate_flux_error(datfile, stdev=4, size=1000, loc=75, scale=50):
    """Estimate Error of spectrum based on Savitzky-Golay filtering over large area and subtracting form original data

    Parameters
    ----------
    datfile : str
        Path to spectral file of which to calculate flux error
    stdev : int
        Bins over which spectrum is smoothed. Standarddeviation in Gaussian kernel. Default: 4
    size : int
        Number of windowlengths to be drawn randomly. Default: 1000
    loc : int or float
        Mean of Gaussian distribution from which windowlengths are drawn. Default: 75
    scale : int or float
        Stdev of Gaussian distribution from which windowlengths are drawn. Default: 50

    Returns
    -------
    data_err : np.ndarray
    unnormalized error
    """

    wav, flux = np.genfromtxt(datfile).T

    spec1 = Spectrum1D(
        spectral_axis=wav * u.AA, flux=flux * u.Unit("erg cm-2 s-1 AA-1")
    )
    spec1_gsmooth = gaussian_smooth(spec1, stddev=stdev)
    fl_smooth = spec1_gsmooth.flux.value
    data_smoothing_err = flux - fl_smooth

    # Calculate std-spectrum
    windowlengths = np.random.normal(size=size, loc=loc, scale=scale)
    std_specs = []
    for window in windowlengths:
        if window < 5:
            continue
        std_spec = np.zeros_like(data_smoothing_err)
        for k in range(len(std_spec)):
            if k < window:
                std_spec[k] = np.std(data_smoothing_err[: k + int(window / 2)])
            elif k > len(std_spec) - window:
                std_spec[k] = np.std(data_smoothing_err[k - int(window / 2) :])
            else:
                std_spec[k] = np.std(
                    data_smoothing_err[k - int(window / 2) : k + int(window / 2)]
                )
        std_specs.append(std_spec)

    std_spec = np.mean(std_specs, axis=0)

    df = pd.DataFrame({"Wavelength": wav, "FluxError": std_spec})
    df.to_csv(
        datfile.replace(".dat", "_error.dat"),
        header=False,
        index=False,
        sep=" ",
    )

    return std_spec
