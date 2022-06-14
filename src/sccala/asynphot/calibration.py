import os

import numpy as np

import sccala.asynphot.config as cfg


def get_vega_spectrum(vega_file=None):
    """
    Get vega spectrum from reference file
    :param vega_file: str, optional
        Path of vega reference spectrum
    :return:
    """
    if vega_file is None:
        vega_file = cfg.get_vega_path()
    if not os.path.exists(vega_file):
        raise FileNotFoundError("No Vega file found!")

    wav, flux = np.genfromtxt(vega_file).T

    return wav, flux
