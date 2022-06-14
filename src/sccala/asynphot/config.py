import os

import numpy as np

from sccala.asynphot import __path__ as ASYNPHOT_PATH


def set_vega_spec(file):
    """
    Set a Vega reference spectrum.
    :param file: Path to Vega reference spectrum
    :return: None
    """

    try:
        wav, flux = np.genfromtxt(file).T
    except:
        raise AssertionError(
            "Reference spectrum does not have needed format, make sure to use a .ascii file..."
        )

    if os.path.exists(get_vega_path()):
        check = input(
            "WARINING: Vega reference spectrum already exists, continue? y/[n]: "
        )
        if check != "y" and check != "yes":
            print("Aborting...")
            return None

    np.savetxt(get_vega_path(), np.array([wav, flux]).T, newline="\n")
    print("Finished configuring Vega reference spectrum!")


def get_vega_path():
    return os.path.join(ASYNPHOT_PATH[0], "vega_spec", "vega_ref.ascii")
