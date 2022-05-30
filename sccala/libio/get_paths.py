import os
from numpy import genfromtxt


def get_data_path():
    paths = genfromtxt(
        os.path.abspath(".sccala_proj/.sccala_paths"), dtype=str, delimiter=" "
    )
    data_path = paths[0][1]
    return data_path


def get_diag_path():
    paths = genfromtxt(
        os.path.abspath(".sccala_proj/.sccala_paths"), dtype=str, delimiter=" "
    )
    data_path = paths[1][1]
    return data_path


def get_res_path():
    paths = genfromtxt(
        os.path.abspath(".sccala_proj/.sccala_paths"), dtype=str, delimiter=" "
    )
    data_path = paths[2][1]
    return data_path
