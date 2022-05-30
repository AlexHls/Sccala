import os
from numpy import genfromtxt


def get_data_path():
    # TODO update once settings directory has been implemented
    paths = genfromtxt(os.path.abspath("paths.txt"), dtype=str, delimiter=" ")
    data_path = paths[0][1]
    return data_path


def get_diag_path():
    # TODO update once settings directory has been implemented
    paths = genfromtxt(os.path.abspath("paths.txt"), dtype=str, delimiter=" ")
    data_path = paths[1][1]
    return data_path


def get_res_path():
    # TODO update once settings directory has been implemented
    paths = genfromtxt(os.path.abspath("paths.txt"), dtype=str, delimiter=" ")
    data_path = paths[2][1]
    return data_path
