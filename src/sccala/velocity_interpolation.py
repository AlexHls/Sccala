import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cloudpickle

from interplib import epoch_interp
from libio import get_paths as pa


def main(args):
    # TODO Make script more flexible/ accept more arguments
    snname = args.snname
    line = args.line
    rules = args.rules

    diag_path = os.path.join(pa.get_diag_path(), snname)
    if not os.path.exists(diag_path):
        os.makedirs(diag_path)

    # Load data
    data = pd.read_csv(
        os.path.join(pa.get_res_path(), "{:s}_PeakFits.csv".format(snname)),
        index_col=[0, 1],
    )
    dataframe = data.loc[line]
    vel = dataframe["PeakLoc"].to_numpy()
    vel_error_lower = dataframe["PeakErrorLower"].to_numpy()
    vel_error_upper = dataframe["PeakErrorUpper"].to_numpy()
    vel_error = np.maximum(vel_error_lower, vel_error_upper)
    mjd = dataframe["MJD"].to_numpy()

    # Import Gaussian KDE for time-prior
    data_path = os.path.join(pa.get_data_path(), snname)
    with open(os.path.join(directory, "{:s}_TimeKDE.pkl".format(snname)), "rb") as f:
        time_kde = cloudpickle.load(f)
    tkde = time_kde.resample(size=10000)

    # Load redshift from info file
    info = pd.read_csv(os.path.join(directory, "{:s}_info.csv".format(snname)))
    red = np.mean(info["Redshift"].to_numpy())

    # Unpack rules
    if rules is not None:
        rules_set = pd.read_csv(rules)
        errorfloor = rules_set[rules_set["SN"] == snname]["errorfloor"].to_numpy()[0]
        errorscale = rules_set[rules_set["SN"] == snname]["errorscale"].to_numpy()[0]
        reg_min = rules_set[rules_set["SN"] == snname]["region_min"].to_numpy()[0]
        reg_max = rules_set[rules_set["SN"] == snname]["region_max"].to_numpy()[0]
        extrapolate = rules_set[rules_set["SN"] == snname]["extrapolate"].to_numpy()[0]
    else:
        errorfloor = 0
        errorscale = 1
        reg_min = 20
        reg_max = 60
        extrapolate = 5

    vel_set = EpochDataSet(
        vel,
        vel_error,
        tkde,
        red,
        mjd,
        snname=snname,
        errorfloor=errorfloor,
        errorscale=errorscale,
        reg_min=reg_min,
        reg_max=reg_max,
        extrapolate=extrapolate,
    )

    vel_int, vel_int_error_lower, vel_int_error_upper, dates = vel_set.data_interp(
        line,
        diagnostic=diag_path,
    )

    expname = os.path.join(
        pa.get_res_path(), "%s_%s_InterpolationResults.csv" % (snname, line)
    )
    ext = 1
    while os.path.exists(expname):
        warning.warn("Results file already exists...")
        expname = os.path.join(
            pa.get_res_path(),
            "%s_%s_InterpolationResults(%d).csv" % (snname, line, ext),
        )
        ext += 1

    expdf = pd.DataFrame(
        {
            "Date": dates,
            "VelInt": vel_int,
            "ErrorLower": vel_int_err_lower,
            "ErrorUpper": vel_int_err_upper,
        }
    )

    expdf.to_csv(expname, index=False)

    return expname


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snname",
        help="Path/ name of the SN in the datadirectory",
    )
    parser.add_argument(
        "line",
        help="Line velocity to be fit",
    )
    parser.add_argument(
        "-r", "--rules", help="File containing velocity interpolation rules"
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()