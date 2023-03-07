import os
import argparse
import warnings

import numpy as np
import pandas as pd
import cloudpickle

from sccala.interplib import epoch_interp
from sccala.libio import get_paths as pa


def main(args):
    # TODO Make script more flexible/ accept more arguments
    snname = args.snname
    line = args.line
    rules = args.rules
    no_reject = args.noreject
    no_overwrite = args.nooverwrite

    # Sampling parameter
    size = args.sample_size
    num_live_points = args.num_live_points
    disable_mean_fit = not args.enable_mean_fit
    disable_white_noise_fit = not args.enable_white_noise_fit
    ignore_toe_uncertainty = args.ignore_toe_uncertainty

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
    with open(os.path.join(data_path, "{:s}_TimeKDE.pkl".format(snname)), "rb") as f:
        time_kde = cloudpickle.load(f)
    tkde = time_kde.resample(size=10000)

    # Load redshift from info file
    info = pd.read_csv(os.path.join(data_path, "{:s}_info.csv".format(snname)))
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

    vel_set = epoch_interp.EpochDataSet(
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
        size=size,
        num_live_points=num_live_points,
        disable_mean_fit=disable_mean_fit,
        disable_white_noise_fit=disable_white_noise_fit,
        ignore_toe_uncertainty=ignore_toe_uncertainty,
    )

    vel_int, vel_int_error_lower, vel_int_error_upper, dates = vel_set.data_interp(
        line,
        diagnostic=diag_path,
        no_reject=no_reject,
    )

    expname = os.path.join(
        pa.get_res_path(), "%s_%s_InterpolationResults.csv" % (snname, line)
    )
    if no_overwrite:
        ext = 1
        while os.path.exists(expname):
            warnings.warn("Results file already exists...")
            expname = os.path.join(
                pa.get_res_path(),
                "%s_%s_InterpolationResults(%d).csv" % (snname, line, ext),
            )
            ext += 1

    expdf = pd.DataFrame(
        {
            "Date": np.around(
                dates, 3
            ),  # TODO Find better solution to deal with floating point error
            "VelInt": vel_int,
            "ErrorLower": vel_int_error_lower,
            "ErrorUpper": vel_int_error_upper,
        }
    )

    expdf.to_csv(expname, index=False)

    print("Finished fitting %s velocity fit for %s" % (line, snname))

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
        "-s",
        "--sample_size",
        help="Number of samples drawn from posterior during sample prediction."
        " Default: 1000",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--num_live_points",
        help="Number of live points used during hyperparameter optimization",
        default=800,
        type=int,
    )
    parser.add_argument(
        "-r", "--rules", help="File containing velocity interpolation rules"
    )
    parser.add_argument(
        "--enable_mean_fit",
        help="Enables mean fit in Gaussian Process",
        action="store_true",
    )
    parser.add_argument(
        "--enable_white_noise_fit",
        help="Enables white noise fit in Gaussian Process",
        action="store_true",
    )
    parser.add_argument(
        "--ignore_toe_uncertainty",
        help="Ignores ToE uncertainty during sample prediction",
        action="store_true",
    )
    parser.add_argument(
        "--noreject",
        action="store_true",
        help="When flag is passed, increasing values in the velocity fit will not be rejected.",
    )
    parser.add_argument(
        "--nooverwrite",
        help="Prevents from overwriting existing results.",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
