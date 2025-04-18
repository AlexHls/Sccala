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
    instrument = args.instrument
    bands = args.bands
    rules = args.rules
    flux_interp = args.fluxinterp
    no_overwrite = args.nooverwrite
    interactive = args.interactive

    # Sampling parameter
    size = args.sample_size
    num_live_points = args.num_live_points
    disable_mean_fit = args.disable_mean_fit
    disable_white_noise_fit = args.disable_white_noise_fit
    ignore_toe_uncertainty = args.ignore_toe_uncertainty

    if not isinstance(bands, list):
        bands = [bands]

    diag_path = os.path.join(pa.get_diag_path(), snname)
    if not os.path.exists(diag_path):
        os.makedirs(diag_path)

    # Load data
    try:
        dataframe = pd.read_csv(
            os.path.join(
                pa.get_data_path(),
                snname,
                "{:s}_{:s}_Photometry.csv".format(snname, instrument),
            ),
            index_col=[0],
        )
        mjd_full = dataframe["MJD"].to_numpy()
    except KeyError:
        # If a key error occurs, assume that there is no index column
        # and try loading it without an index column
        dataframe = pd.read_csv(
            os.path.join(
                pa.get_data_path(),
                snname,
                "{:s}_{:s}_Photometry.csv".format(snname, instrument),
            ),
        )
        mjd_full = dataframe["MJD"].to_numpy()

    # Import Gaussian KDE for time-prior
    data_path = os.path.join(pa.get_data_path(), snname)
    with open(os.path.join(data_path, "{:s}_TimeKDE.pkl".format(snname)), "rb") as f:
        time_kde = cloudpickle.load(f)

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

    for band in bands:
        mag = dataframe[band].to_numpy()
        mag_error = dataframe["{:s}err".format(band)].to_numpy()

        # Check for NaNs
        if np.isnan(mag).any():
            warnings.warn("Input contains NaN, removing invalid values...")
            # Remove NaNs
            mag_error = mag_error[np.logical_not(np.isnan(mag))]
            mjd = mjd_full[np.logical_not(np.isnan(mag))]
            mag = mag[np.logical_not(np.isnan(mag))]
        else:
            mjd = mjd_full

        mag_set = epoch_interp.EpochDataSet(
            mag,
            mag_error,
            time_kde,
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

        mag_int, mag_int_error_lower, mag_int_error_upper, dates = mag_set.data_interp(
            "{:s}_{:s}_phot".format(instrument, band),
            diagnostic=diag_path,
            flux_interp=flux_interp,
            interactive=interactive,
        )

        expname = os.path.join(
            pa.get_res_path(),
            "%s_%s_%s_InterpolationResults.csv" % (snname, instrument, band),
        )
        if no_overwrite:
            ext = 1
            while os.path.exists(expname):
                warnings.warn("Results file already exists...")
                expname = os.path.join(
                    pa.get_res_path(),
                    "%s_%s_%s_InterpolationResults(%d).csv"
                    % (snname, instrument, band, ext),
                )
                ext += 1

        expdf = pd.DataFrame({
            "Date": np.around(
                dates, 3
            ),  # TODO Find better solution to deal with floating point error
            "{:s}".format(band): mag_int,
            "{:s}_err_lower".format(band): mag_int_error_lower,
            "{:s}_err_upper".format(band): mag_int_error_upper,
        })

        expdf.to_csv(expname, index=False)

    return expname


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snname",
        help="Path/ name of the SN in the datadirectory",
    )
    parser.add_argument(
        "instrument",
        help="Instrument name of data to be fit",
    )
    parser.add_argument("bands", nargs="+", help="Photometric band(s) to be fit")
    parser.add_argument(
        "-s",
        "--sample_size",
        help="Number of samples drawn from posterior during sample prediction."
        " Total number of samples is <size>^3.",
        default=50,
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
        "-r", "--rules", help="File containing photometry interpolation rules"
    )
    parser.add_argument(
        "--disable_mean_fit",
        help="Disables mean fit in Gaussian Process",
        action="store_true",
    )
    parser.add_argument(
        "--disable_white_noise_fit",
        help="Disables white noise fit in Gaussian Process",
        action="store_true",
    )
    parser.add_argument(
        "--ignore_toe_uncertainty",
        help="Ignores ToE uncertainty during sample prediction",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--fluxinterp",
        help="Convert mangitudes to flux for interpolation. Default: True",
        action="store_true",
    )
    parser.add_argument(
        "--nooverwrite",
        help="Prevents from overwriting existing results.",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="Show plots interactively",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
