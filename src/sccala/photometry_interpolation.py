import os
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
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

    if not isinstance(bands, list):
        bands = [bands]

    diag_path = os.path.join(pa.get_diag_path(), snname)
    if not os.path.exists(diag_path):
        os.makedirs(diag_path)

    # Load data
    dataframe = pd.read_csv(
        os.path.join(
            pa.get_res_path(), "{:s}_{:s}_Photometry.csv".format(snname, instrument)
        ),
        index_col=[0],
    )
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

    for band in bands:
        mag = dataframe[band].to_numpy()
        mag_error = dataframe["%s_err" % band].to_numpy()

        mag_set = epoch_interp.EpochDataSet(
            mag,
            mag_error,
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

        mag_int, mag_int_error_lower, mag_int_error_upper, dates = mag_set.data_interp(
            "{:s}_{:s}_phot".format(instrument, band),
            diagnostic=diag_path,
        )

        expname = os.path.join(
            pa.get_res_path(),
            "%s_%s_%s_InterpolationResults.csv" % (snname, instrument, band),
        )
        ext = 1
        while os.path.exists(expname):
            warnings.warn("Results file already exists...")
            expname = os.path.join(
                pa.get_res_path(),
                "%s_%s_%s_InterpolationResults(%d).csv"
                % (snname, instrument, band, ext),
            )
            ext += 1

        expdf = pd.DataFrame(
            {
                "Date": dates,
                "{:s}".format(band): mag_int,
                "{:s}_err_lower".format(band): mag_int_error_lower,
                "{:s}_err_upper".format(band): mag_int_error_upper,
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
        "instrument",
        help="Instrument name of data to be fit",
    )
    parser.add_argument("bands", nargs="+", help="Photometric band(s) to be fit")
    parser.add_argument(
        "-r", "--rules", help="File containing velocity interpolation rules"
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
