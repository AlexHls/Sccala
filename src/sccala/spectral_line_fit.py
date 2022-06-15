import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sccala.speclib import linefit as lft
from sccala.speclib import spectools as spc
from sccala.libio import get_paths as pa


def main(args):
    # TODO Make script more flexible/ accept more arguments

    speclist = args.speclist
    """ Expect 'list.txt' file containing data of format
    SN  ID  line  noisefit
    SN : Name of SN or the name of the sub directory within the 'Data' directory
    ID : ID of spectrum. Is used to find filename in infofile & name output files
    line : specifies line to be fitted
    noisefit: specifies if a noisefit is to be done
    """

    try:
        sn, spec_id, line, noisefit = np.genfromtxt(
            speclist, skip_header=1, dtype=str
        ).T
        spec_id = spec_id.astype(int)
    except:
        raise ValueError(
            "Insufficient 'list.txt' supplied, see documentation for more info."
        )

    for i, sid in enumerate(spec_id):
        print("[ %d/%d ]" % (i + 1, len(spec_id)))
        info = pd.read_csv(
            os.path.join(pa.get_data_path(), sn[i], "{:s}_info.csv".format(sn[i]))
        )
        datfile = os.path.join(
            pa.get_data_path(),
            sn[i],
            info[info["ID"] == sid]["File"].to_numpy()[0],
        )
        mjd = info[info["ID"] == sid]["MJD"].to_numpy()[0]

        # Calculate error
        error_file = datfile.replace(".dat", "_error.dat")
        if os.path.exists(error_file):
            _, error = np.genfromtxt(error_file).T
        else:
            print("No error file found, calculating error from scratch...")
            error = spc.calculate_flux_error(datfile)

        wav, flux = np.genfromtxt(datfile).T

        # Fit lines
        diag_path = os.path.join(pa.get_diag_path(), sn[i])

        fit = lft.LineFit(
            wav,
            flux,
            error,
            sid,
        )

        fit.fit_line(
            line[i],
            noisefit=noisefit[i],
            diagnostic=diag_path,
            size=10000,
        )

        peak_loc, peak_error_lower, peak_error_upper = fit.get_results(line[i])

        exp_name = os.path.join(pa.get_res_path(), "%s_PeakFits.csv" % sn[i])

        expdf = pd.DataFrame(
            {
                "MJD": mjd,
                "PeakLoc": peak_loc,
                "PeakErrorLower": peak_error_lower,
                "PeakErrorUpper": peak_error_upper,
            },
            index=[np.array([line[i]]), np.array([sid])],
        )

        if os.path.exists(exp_name):
            data = pd.read_csv(exp_name, index_col=[0, 1])
            if not data.index.isin([(line[i], sid)]).any():
                data = pd.concat([data, expdf])
            else:
                data.loc[(line[i], sid)]["MJD"] = mjd
                data.loc[(line[i], sid)]["PeakLoc"] = peak_loc
                data.loc[(line[i], sid)]["PeakErrorLower"] = peak_error_lower
                data.loc[(line[i], sid)]["PeakErrorUpper"] = peak_error_upper
            data.to_csv(exp_name)
        else:
            expdf.to_csv(exp_name)

    print("Finished fitting spectral lines from %s file!" % speclist)

    return exp_name


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "speclist",
        help="Path to list with spectra and lines to be fitted",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()