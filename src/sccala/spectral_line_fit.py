import os
import argparse
import warnings

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

    # Edge case if only one item is passed
    if isinstance(spec_id, np.int64):
        sn = np.array([sn])
        spec_id = np.array([spec_id])
        line = np.array([line])
        noisefit = np.array([noisefit])

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

        if args.cr_low is not None:
            fit.__modify_builtin_lines__(line[i], cr_low=args.cr_low)
        if args.cr_high is not None:
            fit.__modify_builtin_lines__(line[i], cr_high=args.cr_high)
        if args.rest is not None:
            fit.__modify_builtin_lines__(line[i], rest=args.rest)
        if args.ae_feature is not None:
            if args.ae_feature == "True":
                ae_feature = True
            elif args.ae_feature == "False":
                ae_feature = False
            else:
                raise ValueError("Unrecognized ae_feature specification")
            fit.__modify_builtin_lines__(line[i], ae_feature=ae_feature)

        # For noisefit, use HODLR solver, otherwise use default
        if noisefit[i] == "True":
            if args.disable_hodlrsolver:
                hodlrsolver = False
            else:
                hodlrsolver = True
            nf = True
        elif noisefit[i] == "False":
            hodlrsolver = False
            nf = False
        else:
            raise ValueError(
                "'noisefit' should be 'True' or 'False', but it is %s" % noisefit[i]
            )

        print("ID: %s" % str(sid))
        print("Noisefit: ", nf)
        print("HODLR solver: ", hodlrsolver)
        fit.fit_line(
            line[i],
            noisefit=nf,
            diagnostic=diag_path,
            size=args.sample_size,
            hodlrsolver=hodlrsolver,
            num_live_points=args.num_live_points,
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
                data.loc["MJD", (line[i], sid)] = mjd
                data.loc["PeakLoc", (line[i], sid)] = peak_loc
                data.loc["PeakErrorLower", (line[i], sid)] = peak_error_lower
                data.loc["PeakErrorUpper", (line[i], sid)] = peak_error_upper
            data.to_csv(exp_name)
        else:
            expdf.to_csv(exp_name)

    print("Finished fitting spectral lines from %s file!" % speclist)

    try:
        return exp_name
    except UnboundLocalError:
        warnings.warn("No features fitted, no output file was created...")
        return None


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "speclist",
        help="Path to list with spectra and lines to be fitted",
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
        "--cr_low",
        help="Changes the lower boundary of the lines to be fitted",
        type=float,
    )
    parser.add_argument(
        "--cr_high",
        help="Changes the upper boundary of the lines to be fitted",
        type=float,
    )
    parser.add_argument(
        "--rest",
        help="Changes the rest wavelength of the lines to be fitted",
        type=float,
    )
    parser.add_argument(
        "--ae_feature",
        help="Changes if the line is to be fitted as an ae feature",
    )
    parser.add_argument(
        "--disable_hodlrsolver",
        help="Disables HODLRSolver regardless of noisefit.",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
