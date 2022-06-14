import os
import re
import argparse

import numpy as np
import pandas as pd

import sccala.libio.get_paths as pa
import sccala.asynphot.synphot as base
from sccala.speclib import spectools as spc


def get_synthetic_photometry(i, sid, sn, spec_id, filters):
    """
    Helper function calculating synthetic photometry for given SN
    """
    info = pd.read_csv(
        os.path.join(pa.get_data_path(), sn[i], "{:s}_info.csv".format(sn[i]))
    )
    datfile = os.path.join(
        pa.get_data_path(),
        sn[i],
        info[info["ID"] == sid]["File"].to_numpy()[0],
    )
    mjd = info[info["ID"] == sid]["MJD"].to_numpy()[0]

    # Calculate or get error
    error_file = datfile.replace(".dat", "_error.dat")
    if os.path.exists(error_file):
        _, error = np.genfromtxt(error_file).T
    else:
        print("No error file found, calculating error from scratch...")
        error = spc.calculate_flux_error(datfile)

    wav, flux = np.genfromtxt(datfile).T

    mags, mags_err = filters.calculate_vega_magnitudes(wav, flux, spec_err=error)

    return mjd, mags, mags_err


def main(args):
    # TODO Implement check that filters actually exist
    filter_name_list = []

    filter_sets = {
        "instrument": [],
        "filter": [],
    }

    # Check if supplied filter_list is a text file or a list of filter names
    if not isinstance(args.filter_list, list):
        # Check if filter_list refers to file or a single filter
        if os.path.exists(args.filter_list):
            try:
                filter_list = np.genfromtxt(args.filter_list, dtype=str)
            except ValueError:
                raise ValueError("Invalid filter list file")
        else:
            filter_list = [args.filter_list]
    elif len(args.filter_list) == 1 and os.path.exists(args.filter_list[0]):
        try:
            filter_list = np.genfromtxt(args.filter_list[0], dtype=str)
        except ValueError:
            raise ValueError("Invalid filter list file")
    else:
        filter_list = args.filter_list

    for filter_id in filter_list:
        _, instrument, filter_name = re.split("/|\.", filter_id)
        filter_sets["instrument"].append(instrument)
        filter_sets["filter"].append(filter_name)

    instrument_list = list(set(filter_sets["instrument"]))

    filter_set_frame = pd.DataFrame(filter_sets)

    # Load list of spectra
    try:
        sn, spec_id = np.genfromtxt(args.speclist, skip_header=1, dtype=str).T
        spec_id = spec_id.astype(int)
        # TODO Make sure error treatment is consistent
        calib_err = None
        add_err = None
    except ValueError:
        try:
            sn, spec_id, calib_err, add_err = np.genfromtxt(
                args.speclist, skip_header=1, dtype=str
            ).T
            spec_id = spec_id.astype(int)
            calib_err = calib_err.astype(float)
            add_err = add_err.astype(float)
        except:
            raise ValueError("Invalid speclist format")

    filters = base.FilterSet(filter_list)

    for i, sid in enumerate(spec_id):
        print("[ %d/%d ]" % (i + 1, len(spec_id)))

        mjd, mags, mags_err = get_synthetic_photometry(i, sid, sn, spec_id, filters)

        # Add additional uncertainties
        if add_err is not None:
            mags_err = np.sqrt(mags_err**2 + add_err[i] ** 2)
        if calib_err is not None:
            mags_err += calib_err[i]

        # Print magnitudes
        magnitudes = base.MagnitudeSet(
            filter_list, mags, magnitude_uncertainties=mags_err
        )
        print(magnitudes.__repr__)

        # Export data
        for unique_instrument in instrument_list:
            exp_name = os.path.join(
                pa.get_res_path(), "%s_%s_Photometry.csv" % (sn[i], unique_instrument)
            )

            expdict = {
                "MJD": mjd,
            }
            for j, f in enumerate(
                filter_set_frame[filter_set_frame["instrument"] == unique_instrument][
                    "filter"
                ].to_list()
            ):
                expdict[f] = mags[j]
                expdict["%s_err" % f] = mags_err[j]

            expdf = pd.DataFrame(expdict, index=np.array([sid]))

            # Check if results already exist
            if os.path.exists(exp_name):
                data = pd.read_csv(exp_name, index_col=[0])
                if not data.index.isin([sid]).any():
                    data = pd.concat([data, expdf])
                else:
                    data.loc[sid]["MJD"] = mjd
                    for j, f in enumerate(
                        filter_set_frame[
                            filter_set_frame["instrument"] == unique_instrument
                        ]["filter"].to_list()
                    ):
                        data.loc[sid][f] = mags[j]
                        data.loc[sid]["%s_err" % f] = mags_err[j]
                data.to_csv(exp_name)
            else:
                data = pd.DataFrame(expdf)
                data.to_csv(exp_name)

    print(
        "Finished calculating synthetic photometry for spectra from %s file" % args.filter_list
    )

    return


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "speclist",
        help="Path to list with spectra",
    )
    parser.add_argument(
        "filter_list",
        nargs="+",
        help="Filters for which to calculate photometry. Filters match installed filters.",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
