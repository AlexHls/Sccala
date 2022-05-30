import os
import re
import sys
import argparse

import numpy as np
import pandas as pd

import asynphot.synphot as base
from speclib import spectools as spc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "speclist",
        help="Path to list with spectra",
    )
    parser.add_argument(
        "filter_list",
        nargs="+",
        help="Filters for which to calculate photometry. Filters match installed filters."
    )

    args = parser.parse_args()

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
                filter_list = np.genfromtxt(args.filter_list)
            except ValueError:
                raise ValueError("Invalid filter list file")
        else:
            filter_list = [args.filter_list]
    else:
        filter_list = args.filter_list


    for filter_id in filter_list:
        _, instrument, filter_name = re.split('/|\.', filter_id)
        filter_sets["instrument"].append(instrument)
        filter_sets["filter"].append(filter_name)

    instrument_list = list(set(filter_sets["instrument"]))

    # Load list of spectra
    try:
        sn, spec_id = np.genfromtxt(args.speclist, skip_header=1, dtype=str).T
        spec_id = spec_id.astype(int)
        # TODO Make sure error treatment is consistent
        calib_err = None
        add_err = None
    except ValueError:
        try:
            sn, spec_id, calib_err, add_err = np.genfromtxt(args.speclist, skip_header=1, dtype=str).T
            spec_id = spec_id.astype(int)
            calib_err = calib_err.astype(float)
            add_err = add_err.astype(float)
        except:
            raise ValueError("Invalid speclist format")

    filters = base.FilterSet(filterlist)

    for i, sid in enumerate(spec_id):
        print("[ %d/%d ]" % (i+1, len(spec_id)))

        # TODO Photometry calculation

    print("Finished calculating synthetic photometry for spectra from %s file" % speclist)

