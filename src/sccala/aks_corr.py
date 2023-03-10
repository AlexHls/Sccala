import argparse
from matplotlib.pyplot import inspect

import numpy as np

from sccala.asynphot import aks_correction


def main(args):
    filter_in = list(np.genfromtxt(args.filter_in, dtype=str))
    filter_out = list(np.genfromtxt(args.filter_out, dtype=str))

    aks_correction.aks_correction(
        args.snname,
        args.photometry_file,
        filter_in,
        filter_out,
        output=args.output,
        epoch_region=args.epoch_region,
        lsb=args.lsb,
        maxiter=args.maxiter,
        save_plots=(not args.disable_plots),
        save_results=(not args.disable_output),
        delimiter=args.delimiter,
        disable_mean_fit=args.disable_mean_fit,
        inspect_phot_interp=args.inspect_phot_interp,
    )
    return


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "snname", help="Name of the SN for which to do the AKS correction."
    )
    parser.add_argument(
        "photometry_file", help="Path to the file containing photometry data."
    )
    parser.add_argument(
        "filter_in",
        help="Path to a text file containing list of filters to be corrected.",
    )
    parser.add_argument(
        "filter_out", help="Path to a text file containing list of output filters."
    )

    parser.add_argument(
        "-o",
        "--output",
        help="File where corrected photometry of all bands is to be save.",
    )
    parser.add_argument(
        "--epoch_region",
        help="Region form which to take photometry for correction.",
        nargs=2,
        type=float,
    )
    parser.add_argument(
        "--lsb",
        help="Extrapolation rules. Is used to select spectra in photometry range.",
        default=[0.0, 0.0],
        type=float,
    )
    parser.add_argument(
        "--maxiter",
        help="Maximum number of flux correction iterations.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--disable_plots",
        help="If flag is given, diagnostic plots will be disabled.",
        action="store_true",
    )
    parser.add_argument(
        "--disable_output",
        help="If flag is given, results will not be saved.",
        action="store_true",
    )
    parser.add_argument(
        "--disable_mean_fit",
        help="If flag is given, mean fit in AKS interpolation will be disabled.",
        action="store_true",
    )
    parser.add_argument(
        "--delimiter",
        help="Delimiter used in the model files. Needs to be consistent for all files.",
        default=",",
    )
    parser.add_argument(
        "--inspect_phot_interp",
        help="If flag is given, photometry interpolation plots will be shown interactively.",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
