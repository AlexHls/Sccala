import os
import argparse

from sccala.libio import scm_loader as sl


def main(args):
    sne = args.sne
    date = args.date
    mag = args.mag
    col = tuple(args.col)
    calib_sne = args.calib_sne
    calib_mag = args.calib_mag
    calib_col = tuple(args.calib_mag)
    instrument = args.instrument
    if args.export is None:
        export = False
    elif args.export == "default":
        export = True
    else:
        export = args.export

    df = sl.load_data(
        sne,
        date,
        mag=mag,
        col=col,
        calib_sne=calib_sne,
        calib_mag=calib_mag,
        calib_col=calib_col,
        instrument=instrument,
        export=export,
    )

    return df


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "sne",
        nargs="+",
        help="Which SNe to collect. Can either be a single SN, a list of SNe or the path to a file containing a detailed list.",
    )
    parser.add_argument(
        "date",
        type=float,
        help="Epoch at which data is loaded. Ignored if date column exists in sne file",
    )
    parser.add_argument(
        "-m",
        "--mag",
        default="I",
        help="Filterband used as magnitude. Ignored if mag column exists in sne file",
    )
    parser.add_argument(
        "-c",
        "--col",
        default=["V", "I"],
        nargs=2,
        help="Filterbands used as color. Ignored if col column exists in sne file",
    )
    parser.add_argument(
        "--calib_sne",
        nargs="+",
        help="Which calibrator SNe to collect. Can either be a single SN, a list of SNe or the path to a file containing a detailed list.",
    )
    parser.add_argument(
        "--calib_mag",
        default="I",
        help="Filterband used as magnitude for calibrators. Ignored if mag column exists in sne file",
    )
    parser.add_argument(
        "--calib_col",
        default=["V", "I"],
        nargs=2,
        help="Filterbands used as color for calibrators. Ignored if col column exists in sne file",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        default="Bessell12",
        help="Instrument system of the photometry. Ignored if found in sne file",
    )
    parser.add_argument(
        "-e",
        "--export",
        help="Name of the output file. If 'default' is passed, output name is based on instrument and date and will be saved in the results directory",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
