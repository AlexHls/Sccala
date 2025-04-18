import argparse

from sccala.libio import scm_loader as sl


def main(args):
    sne_list = args.sne
    date = args.date
    mag = args.mag
    col = tuple(args.col)
    calib_sne_list = args.calib_sne
    calib_mag = args.calib_mag
    calib_col = tuple(args.calib_mag)
    instrument = args.instrument
    if args.export is None:
        export = False
    elif args.export == "default":
        export = True
    else:
        export = args.export
    mag_sys = args.mag_sys
    vel_sys = args.vel_sys
    col_sys = args.col_sys
    ae_sys = args.ae_sys
    rho = args.rho
    rho_calib = args.rho_calib
    error_mode = args.error_mode
    m_cut_nom = args.m_cut_nom
    sig_cut_nom = args.sig_cut_nom
    pv_red_file = args.pv_red_file

    df = sl.load_data(
        sne_list,
        date,
        mag=mag,
        col=col,
        calib_sne_list=calib_sne_list,
        calib_mag=calib_mag,
        calib_col=calib_col,
        instrument=instrument,
        export=export,
        mag_sys=mag_sys,
        vel_sys=vel_sys,
        col_sys=col_sys,
        ae_sys=ae_sys,
        rho=rho,
        rho_calib=rho_calib,
        error_mode=error_mode,
        m_cut_nom=m_cut_nom,
        sig_cut_nom=sig_cut_nom,
        pv_red_file=pv_red_file,
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
    parser.add_argument(
        "--mag_sys",
        help="Value of the systematic magnitude uncertainty.",
        type=float,
    )
    parser.add_argument(
        "--vel_sys",
        help="Value of the systematic velocity uncertainty.",
        type=float,
    )
    parser.add_argument(
        "--col_sys",
        help="Value of the systematic color uncertainty.",
        type=float,
    )
    parser.add_argument(
        "--ae_sys",
        help="Value of the systematic a/e uncertainty.",
        type=float,
    )
    parser.add_argument(
        "-r",
        "--rho",
        help="Correlation between the color and magnitude uncertainties. Default: 1.0",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--rho_calib",
        help="Correlation between the color and magnitude uncertainties for calibrator SNe. Default: 0.0",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--error_mode",
        help="Mode to calculate asymmetric errors. Default: mean",
        default="mean",
        choices=["mean", "max", "min"],
    )
    parser.add_argument(
        "--m_cut_nom",
        help="Nominal value of the magnitude cut. Used for the full sample.",
        default=18.5,
    )
    parser.add_argument(
        "--sig_cut_nom",
        help="Nominal value of the magnitude cut uncertainty. Used for the full sample.",
        default=0.5,
    )
    parser.add_argument(
        "-p",
        "--pv_red_file",
        help="Path to the file containing redshifts corrected for peculiar velocities.",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
