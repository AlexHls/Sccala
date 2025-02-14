import os
import argparse

import sccala.scmlib.sccala as sc
import sccala.scmlib.models as models


def main(args):
    sccala_scm = sc.SccalaSCM(
        args.data,
        calib=args.calib_identifier,
        blind=args.unblind,
        blindkey=args.blindkey,
    )

    model = args.model
    if not args.classic:
        if model == "hubble":
            model = models.HubbleSCM()
        elif model == "hubble-free":
            model = models.HubbleFreeSCM()
        elif model == "hubble-nh":
            model = models.NHHubbleSCM()
        elif model == "hubble-nh-simple":
            model = models.NHHubbleSCMSimple()
        elif model == "hubble-free-nh":
            model = models.NHHubbleFreeSCM()
        else:
            raise ValueError("Model not regognized")
    else:
        if model == "hubble":
            model = models.ClassicHubbleSCM()
        elif model == "hubble-free":
            model = models.ClassicHubbleFreeSCM()
        elif model == "hubble-nh":
            model = models.ClassicNHHubbleSCM()
        elif model == "hubble-nh-simple":
            model = models.ClassicNHHubbleSCMSimple()
        elif model == "hubble-free-nh":
            model = models.ClassicNHHubbleFreeSCM()
        else:
            raise ValueError("Model not regognized")

    posterior = sccala_scm.sample(
        model,
        log_dir=args.log_dir,
        chains=args.chains,
        iters=args.iters,
        warmup=args.warmup,
        rho=args.rho,
        rho_calib=args.rho_calib,
        save_warmup=args.save_warmup,
        quiet=False,
        classic=args.classic,
        output_dir=args.output_dir,
        test_data=args.test_data,
        selection_effects=args.selection_effects,
    )

    print("Finished sampling")
    if args.plot is not None:
        print("Saving cornerplot...")
        save = os.path.join(args.log_dir, args.plot)
        sccala_scm.cornerplot(save, args.classic)
        sccala_scm.hubble_diagram(
            save=save.replace(".png", "_hubble.png"), classic=args.classic
        )

    return posterior


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", help="Path to file containing standardisation data")
    parser.add_argument(
        "model",
        choices=[
            "hubble",
            "hubble-free",
            "hubble-nh",
            "hubble-nh-simple",
            "hubble-free-nh",
        ],
        help="Model to be fit to the data. Only selects from built-in models",
    )
    parser.add_argument(
        "-c",
        "--chains",
        help="Number of chains used in sampling procedure. Default: 4",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-i",
        "--iters",
        help="Number of interations used in sampling procedure. Default: 1000",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "-w",
        "--warmup",
        help="Number of interations used as warmup in sampling procedure. Default: 1000",
        default=1000,
        type=int,
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
        "-l",
        "--log_dir",
        help="Directory used for storing sampling results. Default: 'log_dir'",
        default="log_dir",
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="Specifies name of cornerplot to be saved to log_dir. Should end with file type.",
    )
    parser.add_argument(
        "--save_warmup",
        action="store_true",
        help="If flag is given, warmup chains will be stored.",
    )
    parser.add_argument(
        "--calib_identifier",
        help="Identifier used for calibrator SNe. Default: None",
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help="If flag is given, classical SCM is used instead of extended SCM",
    )
    parser.add_argument(
        "--unblind",
        action="store_false",
        help="If flag is given, fitted Hubble constant will be displayed and stored as clear text.",
    )
    parser.add_argument(
        "--blindkey",
        default="HUBBLE",
        help="Encryption key used for blinding H0. Default: HUBBLE",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory used for storing STAN temporary files.",
    )
    parser.add_argument(
        "-t",
        "--test_data",
        action="store_true",
        help="If flag is set, the normalisation will be overwritten by the default values of the test data generation script.",
    )
    parser.add_argument(
        "--no_selection_effects",
        action="store_false",
        help="If flag is set, selection effects will be turned off.",
        dest="selection_effects",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
