import os
import argparse

import sccala.scmlib.sccala as sc
import sccala.scmlib.models as models


def main(args):
    sccala_scm = sc.SccalaSCM(args.data, calib=args.calib_identifier)

    model = args.model
    if model == "hubble":
        raise ValueError("Model not yet implemented")
    elif model == "hubble-free":
        model = models.HubbleFreeSCM()
    elif model == "hubble-nh":
        raise ValueError("Model not yet implemented")
    elif model == "hubble-free-nh":
        raise ValueError("Model not yet implemented")
    else:
        raise ValueError("Model not regognized")

    posterior = sccala_scm.sample(
        model,
        log_dir=args.log_dir,
        chains=args.chains,
        iters=args.iters,
        quiet=False,
    )

    print("Finished sampling")
    if args.plot is not None:
        print("Saving cornerplot...")
        save = os.path.join(args.log_dir, args.plot)
        sccala_scm.cornerplot(save)

    return posterior


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", help="Path to file containing standardisation data")
    parser.add_argument(
        "model",
        choices=["hubble", "hubble-free", "hubble-nh", "hubble-free-nh"],
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
        "--calib_identifier",
        help="Identifier used for calibrator SNe. Default: None",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
