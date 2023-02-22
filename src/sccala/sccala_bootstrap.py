import argparse

import sccala.scmlib.sccala as sc
import sccala.scmlib.models as models


def main(args):
    sccala_scm = sc.SccalaSCM(
        args.data,
        calib=args.calib_identifier,
    )

    model = args.model
    if not args.classic:
        if model == "hubble":
            model = models.HubbleSCM()
        elif model == "hubble-free":
            raise ValueError("Model not campatible with bootstrap resampling")
        elif model == "hubble-nh":
            model = models.NHHubbleSCM()
        elif model == "hubble-free-nh":
            raise ValueError("Model not campatible with bootstrap resampling")
        else:
            raise ValueError("Model not regognized")
    else:
        if model == "hubble":
            model = models.ClassicHubbleSCM()
        elif model == "hubble-free":
            raise ValueError("Model not campatible with bootstrap resampling")
        elif model == "hubble-nh":
            model = models.ClassicNHHubbleSCM()
        elif model == "hubble-free-nh":
            raise ValueError("Model not campatible with bootstrap resampling")
        else:
            raise ValueError("Model not regognized")

    h0_values = sccala_scm.bootstrap(
        model,
        args.output,
        log_dir=args.log_dir,
        chains=args.chains,
        iters=args.iters,
        warmup=args.warmup,
        save_warmup=args.save_warmup,
        classic=args.classic,
        replacement=args.no_replacement,
        restart=args.disable_restart,
        walltime=args.time,
    )

    print("Finished bootstrap resampling")

    return h0_values


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", help="Path to file containing standardisation data")
    parser.add_argument(
        "model",
        choices=["hubble", "hubble-nh"],
        help="Model to be fit to the data. Only selects from built-in models",
    )
    parser.add_argument(
        "output", help="Name of file where H0 values will be written to"
    )
    parser.add_argument(
        "-c",
        "--chains",
        help="Number of chains used in sampling procedure. Default: 4",
        default=2,
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
        "-l",
        "--log_dir",
        help="Directory used for storing sampling results. Default: 'log_dir'",
        default="log_dir",
    )
    parser.add_argument(
        "-t",
        "--time",
        help="Wallclock time (in h) of parent job. Once 95% of the Wallclock time has passed no new iteration will be started. Default: 24.0",
        default=24.0,
        type=float,
    )
    parser.add_argument(
        "--no_replacement",
        help="If flag is given, bootstrap resampling will be done without replacement",
        action="store_false",
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
        "--disable_restart",
        help="Disables writing or restart file",
        action="store_false",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
