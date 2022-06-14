import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utillib.aux import *


def gen_testdata(zrange, save, size=250, plots=False):
    """
    Function generating simulated datasets for standardisation

    Parameters
    ----------
    zrange : list or tuple
        Lower and upper limit of redshift interval for which
        testdata is to be generated.
    save : str
        Filename under which data will be saved
    size : int
        Number of simulated SNe to generate. Default: 250
    plots : bool
        Specified is diagnostic plots are to be generated.
        WARNING: Will output plot to current working directory.
        Any existing plots with the same names will be overwritten.
        Default: False

    Returns
    -------
    data : pd.DataFrame
        DataFrame containing the simulated testdata
    """

    # Check if zrange is a valid tuple or list
    if not isinstance(zrange, tuple) or not (
        isinstance(zrange, list) and len(zrange) == 2
    ):
        raise ValueError("zrange is not a valid tuple or list of length two")

    red = np.random.triangular(
        left=zrange[0], mode=zrange[1], right=zrange[1], size=size
    )

    if plots:
        plt.hist(red, label="z = %.2f - %.2f" % (zrange[0], zrange[1]))
        plt.gca().set_xscale("log")
        plt.xlabel("Redshift")
        plt.ylabel(r"N$_{SNe}$/bin")
        plt.title("Sample redshift distributions")
        plt.legend()
        plt.savefig(
            "sample_redshift distribution.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    rng = np.random.default_rng()

    # Velocity
    vel = rng.normal(loc=7100e3, scale=725e3, size=size)

    # Color
    col = rng.normal(loc=0.5, scale=0.06, size=size)

    # a/e
    ae = np.absolute(rng.normal(loc=0.31, scale=0.13, size=size))

    if plots:
        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=1, ncols=3, figsize=(3 * 6.4, 4.8), sharey=True
        )
        n_bins = 25
        ax1.hist(vel / 1e3, n_bins, histtype="bar", stacked=True)
        ax2.hist(col, n_bins, histtype="bar", stacked=True)
        ax3.hist(ae, n_bins, histtype="bar", stacked=True)

        ax1.set_ylabel(r"N$_{SNe}$/bin")
        ax1.set_xlabel(r"v$_{\mathrm{H}\beta}$ (km$\,$s$^{-1}$)")
        ax2.set_xlabel("c (mag)")
        ax3.set_xlabel("a/e")

        ax1.set_title("Velocity")
        ax2.set_title("Color")
        ax3.set_title("a/e")

        fig.savefig(
            "sample_properties.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    verr = np.sqrt(200e3**2 + 150e3**2)
    cerr = 0.05
    aeerr = 0.013
    r_err = 0.0001

    c_light = 299792.458

    alpha = 3.4
    beta = 1.8
    gamma = -1.5
    mi = -1.6
    sint = 0.25

    # Calculate magnitudes
    mag_err = (
        (r_err * 5 * (1 + red) / (red * (1 + 0.5 * red) * np.log(10))) ** 2
        + (300 / c_light * 5 * (1 + red) / (red * (1 + 0.5 * red) * np.log(10))) ** 2
        + (0.055 * red) ** 2
    )
    mag = (
        mi
        - alpha * np.log10(vel / np.mean(vel))
        + beta * (col - np.mean(col))
        + gamma * (ae - np.mean(ae))
        + 5 * np.log10(distmod_kin(red))
    )

    # Add noise/ scatter to data
    m_sc = []
    v_sc = []
    c_sc = []
    ae_sc = []
    r_sc = []
    merr_sc = []
    for i in range(len(red)):
        m_sc.append(
            rng.normal(
                loc=mag[i],
                scale=np.sqrt(mag_err[i] + 0.05**2 + sint**2),
                size=1,
            )[0]
        )
        v_sc.append(rng.normal(loc=vel[i], scale=verr, size=1)[0])
        c_sc.append(rng.normal(loc=col[i], scale=cerr, size=1)[0])
        ae_sc.append(rng.normal(loc=ae[i], scale=aeerr, size=1)[0])
        r_sc.append(rng.normal(loc=red[i], scale=r_err, size=1)[0])
        merr_sc.append(0.05)
    m_sc = np.array(m_sc)
    merr_sc = np.array(merr_sc)

    if plots:
        plt.scatter(red, m_sc)

        x = np.linspace(zrange[0], zrange[1], 100)

        plt.plot(
            x, 5 * np.log10(distmod_kin(x)) + mi, color="k", ls="--", label="Cosmology"
        )

        plt.ylabel("Observed magnitudes (mag)")
        plt.xlabel("Redshift")

        plt.legend()

        plt.title("Observed magnitudes of simulated data")

        plt.savefig(
            "sample.svg",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    names = []
    for i in range(size):
        names.append("supernova_%d" % i)

    exp_dict = {
        "SN": names,
        "dataset": ["Testdata"] * len(names),
        "mag": m_sc,
        "mag_err": merr_sc,
        "mag_sys": [0] * len(names),
        "col": c_sc,
        "col_err": merr_sc,
        "col_sys": [0] * len(names),
        "vel": v_sc,
        "vel_err": np.ones_like(v_sc) * np.sqrt(200e3**2 + 150e3**2),
        "vel_sys": [0] * len(names),
        "ae": ae_sc,
        "ae_err": np.ones_like(ae_sc) * aeerr,
        "ae_sys": [0] * len(names),
        "red": r_sc,
        "red_err": np.ones_like(r_sc) * r_err,
        "epoch": [35] * len(names),
    }

    data = pd.DataFrame(exp_dict)

    data.to_csv(save)

    return data


def main(args):
    # Ensure that zrange is sorted
    if args.zrange[0] > args.zrange[1]:
        zrange = [args.zrange[1], args.zrange[0]]
    else:
        zrange = [args.zrange[0], args.zrange[1]]

    data = gen_testdata(zrange, args.save, size=args.size, plots=args.plots)

    return data


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "zrange",
        nargs="2",
        help="Redshift range in which testdata is to be generated."
    )
    parser.add_argument(
        "save",
        help="Name of the saved testdata file",
    )
    parser.add_argument(
        "-s",
        "--size",
        help="Number of SNe to generate. Default: 250",
        type=int,
        default=250,
    )
    parser.add_argument(
        "-p",
        "--plots",
        action="store_true",
        help="Flag to activate plots. WARNING: Will overwrite existing plots."
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
