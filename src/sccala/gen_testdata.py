import argparse
from matplotlib import scale

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sccala.utillib.aux import *


def gen_testdata(
    zrange,
    save,
    size=250,
    plots=False,
    hubble=False,
    zrange_hubble=(0.001, 0.005),
    hubble_size=25,
    h0=70.0,
):
    """
    Function generating simulated datasets for standardisation

    Parameters
    ----------
    zrange : list or tuple
        Lower and upper limit of redshift interval for which
        testdata is to be generated
    save : str
        Filename under which data will be saved
    size : int
        Number of simulated SNe to generate. Default: 250
    plots : bool
        Specified is diagnostic plots are to be generated.
        WARNING: Will output plot to current working directory.
        Any existing plots with the same names will be overwritten.
        Default: False
    hubble : bool
        If True, a calibrator sample will be generated as well. Default: False
    zrange_tuple : list or tuple
        Lower and upper limit of redshift interval for which
        calibrator testdata is to be generated. Default: (0.001, 0.005)
    hubble_size : int
        Number of simulated calibrator SNe. Default: 25
    h0 : float
        Value of the Hubble constant used for simulating calibrator sample.
        Default: 70.0

    Returns
    -------
    data : pd.DataFrame
        DataFrame containing the simulated testdata
    """

    # Check if zrange is a valid tuple or list
    if not isinstance(zrange, tuple) and not (
        isinstance(zrange, list) and len(zrange) == 2
    ):
        raise ValueError("zrange is not a valid tuple or list of length two")

    red = np.random.triangular(
        left=zrange[0], mode=zrange[1], right=zrange[1], size=size
    )
    if hubble:
        hubble_red = np.random.triangular(
            left=zrange_hubble[0],
            mode=zrange_hubble[1],
            right=zrange_hubble[1],
            size=hubble_size,
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

    if hubble:
        # Velocity
        hubble_vel = rng.normal(loc=7100e3, scale=725e3, size=hubble_size)
        # Color
        hubble_col = rng.normal(loc=0.5, scale=0.06, size=hubble_size)
        # a/e
        hubble_ae = np.absolute(rng.normal(loc=0.31, scale=0.13, size=hubble_size))
        # Distance modulus
        mu = rng.uniform(29, 32, size=hubble_size)

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
    dl = 10 ** ((mu - 25) / 5)
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
    if hubble:
        hubble_mag_err = (
            (r_err * 5 * (1 + red) / (red * (1 + 0.5 * red) * np.log(10))) ** 2
            + (300 / c_light * 5 * (1 + red) / (red * (1 + 0.5 * red) * np.log(10)))
            ** 2
            + (0.055 * red) ** 2
        )
        hubble_mag = (
            mi
            - alpha * np.log10(hubble_vel / np.mean(hubble_vel))
            + beta * (hubble_col - np.mean(hubble_col))
            + gamma * (hubble_ae - np.mean(hubble_ae))
            + 5 * np.log10(h0 * dl)
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

    if hubble:
        hubble_m_sc = []
        hubble_v_sc = []
        hubble_c_sc = []
        hubble_ae_sc = []
        hubble_r_sc = []
        hubble_merr_sc = []
        hubble_mu_sc = []
        for i in range(len(hubble_red)):
            hubble_m_sc.append(
                rng.normal(
                    loc=hubble_mag[i],
                    scale=np.sqrt(hubble_mag_err[i] + 0.05**2 + sint**2),
                    size=1,
                )[0]
            )
            hubble_v_sc.append(rng.normal(loc=hubble_vel[i], scale=verr, size=1)[0])
            hubble_c_sc.append(rng.normal(loc=hubble_col[i], scale=cerr, size=1)[0])
            hubble_ae_sc.append(rng.normal(loc=hubble_ae[i], scale=aeerr, size=1)[0])
            hubble_r_sc.append(rng.normal(loc=hubble_red[i], scale=r_err, size=1)[0])
            hubble_mu_sc.append(rng.normal(loc=mu[i], scale=0.1, size=1)[0])
            hubble_merr_sc.append(0.05)
        hubble_m_sc = np.array(hubble_m_sc)
        hubble_merr_sc = np.array(hubble_merr_sc)

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
        "mag": list(m_sc),
        "mag_err": list(merr_sc),
        "mag_sys": [0] * len(names),
        "col": list(c_sc),
        "col_err": list(merr_sc),
        "col_sys": [0] * len(names),
        "vel": list(v_sc),
        "vel_err": list(np.ones_like(v_sc) * np.sqrt(200e3**2 + 150e3**2)),
        "vel_sys": [0] * len(names),
        "ae": list(ae_sc),
        "ae_err": list(np.ones_like(ae_sc) * aeerr),
        "ae_sys": [0] * len(names),
        "red": list(r_sc),
        "red_err": list(np.ones_like(r_sc) * r_err),
        "epoch": [35] * len(names),
    }

    if hubble:
        exp_dict["mu"] = [0] * len(names)  # Dummy values for non-calibrators

        hubble_names = []
        for i in range(hubble_size):
            hubble_names.append("calib_supernova_%d" % i)

        exp_dict["SN"].extend(hubble_names)
        exp_dict["dataset"].extend(["CALIB_Testdata"] * len(hubble_names))
        exp_dict["mag"].extend(hubble_m_sc)
        exp_dict["mag_err"].extend(hubble_merr_sc)
        exp_dict["mag_sys"].extend([0] * len(hubble_names))
        exp_dict["col"].extend(hubble_c_sc)
        exp_dict["col_err"].extend(hubble_merr_sc)
        exp_dict["col_sys"].extend([0] * len(hubble_names))
        exp_dict["vel"].extend(hubble_v_sc)
        exp_dict["vel_err"].extend(
            np.ones_like(hubble_v_sc) * np.sqrt(200e3**2 + 150e3**2)
        )
        exp_dict["vel_sys"].extend([0] * len(hubble_names))
        exp_dict["ae"].extend(hubble_ae_sc)
        exp_dict["ae_err"].extend(np.ones_like(hubble_ae_sc) * aeerr)
        exp_dict["ae_sys"].extend([0] * len(hubble_names))
        exp_dict["red"].extend(hubble_r_sc)
        exp_dict["red_err"].extend(np.ones_like(hubble_r_sc) * r_err)
        exp_dict["epoch"].extend([35] * len(hubble_names))

        exp_dict["mu"].extend(mu)

    data = pd.DataFrame(exp_dict)

    data.to_csv(save)

    return data


def main(args):
    # Ensure that zrange is sorted
    if args.zrange[0] > args.zrange[1]:
        zrange = [args.zrange[1], args.zrange[0]]
    else:
        zrange = [args.zrange[0], args.zrange[1]]
    if args.zrange_hubble[0] > args.zrange_hubble[1]:
        zrange_hubble = [args.zrange_hubble[1], args.zrange_hubble[0]]
    else:
        zrange_hubble = [args.zrange_hubble[0], args.zrange_hubble[1]]

    data = gen_testdata(
        zrange,
        args.save,
        size=args.size,
        plots=args.plots,
        hubble=args.hubble,
        zrange_hubble=zrange_hubble,
        hubble_size=args.hubble_size,
        h0=args.h0,
    )

    return data


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "zrange",
        nargs=2,
        help="Redshift range in which testdata is to be generated.",
        type=float,
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
        help="Flag to activate plots. WARNING: Will overwrite existing plots.",
    )
    parser.add_argument(
        "--hubble",
        action="store_true",
        help="If flag is given, a calibrator sample will be generated as well. Default: False",
    )
    parser.add_argument(
        "--zrange_hubble",
        nargs=2,
        type=float,
        help="Redshift range in which the calibrator testdata is to be generated. Defaul: [0.001, 0.005]",
        default=[0.001, 0.005],
    )
    parser.add_argument(
        "--hubble_size",
        type=int,
        help="Number of calibrator SNe to generate. Default: 25",
        default=25,
    )
    parser.add_argument(
        "--h0",
        type=float,
        help="Value of the Hubble constant for which to generate testdata. Default: 70.0",
        default=70.0,
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
