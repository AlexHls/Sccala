import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from sccala.utillib.aux import distmod_kin
from sccala.utillib.const import C_LIGHT


def generate_observed_data(
    zrange,
    vel_range=(7100e3, 725e3),
    col_range=(0.5, 0.06),
    ae_range=(0.31, 0.13),
    verr=np.sqrt(200e3**2 + 150e3**2),
    cerr=0.05,
    aeerr=0.013,
    r_err=0.0001,
    alpha=3.4,
    beta=1.8,
    gamma=-1.5,
    mi=-1.6,
    sint=0.25,
    rng=None,
    hubble=False,
    h0=70.0,
):
    if rng is None:
        rng = np.random.default_rng()

    red = rng.triangular(left=zrange[0], mode=zrange[1], right=zrange[1])
    vel = np.absolute(rng.normal(loc=vel_range[0], scale=vel_range[1]))
    col = rng.normal(loc=col_range[0], scale=col_range[1])
    ae = np.absolute(rng.normal(loc=ae_range[0], scale=ae_range[1]))

    vel_obs = rng.normal(loc=vel, scale=verr)
    col_obs = rng.normal(loc=col, scale=cerr)
    ae_obs = rng.normal(loc=ae, scale=aeerr)

    mag = (
        mi
        - alpha * np.log10(vel / vel_range[0])
        + beta * (col - col_range[0])
        + gamma * (ae - ae_range[0])
        + 5 * np.log10(distmod_kin(red))
        + rng.normal(loc=0, scale=sint)
    )
    if hubble:
        mu = rng.uniform(29, 32)
        mag += 5 * np.log10(10 ** ((mu - 25) / 5) * h0)

    mag_err = (
        (r_err * 5 * (1 + red) / (red * (1 + 0.5 * red) * np.log(10))) ** 2
        + (300 / C_LIGHT * 5 * (1 + red) / (red * (1 + 0.5 * red) * np.log(10))) ** 2
        + (0.055 * red) ** 2
        + 0.05**2
    )

    mag_obs = rng.normal(
        loc=mag,
        scale=np.sqrt(mag_err),
    )

    if hubble:
        return red, vel_obs, col_obs, ae_obs, mag_obs, mag_err, mu
    return red, vel_obs, col_obs, ae_obs, mag_obs, mag_err


@np.vectorize
def detection_probability(mag, m_cut=21, sigma_cut=0.5):
    return 1 - norm.cdf(mag, m_cut, sigma_cut)


def gen_testdata(
    zrange,
    save,
    size=250,
    plots=False,
    hubble=False,
    zrange_hubble=(0.001, 0.005),
    hubble_size=25,
    h0=70.0,
    alpha=3.4,
    beta=1.8,
    gamma=-1.5,
    mi=-1.6,
    sint=0.25,
    vel_range=(7100e3, 725e3),
    col_range=(0.5, 0.06),
    ae_range=(0.31, 0.13),
    verr=np.sqrt(200e3**2 + 150e3**2),
    cerr=0.05,
    aeerr=0.013,
    r_err=0.0001,
    m_cut=21,
    sigma_cut=0.5,
    m_cut_nom=None,
    sig_cut_nom=None,
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
    alpha : float
    beta : float
    gamma : float
    mi : float
    sint : float
        Parameters for the calculation of the magnitudes.
    vel_range : tuple
    col_range : tuple
    ae_range : tuple
        Parameters for the generation of the observed data.
    verr : float
    cerr : float
    aeerr : float
    r_err : float
        Parameters for the calculation of the magnitude errors.
    m_cut : float
    sigma_cut : float
        Parameters for the detection probability.
    m_cut_nom : float
    sig_cut_nom : float
        Parameters for the nominal detection values exported to the data.

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

    if m_cut_nom is None:
        m_cut_nom = m_cut
    if sig_cut_nom is None:
        sig_cut_nom = sigma_cut

    rng = np.random.default_rng()
    m_sc, m_sc_rej = [], []
    v_sc, v_sc_rej = [], []
    c_sc, c_sc_rej = [], []
    ae_sc, ae_sc_rej = [], []
    r_sc, r_sc_rej = [], []
    merr_sc, merr_sc_rej = [], []

    while len(m_sc) < size:
        red, vel, col, ae, mag, mag_err = generate_observed_data(
            zrange,
            verr=verr,
            cerr=cerr,
            aeerr=aeerr,
            r_err=r_err,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            mi=mi,
            sint=sint,
            rng=rng,
            hubble=False,
            h0=h0,
        )
        detection_prob = detection_probability(mag, m_cut=m_cut, sigma_cut=sigma_cut)
        detected = rng.uniform() < detection_prob
        if detected:
            m_sc.append(mag)
            v_sc.append(vel)
            c_sc.append(col)
            ae_sc.append(ae)
            r_sc.append(red)
            merr_sc.append(0.05)
        m_sc_rej.append(mag)
        v_sc_rej.append(vel)
        c_sc_rej.append(col)
        ae_sc_rej.append(ae)
        r_sc_rej.append(red)
        merr_sc_rej.append(mag_err)

    m_sc, v_sc, c_sc, ae_sc, r_sc = (
        np.array(m_sc),
        np.array(v_sc),
        np.array(c_sc),
        np.array(ae_sc),
        np.array(r_sc),
    )
    m_sc_rej, v_sc_rej, c_sc_rej, ae_sc_rej, r_sc_rej = (
        np.array(m_sc_rej),
        np.array(v_sc_rej),
        np.array(c_sc_rej),
        np.array(ae_sc_rej),
        np.array(r_sc_rej),
    )

    if plots:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            nrows=2, ncols=3, figsize=(15, 10)
        )
        ax1.hist(
            r_sc_rej, bins=25, histtype="bar", color="r", label="Rejected", alpha=0.5
        )
        ax1.hist(r_sc, bins=25, histtype="bar", color="b", label="Accepted")
        ax1.set_xscale("log")
        ax1.set_xlabel("Redshift")
        ax1.set_ylabel("Number of SNe")
        ax1.legend()
        ax1.set_title("Redshift distribution of simulated data")

        ax2.scatter(r_sc, m_sc)
        x = np.linspace(zrange[0], zrange[1], 100)
        ax2.plot(
            x, 5 * np.log10(distmod_kin(x)) + mi, color="k", ls="--", label="Cosmology"
        )
        ax2.set_ylabel("Observed magnitudes (mag)")
        ax2.set_xlabel("Redshift")
        ax2.legend()
        ax2.set_title("Observed magnitudes of simulated data")

        m_values = np.linspace(np.min(m_sc), np.max(m_sc), 200)
        prob_values = detection_probability(m_values, m_cut, sigma_cut)

        ax3.plot(m_values, prob_values, label="Detection Probability")
        ax3.set_xlabel("Observed Magnitude")
        ax3.set_ylabel("Detection Probability")
        ax3.set_title("Selection Function")
        ax3.legend()

        # Plot vel, col and a/e distributions as histograms
        ax4.scatter(r_sc, v_sc / 1e3, label="Accepted", color="b")
        ax4.scatter(
            r_sc_rej, v_sc_rej / 1e3, label="Rejected", color="r", alpha=0.5, zorder=0
        )
        ax4.axhline(vel_range[0] / 1e3, color="k", ls="--", label="Norm velocity")
        ax4.set_xlabel("Redshift")
        ax4.set_ylabel("Velocity (km/s)")
        ax4.legend()
        ax4.set_title("Velocity distribution of simulated data")

        ax5.scatter(r_sc, c_sc, label="Accepted", color="b")
        ax5.scatter(
            r_sc_rej, c_sc_rej, label="Rejected", color="r", alpha=0.5, zorder=0
        )
        ax5.axhline(col_range[0], color="k", ls="--", label="Norm color")
        ax5.set_xlabel("Redshift")
        ax5.set_ylabel("Color")
        ax5.legend()
        ax5.set_title("Color distribution of simulated data")

        ax6.scatter(r_sc, ae_sc, label="Accepted", color="b")
        ax6.scatter(
            r_sc_rej, ae_sc_rej, label="Rejected", color="r", alpha=0.5, zorder=0
        )
        ax6.axhline(ae_range[0], color="k", ls="--", label="Norm a/e")
        ax6.set_xlabel("Redshift")
        ax6.set_ylabel("a/e")
        ax6.legend()
        ax6.set_title("a/e distribution of simulated data")

        plt.savefig(
            "test_data_overview.png",
            bbox_inches="tight",
        )

        plt.show()

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
        "m_cut_nom": [m_cut_nom] * len(names),
        "sig_cut_nom": [sig_cut_nom] * len(names),
    }

    if hubble:
        hubble_m_sc, hubble_m_sc_rej = [], []
        hubble_v_sc, hubble_v_sc_rej = [], []
        hubble_c_sc, hubble_c_sc_rej = [], []
        hubble_ae_sc, hubble_ae_sc_rej = [], []
        hubble_r_sc, hubble_r_sc_rej = [], []
        hubble_merr_sc, hubble_merr_sc_rej = [], []
        mu, mu_rej = [], []

        while len(hubble_m_sc) < hubble_size:
            red, vel, col, ae, mag, mag_err, mu_val = generate_observed_data(
                zrange_hubble,
                verr=verr,
                cerr=cerr,
                aeerr=aeerr,
                r_err=r_err,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                mi=mi,
                sint=sint,
                rng=rng,
                hubble=True,
                h0=h0,
            )
            detection_prob = detection_probability(
                mag, m_cut=m_cut, sigma_cut=sigma_cut
            )
            detected = rng.uniform() < detection_prob
            if detected:
                hubble_m_sc.append(mag)
                hubble_v_sc.append(vel)
                hubble_c_sc.append(col)
                hubble_ae_sc.append(ae)
                hubble_r_sc.append(red)
                hubble_merr_sc.append(mag_err)
                mu.append(mu_val)
            hubble_m_sc_rej.append(mag)
            hubble_v_sc_rej.append(vel)
            hubble_c_sc_rej.append(col)
            hubble_ae_sc_rej.append(ae)
            hubble_r_sc_rej.append(red)
            hubble_merr_sc_rej.append(mag_err)
            mu_rej.append(mu_val)

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
        exp_dict["mu_err"] = [0.05] * len(exp_dict["mu"])

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
        m_cut=args.m_cut,
        sigma_cut=args.sigma_cut,
        m_cut_nom=args.m_cut_nom,
        sig_cut_nom=args.sig_cut_nom,
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
    parser.add_argument(
        "-m",
        "--m_cut",
        type=float,
        help="Magnitude cut for the detection probability. Default: 23",
        default=21,
    )
    parser.add_argument(
        "-sig",
        "--sigma_cut",
        type=float,
        help="Sigma cut for the detection probability. Default: 0.5",
        default=0.5,
    )
    parser.add_argument(
        "--m_cut_nom",
        type=float,
        help="Nominal magnitude cut for the exported data. Default: 21",
    )
    parser.add_argument(
        "--sig_cut_nom",
        type=float,
        help="Nominal sigma cut for the exported data. Default: 0.5",
    )

    args = parser.parse_args()

    main(args)

    return


if __name__ == "__main__":
    cli()
