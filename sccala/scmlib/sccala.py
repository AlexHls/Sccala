import os
import warnings

import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt

from models import *

C_LIGHT = 299792.458  # c in km/s


# TODO Include this function more sensible
def distmod_kin(z, q0=-0.55, j0=1):
    # Hubble constant free distance modulus d = d_L * H0 in kinematic expansion
    c = 299792.458  # c in km/s
    return (c * z) * (1 + (1 - q0) * z / 2 - (1 - q0 - 3 * q0**2 + j0) * z**2 / 6)


def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.
    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.
    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.
    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.
    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These
    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.
    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.
    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            print(len(x), len(weights))
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.array(np.interp(q, cdf, x[idx]).tolist())


class SccalaSCM:
    def __init__(self, file, calib="CALIB"):
        """
        Initializes SccalaSCM object.

        Parameters
        ----------
        file : str
            Path to file containing collected data for SCM.
        calib : str
            Identifier for calibrator sample dataset. All SNe containing
            this identifier in dataset name will be collected as calibrators
        """

        df = pd.read_csv(file)

        datasets = list(df["dataset"].unique)

        if calib:
            calib_datasets = [x for x in datasets if calib in x]
            datasets = [x for x in datasets if calib not in x]
        else:
            calib_datasets = []

        self.sn = df[df["dataset"].isin(datasets)]["SN"].to_numpy()

        self.mag = df[df["dataset"].isin(datasets)]["mag"].to_numpy()
        self.mag_err = df[df["dataset"].isin(datasets)]["mag_err"].to_numpy()

        self.col = df[df["dataset"].isin(datasets)]["col"].to_numpy()
        self.col_err = df[df["dataset"].isin(datasets)]["col_err"].to_numpy()

        self.vel = df[df["dataset"].isin(datasets)]["vel"].to_numpy()
        self.vel_err = df[df["dataset"].isin(datasets)]["vel_err"].to_numpy()

        self.ae = df[df["dataset"].isin(datasets)]["ae"].to_numpy()
        self.ae_err = df[df["dataset"].isin(datasets)]["ae_err"].to_numpy()

        self.red = df[df["dataset"].isin(datasets)]["red"].to_numpy()
        self.red_err = df[df["dataset"].isin(datasets)]["red_err"].to_numpy()

        self.mag_sys = df[df["dataset"].isin(datasets)]["mag_sys"].to_numpy()
        self.v_sys = df[df["dataset"].isin(datasets)]["v_sys"].to_numpy()
        self.c_sys = df[df["dataset"].isin(datasets)]["c_sys"].to_numpy()
        self.ae_sys = df[df["dataset"].isin(datasets)]["ae_sys"].to_numpy()

        self.epoch = df[df["dataset"].isin(datasets)]["epoch"].to_numpy()

        if calib:
            self.calib_sn = df[df["dataset"].isin(calib_datasets)]["SN"].to_numpy()

            self.calib_mag = df[df["dataset"].isin(calib_datasets)]["mag"].to_numpy()
            self.calib_mag_err = df[df["dataset"].isin(calib_datasets)][
                "mag_err"
            ].to_numpy()

            self.calib_col = df[df["dataset"].isin(calib_datasets)]["col"].to_numpy()
            self.calib_col_err = df[df["dataset"].isin(calib_datasets)][
                "col_err"
            ].to_numpy()

            self.calib_vel = df[df["dataset"].isin(calib_datasets)]["vel"].to_numpy()
            self.calib_vel_err = df[df["dataset"].isin(calib_datasets)][
                "vel_err"
            ].to_numpy()

            self.calib_ae = df[df["dataset"].isin(calib_datasets)]["ae"].to_numpy()
            self.calib_ae_err = df[df["dataset"].isin(calib_datasets)][
                "ae_err"
            ].to_numpy()

            self.calib_red = df[df["dataset"].isin(calib_datasets)]["red"].to_numpy()
            self.calib_red_err = df[df["dataset"].isin(calib_datasets)][
                "red_err"
            ].to_numpy()

            self.calib_mag_sys = df[df["dataset"].isin(calib_datasets)][
                "mag_sys"
            ].to_numpy()
            self.calib_v_sys = df[df["dataset"].isin(calib_datasets)][
                "v_sys"
            ].to_numpy()
            self.calib_c_sys = df[df["dataset"].isin(calib_datasets)][
                "c_sys"
            ].to_numpy()
            self.calib_ae_sys = df[df["dataset"].isin(calib_datasets)][
                "ae_sys"
            ].to_numpy()

            self.calib_epoch = df[df["dataset"].isin(calib_datasets)][
                "epoch"
            ].to_numpy()
        else:
            self.calib_sn = None

            self.calib_mag = None
            self.calib_mag_err = None

            self.calib_col = None
            self.calib_col_err = None

            self.calib_vel = None
            self.calib_vel_err = None

            self.calib_ae = None
            self.calib_ae_err = None

            self.calib_red = None
            self.calib_red_err = None

            self.calib_mag_sys = None
            self.calib_v_sys = None
            self.calib_c_sys = None
            self.calib_ae_sys = None

            self.calib_epoch = None

        self.datasets = datasets
        if calib:
            self.calib_datasets = calib_datasets
        else:
            self.calib_datasets = None
        self.posterior = None

        return

    # TODO various methods to modify (add/ delete SNe) and display loaded data

    def sample(
        self, model, log_dir="log_dir", chains=4, iters=1000, quiet=False, init=None
    ):
        """
        Samples the posterior for the given data and model

        Parameters
        ----------
        model : SCM_Model
            Model for which to fit the data
        log_dir : str
            Directory in which to save sampling output. If None is passed,
            result will not be saved. Default: log_dir
        chains : int
            Number of chains used in STAN fit. Default: 4
        iters : int
            Number of iterations used in STAN fit. Default: 1000
        quiet : bool
            Enables/ disables output statements after sampling has
            finished. Default: False

        Returns
        -------
        posterior : pandas DataFrame
            Result of the STAN sampling
        """

        assert issubclass(
            type(model), SCM_Model
        ), "'model' should be a subclass of SCM_Model"

        # Observed values
        obs = np.array([self.mag, self.vel, self.col, self.ae]).T

        # Redshift, peculiar velocity and gravitational lensing uncertaintes
        red_uncertainty = (
            (
                self.red_err
                * 5
                * (1 + self.red)
                / (self.red * (1 + 0.5 * self.red) * np.log(10))
            )
            ** 2
            + (
                300
                / C_LIGHT
                * 5
                * (1 + self.red)
                / (self.red * (1 + 0.5 * self.red) * np.log(10))
            )
            ** 2
            + (0.055 * self.red) ** 2
        )
        errors = np.array(
            [
                red_uncertainty + self.mag_err**2,
                self.vel_err**2,
                self.col_err**2,
                self.ae_err**2,
            ]
        ).T

        # Fill model data
        model.data["sn_idx"] = len(self.sn)
        model.data["obs"] = obs
        model.data["errors"] = errors
        model.data["mag_sys"] = self.mag_sys
        model.data["v_sys"] = self.v_sys
        model.data["c_sys"] = self.c_sys
        model.data["ae_sys"] = self.ae_sys
        model.data["vel_avg"] = np.mean(self.vel)
        model.data["col_avg"] = np.mean(self.col)
        model.data["ae_avg"] = np.mean(self.ae)
        model.data["log_dist_mod"] = np.log10(distmod_kin(self.red))

        # TODO Fill calib model data

        model.set_initial_conditions(init)

        # Setup/ build STAN model
        fit = stan.build(model.code, data=model.data)
        samples = fit.sample(
            num_chains=chains, num_samples=iters, init=[model.init] * chains
        )

        self.posterior = samples.to_frame()

        if log_dir is not None:
            self.__save_samples__(self.posterior, log_dir=log_dir)

        if not quiet:
            model.print_results(self.posterior)

        return self.posterior

    def __save_samples__(self, df, log_dir="log_dir"):
        """Exports sample data"""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        savename = "chains_1.csv"
        if os.path.exists(os.path.join(log_dir, savename)):
            i = 1
            while os.path.exists(
                os.path.join(log_dir, savename.replace("1", str(i + 1)))
            ):
                i += 1
            savename = savename.replace("1", str(i + 1))

        df.to_csv(os.path.join(log_dir, savename))

        return os.path.join(log_dir, savename)

    def cornerplot(self, save=None):
        """
        Plots the cornerplot of the posterior

        Parameters
        ----------
        save : str
            Specified where the generated cornerplot will be saved.

        Returns
        -------
        None
        """

        if self.posterior is None:
            warnings.warn("Please run sampling before generating cornerplots")
            return

        try:
            import corner
        except ImportError:
            warnings.warn("corner package not installed, skipping...")
            return

        paramnames = [
            r"$\mathcal{M}_I$",
            r"$\alpha$",
            r"$\beta$",
            r"$\gamma$",
            r"$\sigma_{int}$",
        ]
        ndim = len(paramnames)

        figure = corner.corner(
            self.posterior[:, 8 : 8 + ndim],
            labels=paramnames,
            show_titles=True,
        )

        # This is the empirical mean of the sample:
        value = np.mean(self.posterior[:, 8 : 8 + ndim], axis=0)

        # Extract the axes
        axes = np.array(figure.axes).reshape((ndim, ndim))

        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(value[i], color="g")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(value[xi], color="g")
                ax.axhline(value[yi], color="g")
                ax.plot(value[xi], value[yi], "sg")

        if isinstance(save, str):
            plt.savefig(
                save,
                bbox_inches="tight",
                dpi=300,
            )

        plt.close()

        return
