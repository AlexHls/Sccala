import os
import warnings

import numpy as np
import pandas as pd
import stan
import matplotlib.pyplot as plt

from sccala.scmlib.models import *
from sccala.utillib.aux import *
from sccala.utillib.const import *


class SccalaSCM:
    def __init__(self, file, calib="CALIB", blind=True, blindkey="HUBBLE"):
        """
        Initializes SccalaSCM object.

        Parameters
        ----------
        file : str
            Path to file containing collected data for SCM.
        calib : str
            Identifier for calibrator sample dataset. All SNe containing
            this identifier in dataset name will be collected as calibrators.
            Default: "CALIB"
        blind : bool
            If True, H0 will not be revealed after the fit and saved in an
            encrypted format based on the 'blindkey'. Default: True
        blindkey : str
            Encryption key used for encrypting the H0 values before
            saving the output file.
        """

        if blind:
            assert (
                blindkey is not None
            ), "For blinding, a blindkey has to be specified..."
        self.blind = blind
        self.blindkey = blindkey

        df = pd.read_csv(file)

        datasets = list(df["dataset"].unique())

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
        self.v_sys = df[df["dataset"].isin(datasets)]["vel_sys"].to_numpy()
        self.c_sys = df[df["dataset"].isin(datasets)]["col_sys"].to_numpy()
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
                "vel_sys"
            ].to_numpy()
            self.calib_c_sys = df[df["dataset"].isin(calib_datasets)][
                "col_sys"
            ].to_numpy()
            self.calib_ae_sys = df[df["dataset"].isin(calib_datasets)][
                "ae_sys"
            ].to_numpy()

            self.calib_dist_mod = df[df["dataset"].isin(calib_datasets)][
                "mu"
            ].to_numpy()
            self.calib_dist_mod_err = df[df["dataset"].isin(calib_datasets)][
                "mu_err"
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

            self.calib_dist_mod = None
            self.calib_dist_mod_err = None

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
        self,
        model,
        log_dir="log_dir",
        chains=4,
        iters=1000,
        warmup=1000,
        save_warmup=False,
        quiet=False,
        init=None,
        classic=False,
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
        warmup : int
            Number of iterations used in STAN fit as warmup. Default: 1000
        save_warmup : bool
            If True, warmup elements of chain will be saved as well. Default: False
        quiet : bool
            Enables/ disables output statements after sampling has
            finished. Default: False
        classic : bool
            Switches classic mode on if True. In classic mode, a/e input is
            ignored.

        Returns
        -------
        posterior : pandas DataFrame
            Result of the STAN sampling
        """

        assert issubclass(
            type(model), SCM_Model
        ), "'model' should be a subclass of SCM_Model"
        assert isinstance(iters, int), "'iters' has to by of type 'int'"
        assert isinstance(warmup, int), "'warmup' has to by of type 'int'"

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

        if not classic:
            # Observed values
            obs = np.array([self.mag, self.vel, self.col, self.ae]).T

            # Redshift, peculiar velocity and gravitational lensing uncertaintes
            errors = np.array(
                [
                    red_uncertainty + self.mag_err**2,
                    self.vel_err**2,
                    self.col_err**2,
                    self.ae_err**2,
                ]
            ).T

            model.data["ae_sys"] = self.ae_sys
            model.data["ae_avg"] = np.mean(self.ae)
        else:
            # Observed values
            obs = np.array([self.mag, self.vel, self.col]).T

            # Redshift, peculiar velocity and gravitational lensing uncertaintes
            errors = np.array(
                [
                    red_uncertainty + self.mag_err**2,
                    self.vel_err**2,
                    self.col_err**2,
                ]
            ).T

        # Fill model data
        model.data["sn_idx"] = len(self.sn)
        model.data["obs"] = obs
        model.data["errors"] = errors
        model.data["mag_sys"] = self.mag_sys
        model.data["vel_sys"] = self.v_sys
        model.data["col_sys"] = self.c_sys
        model.data["vel_avg"] = np.mean(self.vel)
        model.data["col_avg"] = np.mean(self.col)
        model.data["log_dist_mod"] = np.log10(distmod_kin(self.red))

        if model.hubble:
            assert self.calib_sn is not None, "No calibrator SNe found..."

            if not classic:
                # Observed values
                calib_obs = np.array(
                    [self.calib_mag, self.calib_vel, self.calib_col, self.calib_ae]
                ).T

                # Redshift, peculiar velocity and gravitational lensing uncertaintes
                calib_errors = np.array(
                    [
                        self.calib_mag_err**2
                        + self.calib_dist_mod_err**2,
                        self.calib_vel_err**2,
                        self.calib_col_err**2,
                        self.calib_ae_err**2,
                    ]
                ).T

                model.data["calib_ae_sys"] = self.calib_ae_sys
            else:
                # Observed values
                calib_obs = np.array([self.calib_mag, self.calib_vel, self.calib_col]).T

                # Redshift, peculiar velocity and gravitational lensing uncertaintes
                calib_errors = np.array(
                    [
                        self.calib_mag_err**2
                        + self.calib_dist_mod_err**2,
                        self.calib_vel_err**2,
                        self.calib_col_err**2,
                    ]
                ).T
            model.data["calib_sn_idx"] = len(self.calib_sn)
            model.data["calib_obs"] = calib_obs
            model.data["calib_errors"] = calib_errors
            model.data["calib_mag_sys"] = self.calib_mag_sys
            model.data["calib_vel_sys"] = self.calib_v_sys
            model.data["calib_col_sys"] = self.calib_c_sys
            model.data["calib_dist_mod"] = self.calib_dist_mod

        model.set_initial_conditions(init)

        # Setup/ build STAN model
        fit = stan.build(model.model, data=model.data)
        samples = fit.sample(
            num_chains=chains,
            num_samples=iters,
            init=[model.init] * chains,
            num_warmup=warmup,
            save_warmup=save_warmup,
        )

        self.posterior = samples.to_frame()

        # Encrypt H0 for blinding
        if self.blind and model.hubble:
            from itsdangerous import URLSafeSerializer

            norm = np.mean(self.posterior["H0"])
            s = URLSafeSerializer(self.blindkey)
            for i in range(len(self.posterior["H0"])):
                self.posterior["H0"][i] = s.dumps(self.posterior["H0"][i] / norm)
            norm = s.dumps(norm)
        else:
            norm = None

        if log_dir is not None:
            self.__save_samples__(self.posterior, log_dir=log_dir, norm=norm)

        if not quiet:
            model.print_results(self.posterior, blind=self.blind)

        return self.posterior

    def __save_samples__(self, df, log_dir="log_dir", norm=None):
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

        if norm is not None:
            normsave = savename.replace(".csv", ".key")
            np.savetxt(os.path.join(log_dir, normsave), [norm], fmt="%s")

        return os.path.join(log_dir, savename)

    def cornerplot(self, save=None, classic=False):
        """
        Plots the cornerplot of the posterior

        Parameters
        ----------
        save : str
            Specified where the generated cornerplot will be saved.
        classic : bool
            Switches classic mode on if True. In classic mode, a/e input is
            ignored.

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

        if not classic:
            paramnames = [
                r"$\mathcal{M}_I$",
                r"$\alpha$",
                r"$\beta$",
                r"$\gamma$",
                r"$\sigma_{int}$",
            ]
            ndim = len(paramnames)

            # Get relevant parameters
            keys = ["Mi", "alpha", "beta", "gamma", "sigma_int"]
        else:
            paramnames = [
                r"$\mathcal{M}_I$",
                r"$\alpha$",
                r"$\beta$",
                r"$\sigma_{int}$",
            ]
            ndim = len(paramnames)

            # Get relevant parameters
            keys = ["Mi", "alpha", "beta", "sigma_int"]

        posterior = self.posterior[keys].to_numpy()

        figure = corner.corner(
            posterior,
            labels=paramnames,
            show_titles=True,
        )

        # This is the empirical mean of the sample:
        value = np.mean(posterior, axis=0)

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
