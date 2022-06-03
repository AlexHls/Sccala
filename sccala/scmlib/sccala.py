import os
import warnings

import numpy as np
import pandas as pd

import stan


C_LIGHT = 299792.458  # c in km/s


# TODO Include this function more sensible
def distmod_kin(z, q0=-0.55, j0=1):
    # Hubble constant free distance modulus d = d_L * H0 in kinematic expansion
    c = 299792.458  # c in km/s
    return (c * z) * (1 + (1 - q0) * z / 2 - (1 - q0 - 3 * q0**2 + j0) * z**2 / 6)


class SCM_Model:
    def __init__():
        self.data = {}
        self.model = None
        self.init = {}
        return

    def get_data():
        return list(self.data.keys())

    def print_data():
        print(self.get_data())
        return

    def print_model():
        print(self.model)
        return

    def set_initial_conditions():
        pass

    def print_results(df):
        for key in list(df.keys()):
            print("%s = %.2e +/- %.2e" % (np.mean(df[key][0]), np.std(df[key][0])))
        return


class HubbleFreeSCM(SCM_Model):
    def __init__():
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "vel_avg": None,
            "col_avg": None,
            "ae_avg": None,
            "log_dist_mod": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                vector[4] obs[sn_idx]; // Observed SN properties
                vector[4] errors[sn_idx]; // Associated uncertaintes (measurement, statistical, systematic)
                real vel_avg; // Normalisation constans
                real col_avg;
                real ae_avg;
                real log_dist_mod[sn_idx]; // Pre-computed, redshift dependent, Hubble-free distance moduli
            }
            parameters {
                real Mi; // Absolute Hubble-free Magnitude
                real alpha; // Velocity correction strength
                real beta; // Color correction strength
                real gamma; // a/e correction strength
                real<lower=-3,upper=0> log_sigma; // Unexplained intrinsic scatter
                real<lower=0> vs; // Mean of latent velocity
                real cs; // Mean of latent color
                real<lower=0> as; // Mean of latent a/e
                real<lower=0> rv; // Dispersion of latent velocity
                real<lower=0> rc; // Dispersion of latent color
                real<lower=0> ra; // Dispersion of latent a/e
                real <lower=0> v_true[sn_idx]; // Modeled latent velocities (cannot be negative)
                real c_true[sn_idx]; // Modeled latent color
                real <lower=0> a_true[sn_idx]; // Modeled latent a/e (cannot be negative)
            }
            transformed parameters{
                real mag_true[sn_idx];
                real sigma_int;
                sigma_int = 10 ^ log_sigma;
                for (i in 1:sn_idx) {
                    mag_true[i] = Mi - alpha * log10(v_true[i] / vel_avg) + beta * (c_true[i] - col_avg) + gamma *             (a_true[i] - ae_avg) + 5 * log_dist_mod[i];
                }
            }
            model {
                Mi ~ uniform(-10,0);
                alpha ~ uniform(-10,10);
                beta ~ uniform(-10,10);
                gamma ~ uniform(-10,10);
                log_sigma ~ uniform(-3,0);

                vs ~ cauchy(7500e3,1500e3);
                cs ~ cauchy(0,0.5);
                as ~ cauchy(0.5,0.5);

                rv ~ normal(0,1500e3);
                rc ~ normal(0,0.05);
                ra ~ normal(0,0.05);

                v_true ~ normal(vs,rv);
                c_true ~ normal(cs,rc);
                a_true ~ normal(as,ra);

                for (i in 1:sn_idx) {
                    target +=  normal_lpdf(obs[i] | [mag_true[i], v_true[i], c_true[i], a_true[i]]', sqrt(errors[i] +          [sigma_int^2, 0, 0, 0]'));
                }
            }
            """
        self.init = {}

        return

    def set_initial_conditions():
        self.init = {
            "vs": [7500e3],
            "rv": [1000e3],
            "v_true": [7500e3] * len(self.data["sn_idxs"]),
        }
        return

    def print_results(df):
        keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        for key in keys:
            print("%s = %.2e +/- %.2e" % (np.mean(df[key][0]), np.std(df[key][0])))
        return


class SccalaSCM:
    def __init__(file, calib="CALIB"):
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

        self.epoch = df[df["dataset"].isin(datasets)]["epoch"].to_numpy()

        if calib:
            self.calib_sn = df[df["dataset"].isin(calbi_datasets)]["SN"].to_numpy()

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

            self.calib_epoch = None

        return

    # TODO various methods to modify (add/ delete SNe) and display loaded data

    def sample(model, log_dir="log_dir", chains=4, iters=1000, quiet=False, init=None):
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
        model.data["sn_idx"] = len(self.sne)
        model.data["obs"] = obs
        model.data["errors"] = errors
        model.data["vel_avg"] = np.mean(self.vel)
        model.data["col_avg"] = np.mean(self.col)
        model.data["ae_avg"] = np.mean(self.ae)
        model.data["log_dist_mod"] = np.log10(distmod_kin(self.red))

        model.set_initial_conditions()

        # Setup/ build STAN model
        fit = stan.build(model.code, data=model.data)
        samples = fit.sample(
            num_chains=chains, num_samples=iters, init=[model.init] * chains
        )

        posterior = samples.to_frame()

        if log_dir is not None:
            self.__save_samples__(posterior, log_dir=log_dir)

        if not quiet:
            model.print_results(posterior)

        return posterior

    def __save_samples__(df, log_dir="log_dir"):
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
