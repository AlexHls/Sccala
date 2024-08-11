import os

import numpy as np


from sccala.utillib.aux import NumpyEncoder


class SCM_Model:
    def __init__(self):
        self.data = {}
        self.model = None
        self.init = {}
        self.hubble = False  # Specifies if model fits Hubble constant
        return

    def get_data(self):
        return list(self.data.keys())

    def print_data(self):
        print(self.get_data())
        return

    def print_model(self):
        print(self.model)
        return

    def set_initial_conditions(self, init=None):
        pass

    def print_results(self, df, blind=True):
        for key in list(df.keys()):
            print("%s = %.2e +/- %.2e" % (np.mean(df[key][0]), np.std(df[key][0])))
        return

    def write_json(self, filename, path=""):
        try:
            import json
        except ImportError:
            print("json module not available")
            return

        with open(os.path.join(path, filename), "w") as f:
            json.dump(self.data, f, cls=NumpyEncoder)

        return os.path.join(path, filename)

    def write_stan(self, filename, path=""):
        # Check if file exists and if the contents are identical
        # to the current to avoid re-compilation
        if os.path.exists(os.path.join(path, filename)):
            with open(os.path.join(path, filename), "r") as f:
                if f.read() == self.model:
                    print("Model already exists, skipping compilation...")
                    return os.path.join(path, filename)

        with open(os.path.join(path, filename), "w") as f:
            f.write(self.model)
        return os.path.join(path, filename)


class NHHubbleFreeSCM(SCM_Model):
    def __init__(self):
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "mag_sys": None,
            "vel_sys": None,
            "col_sys": None,
            "ae_sys": None,
            "vel_avg": None,
            "col_avg": None,
            "ae_avg": None,
            "log_dist_mod": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                array[sn_idx] vector[4] obs; // Observed SN properties
                array[sn_idx] matrix[4,4] errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[sn_idx] real mag_sys; // Systematic magnitude uncertainties
                array[sn_idx] real vel_sys; // Systematic velocity uncertainties
                array[sn_idx] real col_sys; // Systematic color uncertainties
                array[sn_idx] real ae_sys; // Systematic ae uncertainties
                real vel_avg; // Normalisation constans
                real col_avg;
                real ae_avg;
                array[sn_idx] real log_dist_mod; // Pre-computed, redshift dependent, Hubble-free distance moduli
            }
            parameters {
                real Mi; // Absolute Hubble-free Magnitude
                real alpha; // Velocity correction strength
                real beta; // Color correction strength
                real gamma; // a/e correction strength
                real<lower=-3,upper=0> log_sigma; // Unexplained intrinsic scatter
            }
            transformed parameters{
                array[sn_idx] real mag_true;
                real sigma_int;
                array[sn_idx] real sigma_tot;
                for (i in 1:sn_idx) {
                    sigma_tot[i] = sqrt((errors[i][1,1] + mag_sys[i]) + (alpha / log(10) /obs[i][2])^2 * (errors[i][2,2] + vel_sys[i])+ beta^2 * (errors[i][3,3] + col_sys[i]) + gamma^2 *  (errors[i][4,4] + ae_sys[i]));
                    mag_true[i] = Mi - alpha * log10(obs[i][2] / vel_avg) + beta * (obs[i][3] - col_avg) + gamma * (obs[i][4] - ae_avg) + 5 * log_dist_mod[i];
                }
                sigma_int = 10 ^ log_sigma;
            }
            model {
                Mi ~ uniform(-30,0);
                alpha ~ uniform(-20,20);
                beta ~ uniform(-20,20);
                gamma ~ uniform(-20,20);
                log_sigma ~ uniform(-3,0);

                for (i in 1:sn_idx) {
                    target +=  normal_lpdf(obs[i][1] | mag_true[i], sqrt(sigma_tot[i]^2 + sigma_int^2 ));
                }
            }
            """
        self.init = {}
        self.hubble = False

        return

    def print_results(self, df, blind=True):
        keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        for key in keys:
            print("%s = %.2g +/- %.2g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class NHHubbleSCM(SCM_Model):
    def __init__(self):
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "mag_sys": None,
            "vel_sys": None,
            "col_sys": None,
            "ae_sys": None,
            "vel_avg": None,
            "col_avg": None,
            "ae_avg": None,
            "log_dist_mod": None,
            "calib_sn_idx": None,
            "calib_obs": None,
            "calib_errors": None,
            "calib_mag_sys": None,
            "calib_vel_sys": None,
            "calib_col_sys": None,
            "calib_ae_sys": None,
            "calib_dist_mod": None,
            "calib_dset_idx": None,
            "num_calib_dset": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                array[sn_idx] vector[4] obs; // Observed SN properties
                array[sn_idx] matrix[4,4] errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[sn_idx] real mag_sys; // Systematic magnitude uncertainties
                array[sn_idx] real vel_sys; // Systematic velocity uncertainties
                array[sn_idx] real col_sys; // Systematic color uncertainties
                array[sn_idx] real ae_sys; // Systematic ae uncertainties
                real vel_avg; // Normalisation constants
                real col_avg;
                real ae_avg;
                array[sn_idx] real log_dist_mod;
                int<lower=0> calib_sn_idx;
                array[calib_sn_idx] vector[4] calib_obs; // Observed SN properties
                array[calib_sn_idx] vector[4] calib_errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[calib_sn_idx] real calib_mag_sys; // Systematic magnitude uncertainties
                array[calib_sn_idx] real calib_vel_sys; // Systematic velocity uncertainties
                array[calib_sn_idx] real calib_col_sys; // Systematic color uncertainties
                array[calib_sn_idx] real calib_ae_sys; // Systematic ae uncertainties
                array[calib_sn_idx] real calib_dist_mod; // Distance moduli of calibrators
                array[calib_sn_idx] int<lower=0> calib_dset_idx; // Index of the calibrator dataset
                int<lower=0> num_calib_dset; // Number of calibrator datasets
            }
            parameters {
                real H0; // Hubble constant
                real Mi; // Absolute Hubble-free Magnitude
                real alpha; // Velocity correction strength
                real beta; // Color correction strength
                real gamma; // a/e correction strength
                real<lower=-3,upper=0> log_sigma; // Unexplained intrinsic scatter
                array[num_calib_dset] real<lower=-3,upper=0> calib_log_sigma; // Unexplained calibrator intrinsic scatter
            }
            transformed parameters{
                array[sn_idx] real mag_true;
                real sigma_int;
                array[sn_idx] real sigma_tot;
                array[calib_sn_idx] real calib_mag_true;
                array[num_calib_dset] real calib_sigma_int;
                array[calib_sn_idx] real calib_sigma_tot;
                for (i in 1:sn_idx) {
                    sigma_tot[i] = sqrt((errors[i][1,1] + mag_sys[i]) + (alpha / log(10) / obs[i][2])^2 * (errors[i][2,2] + vel_sys[i])+ beta^2 * (errors[i][3,3] + col_sys[i]) + gamma^2 *  (errors[i][4,4] + ae_sys[i]));
                    mag_true[i] = obs[i][1] + alpha * log10(obs[i][2] / vel_avg) - beta * (obs[i][3] - col_avg) - gamma * (obs[i][4] - ae_avg) - 5 * log_dist_mod[i] + 5 * log10(H0) - 25;
                }
                sigma_int = 10 ^ log_sigma;
                for (i in 1:calib_sn_idx) {
                    calib_sigma_tot[i] = sqrt((calib_errors[i][1] + calib_mag_sys[i]) + (alpha / log(10) / calib_obs[i][2])^2 * (calib_errors[i][2] + calib_vel_sys[i])+ beta^2 * (calib_errors[i][3] + calib_col_sys[i]) + gamma^2 *  (calib_errors[i][4] + calib_ae_sys[i]));
                    calib_mag_true[i] = calib_obs[i][1] + alpha * log10(calib_obs[i][2] / vel_avg) - beta * (calib_obs[i][3] - col_avg) - gamma * (calib_obs[i][4] - ae_avg) - calib_dist_mod[i];
                }
                for (i in 1:num_calib_dset) {
                    calib_sigma_int[i] = 10 ^ calib_log_sigma[i];
                }
            }
            model {
                H0 ~ uniform(0,200);
                Mi ~ uniform(-30,0);
                alpha ~ uniform(-20,20);
                beta ~ uniform(-20,20);
                gamma ~ uniform(-20,20);
                log_sigma ~ uniform(-3,0);
                for (i in 1:num_calib_dset) {
                    calib_log_sigma[i] ~ uniform(-3,0);
                }

                for (i in 1:sn_idx) {
                    target +=  normal_lpdf(mag_true[i] | Mi, sqrt(sigma_tot[i]^2 + sigma_int^2));
                }
                for (i in 1:calib_sn_idx) {
                    target +=  normal_lpdf(calib_mag_true[i] | Mi, sqrt(calib_sigma_tot[i]^2 + calib_sigma_int[calib_dset_idx[i]]^2));
                }
            }
            """
        self.init = {}
        self.hubble = True

        return

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class NHHubbleSCMSimple(SCM_Model):
    def __init__(self):
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "mag_sys": None,
            "vel_sys": None,
            "col_sys": None,
            "ae_sys": None,
            "vel_avg": None,
            "col_avg": None,
            "ae_avg": None,
            "log_dist_mod": None,
            "calib_sn_idx": None,
            "calib_obs": None,
            "calib_errors": None,
            "calib_mag_sys": None,
            "calib_vel_sys": None,
            "calib_col_sys": None,
            "calib_ae_sys": None,
            "calib_dist_mod": None,
            "calib_dset_idx": None,
            "num_calib_dset": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                array[sn_idx] vector[4] obs; // Observed SN properties
                array[sn_idx] matrix[4,4] errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[sn_idx] real mag_sys; // Systematic magnitude uncertainties
                array[sn_idx] real vel_sys; // Systematic velocity uncertainties
                array[sn_idx] real col_sys; // Systematic color uncertainties
                array[sn_idx] real ae_sys; // Systematic ae uncertainties
                real vel_avg; // Normalisation constants
                real col_avg;
                real ae_avg;
                array[sn_idx] real log_dist_mod;
                int<lower=0> calib_sn_idx;
                array[calib_sn_idx] vector[4] calib_obs; // Observed SN properties
                array[calib_sn_idx] vector[4] calib_errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[calib_sn_idx] real calib_mag_sys; // Systematic magnitude uncertainties
                array[calib_sn_idx] real calib_vel_sys; // Systematic velocity uncertainties
                array[calib_sn_idx] real calib_col_sys; // Systematic color uncertainties
                array[calib_sn_idx] real calib_ae_sys; // Systematic ae uncertainties
                array[calib_sn_idx] real calib_dist_mod; // Distance moduli of calibrators
                array[calib_sn_idx] int<lower=0> calib_dset_idx; // Index of the calibrator dataset
                int<lower=0> num_calib_dset; // Number of calibrator datasets
            }
            parameters {
                real H0; // Hubble constant
                real Mi; // Absolute Hubble-free Magnitude
                real alpha; // Velocity correction strength
                real beta; // Color correction strength
                real gamma; // a/e correction strength
                real<lower=-3,upper=0> log_sigma; // Unexplained intrinsic scatter
            }
            transformed parameters{
                array[sn_idx] real mag_true;
                real sigma_int;
                array[sn_idx] real sigma_tot;
                array[calib_sn_idx] real calib_mag_true;
                array[num_calib_dset]real calib_sigma_int;
                array[calib_sn_idx]real calib_sigma_tot;
                for (i in 1:sn_idx) {
                    sigma_tot[i] = sqrt((errors[i][1,1] + mag_sys[i]) + (alpha / log(10) / obs[i][2])^2 * (errors[i][2,2] + vel_sys[i])+ beta^2 * (errors[i][3,3] + col_sys[i]) + gamma^2 *  (errors[i][4,4] + ae_sys[i]));
                    mag_true[i] = obs[i][1] + alpha * log10(obs[i][2] / vel_avg) - beta * (obs[i][3] - col_avg) - gamma * (obs[i][4] - ae_avg) - 5 * log_dist_mod[i] + 5 * log10(H0) - 25;
                }
                sigma_int = 10 ^ log_sigma;
                for (i in 1:calib_sn_idx) {
                    calib_sigma_tot[i] = sqrt((calib_errors[i][1] + calib_mag_sys[i]) + (alpha / log(10) / calib_obs[i][2])^2 * (calib_errors[i][2] + calib_vel_sys[i])+ beta^2 * (calib_errors[i][3] + calib_col_sys[i]) + gamma^2 *  (calib_errors[i][4] + calib_ae_sys[i]));
                    calib_mag_true[i] = calib_obs[i][1] + alpha * log10(calib_obs[i][2] / vel_avg) - beta * (calib_obs[i][3] - col_avg) - gamma * (calib_obs[i][4] - ae_avg) - calib_dist_mod[i];
                }
            }
            model {
                H0 ~ uniform(0,200);
                Mi ~ uniform(-30,0);
                alpha ~ uniform(-20,20);
                beta ~ uniform(-20,20);
                gamma ~ uniform(-20,20);
                log_sigma ~ uniform(-3,0);

                for (i in 1:sn_idx) {
                    target +=  normal_lpdf(mag_true[i] | Mi, sqrt(sigma_tot[i]^2 + sigma_int^2));
                }
                for (i in 1:calib_sn_idx) {
                    target +=  normal_lpdf(calib_mag_true[i] | Mi, sqrt(calib_sigma_tot[i]^2 + sigma_int^2));
                }
            }
            """
        self.init = {}
        self.hubble = True

        return

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class HubbleFreeSCM(SCM_Model):
    def __init__(self):
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "mag_sys": None,
            "vel_sys": None,
            "col_sys": None,
            "ae_sys": None,
            "vel_avg": None,
            "col_avg": None,
            "ae_avg": None,
            "log_dist_mod": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                array[sn_idx] vector[4] obs; // Observed SN properties
                array[sn_idx] matrix[4,4] errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[sn_idx] real mag_sys; // Systematic magnitude uncertainties
                array[sn_idx] real vel_sys; // Systematic velocity uncertainties
                array[sn_idx] real col_sys; // Systematic color uncertainties
                array[sn_idx] real ae_sys; // Systematic ae uncertainties
                real vel_avg; // Normalisation constans
                real col_avg;
                real ae_avg;
                array[sn_idx] real log_dist_mod; // Pre-computed, redshift dependent, Hubble-free distance moduli
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
                array[sn_idx] real <lower=0> v_true; // Modeled latent velocities (cannot be negative)
                array[sn_idx] real c_true; // Modeled latent color
                array[sn_idx] real <lower=0> a_true; // Modeled latent a/e (cannot be negative)
            }
            transformed parameters{
                array[sn_idx] real mag_true;
                real sigma_int;
                sigma_int = 10 ^ log_sigma;
                for (i in 1:sn_idx) {
                    mag_true[i] = Mi - alpha * log10(v_true[i] / vel_avg) + beta * (c_true[i] - col_avg) + gamma * (a_true[i] - ae_avg) + 5 * log_dist_mod[i];
                }
            }
            model {
                Mi ~ uniform(-30,0);
                alpha ~ uniform(-20,20);
                beta ~ uniform(-20,20);
                gamma ~ uniform(-20,20);
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
                    target +=  multi_normal_lpdf(obs[i] | [mag_true[i] + mag_sys[i], v_true[i] + vel_sys[i], c_true[i] + col_sys[i], a_true[i] + ae_sys[i]]', errors[i] +[[sigma_int^2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
                }
            }
            """
        self.init = {}
        self.hubble = False

        return

    def set_initial_conditions(self, init=None):
        if init is None:
            self.init = {
                "vs": 7500e3,
                "rv": 1000e3,
                "v_true": [7500e3] * self.data["sn_idx"],
            }
        else:
            self.init = init
        return

    def print_results(self, df, blind=True):
        keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        for key in keys:
            print("%s = %.2g +/- %.2g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class HubbleSCM(SCM_Model):
    def __init__(self):
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "mag_sys": None,
            "vel_sys": None,
            "col_sys": None,
            "ae_sys": None,
            "vel_avg": None,
            "col_avg": None,
            "ae_avg": None,
            "log_dist_mod": None,
            "calib_sn_idx": None,
            "calib_obs": None,
            "calib_errors": None,
            "calib_mag_sys": None,
            "calib_vel_sys": None,
            "calib_col_sys": None,
            "calib_ae_sys": None,
            "calib_dist_mod": None,
            "calib_dset_idx": None,
            "num_calib_dset": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                array[sn_idx] vector[4] obs; // Observed SN properties
                array[sn_idx] matrix[4,4] errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[sn_idx] real mag_sys; // Systematic magnitude uncertainties
                array[sn_idx] real vel_sys; // Systematic velocity uncertainties
                array[sn_idx] real col_sys; // Systematic color uncertainties
                array[sn_idx] real ae_sys; // Systematic ae uncertainties
                real vel_avg; // Normalisation constans
                real col_avg;
                real ae_avg;
                array[sn_idx] real log_dist_mod; // Pre-computed, redshift dependent, Hubble-free distance moduli
                int<lower=0> calib_sn_idx;
                array[calib_sn_idx] vector[4] calib_obs; // Observed SN properties
                array[calib_sn_idx] vector[4] calib_errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[calib_sn_idx] real calib_mag_sys; // Systematic magnitude uncertainties
                array[calib_sn_idx] real calib_vel_sys; // Systematic velocity uncertainties
                array[calib_sn_idx] real calib_col_sys; // Systematic color uncertainties
                array[calib_sn_idx] real calib_ae_sys; // Systematic ae uncertainties
                array[calib_sn_idx] real calib_dist_mod; // Distance moduli of calibrators
                array[calib_sn_idx] int<lower=0> calib_dset_idx; // Index of the calibrator dataset
                int<lower=0> num_calib_dset; // Number of calibrator datasets
            }
            parameters {
                real H0; // Hubble constant
                real Mi; // Absolute Hubble-free Magnitude
                real alpha; // Velocity correction strength
                real beta; // Color correction strength
                real gamma; // a/e correction strength
                real<lower=-3,upper=0> log_sigma; // Unexplained intrinsic scatter
                array[num_calib_dset] real<lower=-3,upper=0> calib_log_sigma; // Unexplained intrinsic scatter
                real<lower=0> vs; // Mean of latent velocity
                real cs; // Mean of latent color
                real<lower=0> as; // Mean of latent a/e
                real<lower=0> rv; // Dispersion of latent velocity
                real<lower=0> rc; // Dispersion of latent color
                real<lower=0> ra; // Dispersion of latent a/e
                array[num_calib_dset] real<lower=0> calib_vs; // Mean of latent velocity
                array[num_calib_dset] real calib_cs; // Mean of latent color
                array[num_calib_dset] real<lower=0> calib_as; // Mean of latent a/e
                array[num_calib_dset] real<lower=0> calib_rv; // Dispersion of latent velocity
                array[num_calib_dset] real<lower=0> calib_rc; // Dispersion of latent color
                array[num_calib_dset] real<lower=0> calib_ra; // Dispersion of latent a/e
                array[sn_idx] real<lower=0> v_true; // Modeled latent velocities (cannot be negative)
                array[sn_idx] real c_true; // Modeled latent color
                array[sn_idx] real<lower=0> a_true; // Modeled latent a/e (cannot be negative)
                array[calib_sn_idx] real<lower=0> calib_v_true; // Modeled latent velocities (cannot be negative)
                array[calib_sn_idx] real calib_c_true; // Modeled latent color
                array[calib_sn_idx] real<lower=0> calib_a_true; // Modeled latent a/e (cannot be negative)
            }
            transformed parameters{
                array[sn_idx] real mag_true;
                array[calib_sn_idx] real calib_mag_true;
                real sigma_int;
                array[num_calib_dset] real calib_sigma_int;
                sigma_int = 10 ^ log_sigma;
                for (i in 1:num_calib_dset) {
                    calib_sigma_int[i] = 10 ^ calib_log_sigma[i];
                }
                for (i in 1:sn_idx) {
                    mag_true[i] = Mi - 5 * log10(H0) + 25 - alpha * log10(v_true[i] / vel_avg) + beta * (c_true[i] - col_avg) + gamma * (a_true[i] - ae_avg) + 5 * log_dist_mod[i];
                }
                for (i in 1:calib_sn_idx) {
                    calib_mag_true[i] = Mi - alpha * log10(calib_v_true[i] / vel_avg) + beta * (calib_c_true[i] - col_avg) + gamma * (calib_a_true[i] - ae_avg) + calib_dist_mod[i];
                }
            }
            model {
                H0 ~ uniform(0,200);
                Mi ~ uniform(-30,0);
                alpha ~ uniform(-20,20);
                beta ~ uniform(-20,20);
                gamma ~ uniform(-20,20);
                log_sigma ~ uniform(-3,0);
                for (i in 1:num_calib_dset) {
                    calib_log_sigma[i] ~ uniform(-3,0);
                }

                vs ~ cauchy(7500e3,1500e3);
                cs ~ cauchy(0,0.5);
                as ~ cauchy(0.5,0.5);

                rv ~ normal(0,1500e3);
                rc ~ normal(0,0.05);
                ra ~ normal(0,0.05);

                v_true ~ normal(vs,rv);
                c_true ~ normal(cs,rc);
                a_true ~ normal(as,ra);

                for (i in 1:num_calib_dset) {
                    calib_vs[i] ~ cauchy(7500e3,1500e3);
                    calib_cs[i] ~ cauchy(0,0.5);
                    calib_as[i] ~ cauchy(0.5,0.5);

                    calib_rv[i] ~ normal(0,1500e3);
                    calib_rc[i] ~ normal(0,0.05);
                    calib_ra[i] ~ normal(0,0.05);
                }
                
                for (i in 1:calib_sn_idx) {
                    calib_v_true[i] ~ normal(calib_vs[calib_dset_idx[i]],calib_rv[calib_dset_idx[i]]);
                    calib_c_true[i] ~ normal(calib_cs[calib_dset_idx[i]],calib_rc[calib_dset_idx[i]]);
                    calib_a_true[i] ~ normal(calib_as[calib_dset_idx[i]],calib_ra[calib_dset_idx[i]]);
                }

                for (i in 1:sn_idx) {
                    target +=  multi_normal_lpdf(obs[i] | [mag_true[i] + mag_sys[i], v_true[i] + vel_sys[i], c_true[i] + col_sys[i], a_true[i] + ae_sys[i]]', errors[i] + [[sigma_int^2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
                }
                for (i in 1:calib_sn_idx) {
                    target +=  normal_lpdf(calib_obs[i] | [calib_mag_true[i] + calib_mag_sys[i], calib_v_true[i] + calib_vel_sys[i], calib_c_true[i] + calib_col_sys[i], calib_a_true[i] + calib_ae_sys[i]]', sqrt(calib_errors[i] + [calib_sigma_int[calib_dset_idx[i]]^2, 0, 0, 0]'));
                }
            }
            """
        self.init = {}
        self.hubble = True

        return

    def set_initial_conditions(self, init=None):
        if init is None:
            self.init = {
                "vs": 7500e3,
                "rv": 1000e3,
                "v_true": [7500e3] * self.data["sn_idx"],
                "calib_v_true": [7500e3] * self.data["calib_sn_idx"],
            }
            for i in range(self.data["num_calib_dset"]):
                self.init["calib_vs.%d" % (i + 1)] = 7500e3
                self.init["calib_rv.%d" % (i + 1)] = 1000e3
        else:
            self.init = init
        return

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class ClassicNHHubbleFreeSCM(SCM_Model):
    def __init__(self):
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "mag_sys": None,
            "vel_sys": None,
            "col_sys": None,
            "vel_avg": None,
            "col_avg": None,
            "log_dist_mod": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                array[sn_idx] vector[3] obs; // Observed SN properties
                array[sn_idx] matrix[3,3] errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[sn_idx] real mag_sys; // Systematic magnitude uncertainties
                array[sn_idx] real vel_sys; // Systematic velocity uncertainties
                array[sn_idx] real col_sys; // Systematic color uncertainties
                real vel_avg; // Normalisation constans
                real col_avg;
                array[sn_idx] real log_dist_mod; // Pre-computed, redshift dependent, Hubble-free distance moduli
            }
            parameters {
                real Mi; // Absolute Hubble-free Magnitude
                real alpha; // Velocity correction strength
                real beta; // Color correction strength
                real<lower=-3,upper=0> log_sigma; // Unexplained intrinsic scatter
            }
            transformed parameters{
                array[sn_idx] real mag_true;
                real sigma_int;
                array[sn_idx] real sigma_tot;
                for (i in 1:sn_idx) {
                    sigma_tot[i] = sqrt((errors[i][1,1] + mag_sys[i]) + (alpha / log(10) /obs[i][2])^2 * (errors[i][2,2] + vel_sys[i])+ beta^2 * (errors[i][3,3] + col_sys[i]));
                    mag_true[i] = Mi - alpha * log10(obs[i][2] / vel_avg) + beta * (obs[i][3] - col_avg) + 5 * log_dist_mod[i];
                }
                sigma_int = 10 ^ log_sigma;
            }
            model {
                Mi ~ uniform(-30,0);
                alpha ~ uniform(-20,20);
                beta ~ uniform(-20,20);
                log_sigma ~ uniform(-3,0);

                for (i in 1:sn_idx) {
                    target +=  normal_lpdf(obs[i][1] | mag_true[i], sqrt(sigma_tot[i]^2 + sigma_int^2 ));
                }
            }
            """
        self.init = {}
        self.hubble = False

        return

    def print_results(self, df, blind=True):
        keys = ["alpha", "beta", "sigma_int", "Mi"]
        for key in keys:
            print("%s = %.2g +/- %.2g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class ClassicNHHubbleSCM(SCM_Model):
    def __init__(self):
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "mag_sys": None,
            "vel_sys": None,
            "col_sys": None,
            "vel_avg": None,
            "col_avg": None,
            "log_dist_mod": None,
            "calib_sn_idx": None,
            "calib_obs": None,
            "calib_errors": None,
            "calib_mag_sys": None,
            "calib_vel_sys": None,
            "calib_col_sys": None,
            "calib_dist_mod": None,
            "calib_dset_idx": None,
            "num_calib_dset": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                array[sn_idx] vector[3] obs; // Observed SN properties
                array[sn_idx] matrix[3,3] errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[sn_idx] real mag_sys; // Systematic magnitude uncertainties
                array[sn_idx] real vel_sys; // Systematic velocity uncertainties
                array[sn_idx] real col_sys; // Systematic color uncertainties
                real vel_avg; // Normalisation constants
                real col_avg;
                array[sn_idx] real log_dist_mod;
                int<lower=0> calib_sn_idx;
                array[calib_sn_idx] vector[3] calib_obs; // Observed SN properties
                array[calib_sn_idx] vector[3] calib_errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[calib_sn_idx] real calib_mag_sys; // Systematic magnitude uncertainties
                array[calib_sn_idx] real calib_vel_sys; // Systematic velocity uncertainties
                array[calib_sn_idx] real calib_col_sys; // Systematic color uncertainties
                array[calib_sn_idx] real calib_dist_mod; // Distance moduli of calibrators
                array[calib_sn_idx] int<lower=0> calib_dset_idx; // Index of the calibrator dataset
                int<lower=0> num_calib_dset; // Number of calibrator datasets
            }
            parameters {
                real H0; // Hubble constant
                real Mi; // Absolute Hubble-free Magnitude
                real alpha; // Velocity correction strength
                real beta; // Color correction strength
                real<lower=-3,upper=0> log_sigma; // Unexplained intrinsic scatter
                array[num_calib_dset] real<lower=-3,upper=0> calib_log_sigma; // Unexplained intrinsic scatter
            }
            transformed parameters{
                array[sn_idx] real mag_true;
                real sigma_int;
                array[sn_idx] real sigma_tot;
                array[calib_sn_idx] real calib_mag_true;
                array[num_calib_dset] real calib_sigma_int;
                array[calib_sn_idx] real calib_sigma_tot;
                for (i in 1:sn_idx) {
                    sigma_tot[i] = sqrt((errors[i][1,1] + mag_sys[i]) + (alpha / log(10) / obs[i][2])^2 * (errors[i][2,2] + vel_sys[i])+ beta^2 * (errors[i][3,3] + col_sys[i]));
                    mag_true[i] = obs[i][1] + alpha * log10(obs[i][2] / vel_avg) - beta * (obs[i][3] - col_avg) - 5 * log_dist_mod[i] + 5 * log10(H0) - 25;
                }
                sigma_int = 10 ^ log_sigma;
                for (i in 1:calib_sn_idx) {
                    calib_sigma_tot[i] = sqrt((calib_errors[i][1] + calib_mag_sys[i]) + (alpha / log(10) / calib_obs[i][2])^2 * (calib_errors[i][2] + calib_vel_sys[i])+ beta^2 * (calib_errors[i][3] + calib_col_sys[i]));
                    calib_mag_true[i] = calib_obs[i][1] + alpha * log10(calib_obs[i][2] / vel_avg) - beta * (calib_obs[i][3] - col_avg) - calib_dist_mod[i];
                }
                for (i in 1:num_calib_dset) {
                    calib_sigma_int[i] = 10 ^ calib_log_sigma[i];
                }
            }
            model {
                H0 ~ uniform(0,200);
                Mi ~ uniform(-30,0);
                alpha ~ uniform(-20,20);
                beta ~ uniform(-20,20);
                log_sigma ~ uniform(-3,0);
                for (i in 1:num_calib_dset) {
                    calib_log_sigma[i] ~ uniform(-3,0);
                }

                for (i in 1:sn_idx) {
                    target +=  normal_lpdf(mag_true[i] | Mi, sqrt(sigma_tot[i]^2 + sigma_int^2));
                }
                for (i in 1:calib_sn_idx) {
                    target +=  normal_lpdf(calib_mag_true[i] | Mi, sqrt(calib_sigma_tot[i]^2 + calib_sigma_int[calib_dset_idx[i]]^2));
                }
            }
            """
        self.init = {}
        self.hubble = True

        return

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class ClassicNHHubbleSCMSimple(SCM_Model):
    # This model resemples the model used by de Jaeger 2022
    def __init__(self):
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "mag_sys": None,
            "vel_sys": None,
            "col_sys": None,
            "vel_avg": None,
            "col_avg": None,
            "log_dist_mod": None,
            "calib_sn_idx": None,
            "calib_obs": None,
            "calib_errors": None,
            "calib_mag_sys": None,
            "calib_vel_sys": None,
            "calib_col_sys": None,
            "calib_dist_mod": None,
            "calib_dset_idx": None,
            "num_calib_dset": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                array[sn_idx] vector[3] obs; // Observed SN properties
                array[sn_idx] matrix[3,3] errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[sn_idx] real mag_sys; // Systematic magnitude uncertainties
                array[sn_idx] real vel_sys; // Systematic velocity uncertainties
                array[sn_idx] real col_sys; // Systematic color uncertainties
                real vel_avg; // Normalisation constants
                real col_avg;
                array[sn_idx] real log_dist_mod;
                int<lower=0> calib_sn_idx;
                array[calib_sn_idx] vector[3] calib_obs; // Observed SN properties
                array[calib_sn_idx] vector[3] calib_errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[calib_sn_idx] real calib_mag_sys; // Systematic magnitude uncertainties
                array[calib_sn_idx] real calib_vel_sys; // Systematic velocity uncertainties
                array[calib_sn_idx] real calib_col_sys; // Systematic color uncertainties
                array[calib_sn_idx] real calib_dist_mod; // Distance moduli of calibrators
                array[calib_sn_idx] int<lower=0> calib_dset_idx; // Index of the calibrator dataset
                int<lower=0> num_calib_dset; // Number of calibrator datasets
            }
            parameters {
                real H0; // Hubble constant
                real Mi; // Absolute Hubble-free Magnitude
                real alpha; // Velocity correction strength
                real beta; // Color correction strength
                real<lower=-3,upper=0> log_sigma; // Unexplained intrinsic scatter
            }
            transformed parameters{
                array[sn_idx] real mag_true;
                real sigma_int;
                array[sn_idx] real sigma_tot;
                array[calib_sn_idx] real calib_mag_true;
                array[num_calib_dset] real calib_sigma_int;
                array[calib_sn_idx] real calib_sigma_tot;
                for (i in 1:sn_idx) {
                    sigma_tot[i] = sqrt((errors[i][1,1] + mag_sys[i]) + (alpha / log(10) / obs[i][2])^2 * (errors[i][2,2] + vel_sys[i])+ beta^2 * (errors[i][3,3] + col_sys[i]));
                    mag_true[i] = obs[i][1] + alpha * log10(obs[i][2] / vel_avg) - beta * (obs[i][3] - col_avg) - 5 * log_dist_mod[i] + 5 * log10(H0) - 25;
                }
                sigma_int = 10 ^ log_sigma;
                for (i in 1:calib_sn_idx) {
                    calib_sigma_tot[i] = sqrt((calib_errors[i][1] + calib_mag_sys[i]) + (alpha / log(10) / calib_obs[i][2])^2 * (calib_errors[i][2] + calib_vel_sys[i])+ beta^2 * (calib_errors[i][3] + calib_col_sys[i]));
                    calib_mag_true[i] = calib_obs[i][1] + alpha * log10(calib_obs[i][2] / vel_avg) - beta * (calib_obs[i][3] - col_avg) - calib_dist_mod[i];
                }
            }
            model {
                H0 ~ uniform(0,200);
                Mi ~ uniform(-30,0);
                alpha ~ uniform(-20,20);
                beta ~ uniform(-20,20);
                log_sigma ~ uniform(-3,0);

                for (i in 1:sn_idx) {
                    target +=  normal_lpdf(mag_true[i] | Mi, sqrt(sigma_tot[i]^2 + sigma_int^2));
                }
                for (i in 1:calib_sn_idx) {
                    target +=  normal_lpdf(calib_mag_true[i] | Mi, sqrt(calib_sigma_tot[i]^2 + sigma_int^2));
                }
            }
            """
        self.init = {}
        self.hubble = True

        return

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class ClassicHubbleFreeSCM(SCM_Model):
    def __init__(self):
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "mag_sys": None,
            "vel_sys": None,
            "col_sys": None,
            "vel_avg": None,
            "col_avg": None,
            "log_dist_mod": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                array[sn_idx] vector[3] obs; // Observed SN properties
                array[sn_idx] matrix[3,3] errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[sn_idx] real mag_sys; // Systematic magnitude uncertainties
                array[sn_idx] real vel_sys; // Systematic velocity uncertainties
                array[sn_idx] real col_sys; // Systematic color uncertainties
                real vel_avg; // Normalisation constans
                real col_avg;
                array[sn_idx] real log_dist_mod; // Pre-computed, redshift dependent, Hubble-free distance moduli
            }
            parameters {
                real Mi; // Absolute Hubble-free Magnitude
                real alpha; // Velocity correction strength
                real beta; // Color correction strength
                real<lower=-3,upper=0> log_sigma; // Unexplained intrinsic scatter
                real<lower=0> vs; // Mean of latent velocity
                real cs; // Mean of latent color
                real<lower=0> rv; // Dispersion of latent velocity
                real<lower=0> rc; // Dispersion of latent color
                array[sn_idx] real <lower=0> v_true; // Modeled latent velocities (cannot be negative)
                array[sn_idx] real c_true; // Modeled latent color
            }
            transformed parameters{
                array[sn_idx] real mag_true;
                real sigma_int;
                sigma_int = 10 ^ log_sigma;
                for (i in 1:sn_idx) {
                    mag_true[i] = Mi - alpha * log10(v_true[i] / vel_avg) + beta * (c_true[i] - col_avg) + 5 * log_dist_mod[i];
                }
            }
            model {
                Mi ~ uniform(-30,0);
                alpha ~ uniform(-20,20);
                beta ~ uniform(-20,20);
                log_sigma ~ uniform(-3,0);

                vs ~ cauchy(7500e3,1500e3);
                cs ~ cauchy(0,0.5);

                rv ~ normal(0,1500e3);
                rc ~ normal(0,0.05);

                v_true ~ normal(vs,rv);
                c_true ~ normal(cs,rc);

                for (i in 1:sn_idx) {
                    target +=  multi_normal_lpdf(obs[i] | [mag_true[i] + mag_sys[i], v_true[i] + vel_sys[i], c_true[i] + col_sys[i]]', errors[i] + [[sigma_int^2, 0, 0], [0, 0, 0], [0, 0, 0]]);
                }
            }
            """
        self.init = {}
        self.hubble = False

        return

    def set_initial_conditions(self, init=None):
        if init is None:
            self.init = {
                "vs": 7500e3,
                "rv": 1000e3,
                "v_true": [7500e3] * self.data["sn_idx"],
            }
        else:
            self.init = init
        return

    def print_results(self, df, blind=True):
        keys = ["alpha", "beta", "sigma_int", "Mi"]
        for key in keys:
            print("%s = %.2g +/- %.2g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class ClassicHubbleSCM(SCM_Model):
    def __init__(self):
        self.data = {
            "sn_idx": None,
            "obs": None,
            "errors": None,
            "mag_sys": None,
            "vel_sys": None,
            "col_sys": None,
            "vel_avg": None,
            "col_avg": None,
            "log_dist_mod": None,
            "calib_sn_idx": None,
            "calib_obs": None,
            "calib_errors": None,
            "calib_mag_sys": None,
            "calib_vel_sys": None,
            "calib_col_sys": None,
            "calib_dist_mod": None,
            "calib_dset_idx": None,
            "num_calib_dset": None,
        }

        self.model = """
            data {
                int<lower=0> sn_idx;
                array[sn_idx] vector[3] obs; // Observed SN properties
                array[sn_idx] matrix[3,3] errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[sn_idx] real mag_sys; // Systematic magnitude uncertainties
                array[sn_idx] real vel_sys; // Systematic velocity uncertainties
                array[sn_idx] real col_sys; // Systematic color uncertainties
                real vel_avg; // Normalisation constans
                real col_avg;
                array[sn_idx] real log_dist_mod; // Pre-computed, redshift dependent, Hubble-free distance moduli
                int<lower=0> calib_sn_idx;
                array[calib_sn_idx] vector[3] calib_obs; // Observed SN properties
                array[calib_sn_idx] vector[3] calib_errors; // Associated uncertaintes (measurement, statistical, systematic)
                array[calib_sn_idx] real calib_mag_sys; // Systematic magnitude uncertainties
                array[calib_sn_idx] real calib_vel_sys; // Systematic velocity uncertainties
                array[calib_sn_idx] real calib_col_sys; // Systematic color uncertainties
                array[calib_sn_idx] real calib_dist_mod; // Distance moduli of calibrators
                array[calib_sn_idx] int<lower=0> calib_dset_idx; // Index of the calibrator dataset
                int<lower=0> num_calib_dset; // Number of calibrator datasets
            }
            parameters {
                real H0; // Hubble constant
                real Mi; // Absolute Hubble-free Magnitude
                real alpha; // Velocity correction strength
                real beta; // Color correction strength
                real<lower=-3,upper=0> log_sigma; // Unexplained intrinsic scatter
                array[num_calib_dset] real<lower=-3,upper=0> calib_log_sigma; // Unexplained intrinsic scatter
                real<lower=0> vs; // Mean of latent velocity
                real cs; // Mean of latent color
                real<lower=0> rv; // Dispersion of latent velocity
                real<lower=0> rc; // Dispersion of latent color
                array[num_calib_dset] real<lower=0> calib_vs; // Mean of latent velocity
                array[num_calib_dset] real calib_cs; // Mean of latent color
                array[num_calib_dset] real<lower=0> calib_rv; // Dispersion of latent velocity
                array[num_calib_dset] real<lower=0> calib_rc; // Dispersion of latent color
                array[sn_idx] real<lower=0> v_true; // Modeled latent velocities (cannot be negative)
                array[sn_idx] real c_true; // Modeled latent color
                array[calib_sn_idx] real<lower=0> calib_v_true; // Modeled latent velocities (cannot be negative)
                array[calib_sn_idx] real calib_c_true; // Modeled latent color
            }
            transformed parameters{
                array[sn_idx] real mag_true;
                array[calib_sn_idx] real calib_mag_true;
                real sigma_int;
                array[num_calib_dset] real calib_sigma_int;
                sigma_int = 10 ^ log_sigma;
                for (i in 1:num_calib_dset) {
                    calib_sigma_int[i] = 10 ^ calib_log_sigma[i];
                }
                for (i in 1:sn_idx) {
                    mag_true[i] = Mi - 5 * log10(H0) + 25 - alpha * log10(v_true[i] / vel_avg) + beta * (c_true[i] - col_avg) + 5 * log_dist_mod[i];
                }
                for (i in 1:calib_sn_idx) {
                    calib_mag_true[i] = Mi - alpha * log10(calib_v_true[i] / vel_avg) + beta * (calib_c_true[i] - col_avg) + calib_dist_mod[i];
                }
            }
            model {
                H0 ~ uniform(0,200);
                Mi ~ uniform(-30,0);
                alpha ~ uniform(-20,20);
                beta ~ uniform(-20,20);
                log_sigma ~ uniform(-3,0);
                for (i in 1:num_calib_dset) {
                    calib_log_sigma[i] ~ uniform(-3,0);
                }

                vs ~ cauchy(7500e3,1500e3);
                cs ~ cauchy(0,0.5);

                rv ~ normal(0,1500e3);
                rc ~ normal(0,0.05);

                v_true ~ normal(vs,rv);
                c_true ~ normal(cs,rc);

                for (i in 1:num_calib_dset) {
                    calib_vs[i] ~ cauchy(7500e3,1500e3);
                    calib_cs[i] ~ cauchy(0,0.5);

                    calib_rv[i] ~ normal(0,1500e3);
                    calib_rc[i] ~ normal(0,0.05);
                }
                
                for (i in 1:calib_sn_idx) {
                    calib_v_true[i] ~ normal(calib_vs[calib_dset_idx[i]],calib_rv[calib_dset_idx[i]]);
                    calib_c_true[i] ~ normal(calib_cs[calib_dset_idx[i]],calib_rc[calib_dset_idx[i]]);
                }

                calib_v_true ~ normal(vs,rv);
                calib_c_true ~ normal(cs,rc);

                for (i in 1:sn_idx) {
                    target +=  normal_lpdf(obs[i] | [mag_true[i] + mag_sys[i], v_true[i] + vel_sys[i], c_true[i] + col_sys[i]]', errors[i] + [[sigma_int^2, 0, 0], [0, 0, 0], [0, 0, 0]]);
                }
                for (i in 1:calib_sn_idx) {
                    target +=  normal_lpdf(calib_obs[i] | [calib_mag_true[i] + calib_mag_sys[i], calib_v_true[i] + calib_vel_sys[i], calib_c_true[i] + calib_col_sys[i]]', sqrt(calib_errors[i] + [calib_sigma_int[calib_dset_idx[i]]^2, 0, 0]'));
                }
            }
            """
        self.init = {}
        self.hubble = True

        return

    def set_initial_conditions(self, init=None):
        if init is None:
            self.init = {
                "vs": 7500e3,
                "rv": 1000e3,
                "v_true": [7500e3] * self.data["sn_idx"],
                "calib_v_true": [7500e3] * self.data["calib_sn_idx"],
            }
            for i in range(self.data["num_calib_dset"]):
                self.init["calib_vs.%d" % (i + 1)] = 7500e3
                self.init["calib_rv.%d" % (i + 1)] = 1000e3
        else:
            self.init = init
        return

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return
