import numpy as np


class SCM_Model:
    def __init__(self):
        self.data = {}
        self.model = None
        self.init = {}
        return

    def get_data(self):
        return list(self.data.keys())

    def print_data(self):
        print(self.get_data())
        return

    def print_model(self):
        print(self.model)
        return

    def set_initial_conditions(self):
        pass

    def print_results(self, df):
        for key in list(df.keys()):
            print("%s = %.2e +/- %.2e" % (np.mean(df[key][0]), np.std(df[key][0])))
        return


class HubbleFreeSCM(SCM_Model):
    def __init__(self):
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

    def set_initial_conditions(self):
        self.init = {
            "vs": [7500e3],
            "rv": [1000e3],
            "v_true": [7500e3] * len(self.data["sn_idxs"]),
        }
        return

    def print_results(self, df):
        keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        for key in keys:
            print("%s = %.2e +/- %.2e" % (np.mean(df[key][0]), np.std(df[key][0])))
        return
