data {
    int<lower=0> sn_idx;
    array[sn_idx] vector[4] obs; // Observed SN properties
    array[sn_idx] matrix[4,4] errors; // Associated uncertaintes (measurement, statistical, systematic)
    real vel_avg; // Normalisation constants
    real col_avg;
    real ae_avg;
    array[sn_idx] real log_dist_mod;
    int<lower=0> calib_sn_idx;
    array[calib_sn_idx] vector[4] calib_obs; // Observed SN properties
    array[calib_sn_idx] vector[4] calib_errors; // Associated uncertaintes (measurement, statistical, systematic)
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
        sigma_tot[i] = sqrt(errors[i][1,1] + (alpha / log(10) / obs[i][2])^2 * errors[i][2,2] + beta^2 * errors[i][3,3] + gamma^2 * errors[i][4,4]);
        mag_true[i] = obs[i][1] + alpha * log10(obs[i][2] / vel_avg) - beta * (obs[i][3] - col_avg) - gamma * (obs[i][4] - ae_avg) - 5 * log_dist_mod[i] + 5 * log10(H0) - 25;
    }
    sigma_int = 10 ^ log_sigma;
    for (i in 1:calib_sn_idx) {
        calib_sigma_tot[i] = sqrt(calib_errors[i][1] + (alpha / log(10) / calib_obs[i][2])^2 * calib_errors[i][2] + beta^2 * calib_errors[i][3] + gamma^2 *  calib_errors[i][4]);
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
