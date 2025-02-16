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
        sigma_tot[i] = sqrt((errors[i][1,1] + mag_sys[i]) + (alpha / log(10) /obs[i][2])^2 * (errors[i][2,2] + vel_sys[i])+ beta^2 * (errors[i][3,3] + col_sys[i]);
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
