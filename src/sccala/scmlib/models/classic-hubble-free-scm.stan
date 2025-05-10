data {
    int<lower=0> sn_idx;
    array[sn_idx] vector[3] obs; // Observed SN properties
    array[sn_idx] matrix[3,3] errors; // Associated uncertaintes (measurement, statistical, systematic)
    real vel_avg; // Normalisation constants
    real col_avg;
    array[sn_idx] real log_dist_mod; // Pre-computed, redshift dependent, Hubble-free distance moduli
    real m_cut_nom; // Nominal magnitude cut for selection effect calculation
    real sig_cut_nom; // Nominal uncertainty of the magnitude cut
    int use_selection; // Flag to use selection effect. If 0, the selection effect is not used
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
    real <lower=14, upper=30> mag_cut; // Magnitude cut for selection effect calculation
    real <lower=0.1, upper=3> sigma_cut; // Uncertainty of the magnitude cut
    real <lower = 0, upper = 1> outl_frac; // Fraction of outliers
}
transformed parameters{
    array[sn_idx] real mag_true;
    array[sn_idx] real sn_log_like;
    array[sn_idx] real mean;
    array[sn_idx] real v_mi;
    real sigma_int;
    sigma_int = 10 ^ log_sigma;
    for (i in 1:sn_idx) {
        mag_true[i] = Mi - alpha * log10(v_true[i] / vel_avg) + beta * (c_true[i] - col_avg) + 5 * log_dist_mod[i];
        if (use_selection != 0) {
          mean[i] = Mi - alpha * log10(vs / vel_avg) + beta * (cs - col_avg) + 5 * log_dist_mod[i];
          v_mi[i] = (errors[i][1,1] + sigma_int^2) + sigma_cut^2 + (alpha * rv / (vs * log10()))^2 + (beta * rc)^2;
        }
    }

    for (i in 1:sn_idx) {
        sn_log_like[i] = multi_normal_lpdf(obs[i] | [mag_true[i], v_true[i], c_true[i]]', errors[i] + diag_matrix([sigma_int^2, 0, 0]'));
        if (use_selection != 0) {
          sn_log_like[i] += normal_lcdf(mag_cut | obs[i][1], sigma_cut) 
            - log(normal_cdf(mag_cut | mean[i], sqrt(v_mi[i])) + 0.0001);
        }
    }
}
model {
    Mi ~ uniform(-30,0);
    alpha ~ uniform(-20,20);
    beta ~ uniform(-20,20);
    log_sigma ~ uniform(-3,0);

    vs ~ cauchy(7.5,1.5);
    cs ~ cauchy(0,0.5);

    rv ~ normal(0,1.5);
    rc ~ normal(0,0.5);

    v_true ~ normal(vs,rv);
    c_true ~ normal(cs,rc);

    mag_cut ~ normal(m_cut_nom,0.5);
    sigma_cut ~ normal(sig_cut_nom,0.25);

    outl_frac ~ lognormal(-3,0.25);

    for (i in 1:sn_idx) {
      target += log_sum_exp(
        (log(1 - outl_frac) + sn_log_like[i]),
        (log(outl_frac) + multi_normal_lpdf(obs[i] | [mag_true[i], v_true[i], c_true[i]]', diag_matrix([1, 1, 1]')))
      );
    }
}
