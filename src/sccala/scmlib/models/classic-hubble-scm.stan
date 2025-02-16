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
    real m_cut_nom; // Nominal magnitude cut for selection effect calculation
    real sig_cut_nom; // Nominal uncertainty of the magnitude cut
    array[num_calib_dset] real calib_m_cut_nom; // Nominal magnitude cut for selection effect calculation
    array[num_calib_dset] real calib_sig_cut_nom; // Nominal uncertainty of the magnitude cut
    int use_selection; // Flag to use selection effect. If 0, the selection effect is not used
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
    real <lower=14, upper=30> mag_cut; // Magnitude cut for selection effect calculation
    real <lower=0.1, upper=3> sigma_cut; // Uncertainty of the magnitude cut
    array[num_calib_dset] real <lower=14, upper=30> calib_mag_cut; // Magnitude cut for selection effect calculation
    array[num_calib_dset] real <lower=0.1, upper=3> calib_sigma_cut; // Uncertainty of the magnitude cut
}
transformed parameters{
    array[sn_idx] real mag_true;
    array[sn_idx] real mean;
    array[sn_idx] real v_mi;
    array[calib_sn_idx] real calib_mag_true;
    array[calib_sn_idx] real calib_mean;
    array[calib_sn_idx] real calib_v_mi;
    real sigma_int;
    array[num_calib_dset] real calib_sigma_int;
    sigma_int = 10 ^ log_sigma;
    for (i in 1:num_calib_dset) {
        calib_sigma_int[i] = 10 ^ calib_log_sigma[i];
    }
    for (i in 1:sn_idx) {
        mag_true[i] = Mi - 5 * log10(H0) + 25 - alpha * log10(v_true[i] / vel_avg) + beta * (c_true[i] - col_avg) + 5 * log_dist_mod[i];
        if (use_selection != 0) {
          mean[i] = Mi - alpha * log10(vs / vel_avg) + beta * (cs - col_avg) + 5 * log_dist_mod[i];
          v_mi[i] = (errors[i][1,1] + sigma_int^2) + sigma_cut^2 + (alpha * rv / (vs * log10()))^2 + (beta * rc)^2;
        }
    }
    for (i in 1:calib_sn_idx) {
        calib_mag_true[i] = Mi - alpha * log10(calib_v_true[i] / vel_avg) + beta * (calib_c_true[i] - col_avg) + calib_dist_mod[i];
        if (use_selection != 0) {
          calib_mean[i] = Mi - alpha * log10(calib_vs[calib_dset_idx[i]] / vel_avg) + beta * (calib_cs[calib_dset_idx[i]] - col_avg) + calib_dist_mod[i];
          calib_v_mi[i] = (calib_errors[i][1,1] + calib_sigma_int[calib_dset_idx[i]]^2) + calib_sigma_cut[calib_dset_idx[i]]^2 + (alpha * calib_rv[calib_dset_idx[i]] / (calib_vs[calib_dset_idx[i]] * log10()))^2 + (beta * calib_rc[calib_dset_idx[i]])^2;
        }
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

    mag_cut ~ normal(m_cut_nom,0.5);
    sigma_cut ~ normal(sig_cut_nom,0.25);

    for (i in 1:calib_sn_idx) {
        calib_mag_cut[calib_dset_idx[i]] ~ normal(calib_m_cut_nom[calib_dset_idx[i]],0.5);
        calib_sigma_cut[calib_dset_idx[i]] ~ normal(calib_sig_cut_nom[calib_dset_idx[i]],0.25);
    }

    for (i in 1:sn_idx) {
        target +=  multi_normal_lpdf(obs[i] | [mag_true[i] + mag_sys[i], v_true[i] + vel_sys[i], c_true[i] + col_sys[i]]', errors[i] + [[sigma_int^2, 0, 0], [0, 0, 0], [0, 0, 0]]);
        if (use_selection != 0) {
          target += normal_lcdf(mag_cut | obs[i][1], sigma_cut) 
            - log(normal_cdf(mag_cut | mean[i], sqrt(v_mi[i])) + 0.0001);
        }
    }
    for (i in 1:calib_sn_idx) {
        target +=  normal_lpdf(calib_obs[i] | [calib_mag_true[i] + calib_mag_sys[i], calib_v_true[i] + calib_vel_sys[i], calib_c_true[i] + calib_col_sys[i]]', sqrt(calib_errors[i] + [calib_sigma_int[calib_dset_idx[i]]^2, 0, 0]'));
        if (use_selection != 0) {
          target += normal_lcdf(calib_mag_cut[calib_dset_idx[i]] | calib_obs[i][1], calib_sigma_cut[calib_dset_idx[i]]) 
            - log(normal_cdf(calib_mag_cut[calib_dset_idx[i]] | calib_mean[i], sqrt(calib_v_mi[i])) + 0.0001);
        }
    }
}
