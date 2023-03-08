import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import integrate
from scipy import interpolate
import scipy.optimize as op
import matplotlib.pyplot as plt

from dust_extinction.parameter_averages import F19
from specutils import Spectrum1D
from astropy import units as u
import george
from george import kernels

from sccala.interplib import interpolators as it
import sccala.asynphot.synphot as base
from sccala.utillib.const import C_AA, H_ERG


# Get SN metadata
def get_sn_info(snname, modelpath="Models"):
    info = pd.read_csv(os.path.join(modelpath, snname, snname + "_info.csv"))
    a_v = np.mean(info["A_V"])
    z_hel = np.mean(info["Redshift"])
    mjd_explo = np.mean(info["MJD_Explosion"])

    return a_v, z_hel, mjd_explo


# Get SN photometry
def get_sn_phot(photometry_file):
    data = pd.read_csv(photometry_file)
    data.rename(
        columns={
            "magnitude": "mag",
            "e_magnitude": "emag",
            "time": "mjd",
        },
        inplace=True,
    )
    data.sort_values(by=["mjd"])
    tels = data["telescope"].unique().tolist()
    assert (
        len(tels) == 1
    ), "More than one telescope found. Please correct each telescope separately."

    phot_data = data.pivot_table(
        index=["mjd"], columns=["band"], values=["mag", "emag"]
    )
    phot_data = phot_data.reset_index(drop=False)

    return phot_data


# Get SN spectra
def get_sn_spectra(snname, modelpath="Models", check_spectra=True, rej_100=True):
    info = pd.read_csv(os.path.join(modelpath, snname, snname + "_info.csv"))
    model = info["File"].to_list()
    for i in range(len(Model)):
        model[i] = os.path.join("Models", str(snname), model[i])
    epoch_mod = info["JD"].to_numpy() - 2400000.5 - info["MJD_Explosion"].to_numpy()
    # Exclude spectra with insufficient wavelength coverage
    if check_spectra:
        mask = []
        for i, mod in enumerate(model):
            data = np.genfromtxt(mod, delimiter="  ").T
            if max(data[0]) < 9200 or min(data[0]) > 4700:
                mask.append(i)
        model = np.delete(model, mask)
        epoch_mod = np.delete(epoch_mod, mask)
    if rej_100:
        mask = []
        for i, ep in enumerate(epoch_mod):
            if ep > 100:
                mask.append(i)
        model = np.delete(model, mask)
        epoch_mod = np.delete(epoch_mod, mask)

    print("Found %d spectra for %s" % (len(model), snname))

    return model, epoch_mod


# Interpolate magnitudes
def mag_interp(mjd, mag, emag, spec_epoch, band=None, visualize=True):
    lc_interp = it.LC_Interpolator(mag, emag, mjd)
    lc_interp.sample_posterior(num_live_points=400)
    data_int = lc_interp.predict_from_posterior(spec_epoch)

    pred_std = np.std(data_int, axis=0)
    pred = np.mean(data_int, axis=0)

    if visualize:
        plt.errorbar(
            spec_epoch,
            pred,
            yerr=pred_std,
            fmt=".r",
            capsize=0,
            label="Spec-Epochs",
        )
        plt.fill_between(
            spec_epoch,
            pred + pred_std,
            pred - pred_std,
            color="red",
            alpha=0.3,
        )
        plt.errorbar(mjd, mag, yerr=emag, fmt=".k", capsize=0, label="Photometry")
        if band is not None:
            plt.title(band)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.xlabel("Epoch")
        plt.ylabel("Magnitude")
        plt.show()

    return pred, pred_std, spec_epoch


# Unred model spectra with Fitzenpatrick 2019
def f19_unred(wave, flux, av, **kwargs):
    """
    Unred spectra with Fitzenpatrick 2019
    Only uses Milky Way E(B-V), ignoring host galaxy extinction
    Assumes R_V = 3.1 per default
    """
    # kwargs
    # Set defaults
    R_V = 3.1
    REDDEN = False

    for key in kwargs:
        if key.lower() == "r_v":
            R_V = kwargs[key]
        if key.lower() == "redden":
            REDDEN = kwargs[key]

    # Define extinction model
    ext = F19(Rv=R_V)

    # Pack data into Spectrum 1D object
    wave = wave * u.AA
    flux = flux * u.Jy  # Units here are just dummy units
    spec = Spectrum1D(spectral_axis=wave, flux=flux)

    if REDDEN:
        # Extinguish (redden) the spectrum
        spectrum_noext = spec * ext.extinguish(wave, Av=av)
    else:
        # Unextinguish (deredden) the spectrum
        spectrum_noext = spec / ext.extinguish(wave, Av=av)

    return spectrum_noext.flux.value


def warp_spectrum(offset, lambda_effs, wav, flux, filters, kind="linear"):
    if len(offset) > 1:
        interp = interpolate.interp1d(lambda_effs, offset, kind=kind)
        fine_x = np.linspace(min(lambda_effs), max(lambda_effs), 150)
        y_interp = interp(fine_x)
    else:
        fine_x = lambda_effs
        y_interp = offset

    ### Flat extrapolation ###
    # Create fake filter with the same value as the first filter to cover the
    # blue part of the spectrum
    y_interp = np.insert(y_interp, 0, y_interp[0])
    fine_x = np.insert(fine_x, 0, min(wav))

    # Create fake filter with the same value as the last filter to cover the
    # red part of the spectrum
    y_interp = np.append(y_interp, y_interp[-1])
    fine_x = np.append(fine_x, max(wav))

    # Create warping function
    f_warp_func = interpolate.interp1d(fine_x, y_interp)
    coeff_flux = f_warp_func(wav)
    flux_warp = np.abs(coeff_flux * flux)

    ### Verification ###
    mags_warp = filters.calculate_vega_magnitudes(wav, flux_warp)

    return mags_warp, flux_warp


def main(
    snname,
    photometry_file,
    filter_in,
    filter_out,
    epoch_region=None,
    lsb=[0, 0],
    maxiter=10,
):
    # SN data
    a_v, z_hel, mjd_explo = get_sn_info(snname)

    # SN photometry
    phot_data = get_sn_phot(photometry_file=photometry_file)

    model, epoch_mod = get_sn_spectra(snname)

    # Update mjd to time after explosion
    phot_data["mjd"] -= mjd_explo

    # Drop data that is outside the epoch region
    if epoch_region:
        inds = phot_data[
            ((phot_data.mjd > min(epoch_region)) | (phot_data.mjd < max(epoch_region)))
        ]
        phot_data.drop(inds)

    # Select spectral epochs that are covered by photometry
    ep_mask = np.logical_and(
        epoch_mod > min(phot_data.mjd) - lsb[0], epoch_mod < max(phot_data.mjd) + lsb[1]
    )
    print(
        "Neglecting %d spectra due to extrapolation restriction..."
        % (len(model) - len(model[ep_mask]))
    )
    model = model[ep_mask]
    epoch_mod = epoch_mod[ep_mask]
    print(epoch_mod)

    # Load filters
    filters_in = base.FilterSet(filter_in)
    filters_out = base.FilterSet(filter_out)

    bands_obs = phot_data.mag.columns.tolist()
    bands = []
    # Make sure that only observed bands are on input filters
    for i, f in enumerate(filters_in):
        _, _, filter_name = re.split("/|\.", f.filter_id)
        if filter_name not in bands:
            filter_in.remove(filter_name)
            warnings.warn(
                "Could not find %s in observed photometry. This might"
                " lead to unexpected results." % filter_name
            )
        else:
            # This ensures that the band indices match the filter ones
            bands.append(filter_name)

    filters_in = base.FilterSet(filter_in)

    mjd = {}
    mag = {}
    emag = {}
    for band in bands:
        m = phot_data.mag[band].to_numpy()
        em = phot_data.emag[band].to_numpy()
        time = phot_data.mjd[band].to_numpy()

        # Remove potential NaN values
        # Assumes that if a magnitude is missing, the rest is
        # missing as well and vice versa.
        m = m[~np.isnan(m)]
        em = em[~np.isnan(m)]
        time = time[~np.isnan(m)]

        mag[band], emag[band], mjd[band] = mag_interp(time, m, em, epoch_mod)

    ### Calibrate spectra ###
    wav_mod_warped = []
    flux_mod_warped = []
    for i in range(len(model)):
        # Load spectrum
        wav_mod, flux_mod = np.genfromtxt(model[i]).T

        # Remove IR part
        mask = np.logical_and(wav_mod < 16000, wav_mod > 1500)
        wav_mod = wav_mod[mask]
        flux_mod = flux_mod[mask]

        # Check for NaNs
        if np.isnan(wav_mod).any() or np.isnan(flux_mod).any():
            print(model[i])
            raise ValueError("Spectra contain NaN")

        # If model file name contains 'dered', redden spectrum again
        if "dered" in model[i]:
            flux_mod = f19_unred(wav_mod, flux_mod, a_v, redden=True)

        # Extend spectra to prevent interpolation problems
        wav_blank_low = np.arange(1500, min(wav_mod))
        wav_blank_upper = np.arange(max(wav_mod), 16000)
        wav_mod = np.concatenate((wav_blank_low, wav_mod, wav_blank_upper), axis=None)

        flux_blank_low = np.zeros_like(wav_blank_low)
        flux_blank_upper = np.zeros_like(wav_blank_upper)
        flux_mod = np.concatenate(
            (flux_blank_low, flux_mod, flux_blank_upper), axis=None
        )

        # Put the rest frame model into the observed frame
        wav_mod_obs = wav_mod * (1 + z_hel)
        flux_mod_obs = flux_mod / (1 + z_hel)

        # Derive effective wavelength for input filters
        lambda_effs = filters_in.calculate_lambda_effs(wav_mod, flux_mod)

        # Normalize spectra using the V band. It is always better to
        # First normalize using one band, therefore, the other
        # corrections will be smallest
        # If there is no 'V' band in the filter list, take the
        # band that has the closest effective wavelength (5455)
        # To make this more flexible,
        if "V" in filters_in:
            filter_ind = [i for i, s in enumerate(bands) if "V" in s][0]
        else:
            filter_ind = np.argmin(np.abs(lambda_effs - 5455))

        m_v = filter_in[filter_ind].calculate_vega_magnitude(wav_mod_obs, flux_mod_obs)

        coeff_norm = (10 ** (-0.4 * mag[filter_ind][i])) / (10 ** (-0.4 * m_v))
        flux_mod_obs = flux_mod_obs * coeff_norm

        # Calculate synthetic magnitudes
        mag_syn = filters_in.calculate_vega_magnitudes(wav_mod_obs, flux_mod_obs)

        # For each filter compare observed photometry to synthetic magnitudes
        offset = []
        for j, band in enumerate(bands):
            delta = mag[j][i] - mag_syn[j]
            offset.append(10 ** (-0.4 * delta))

        mags_warp, flux_warp = warp_spectrum(
            offset, lambda_effs, wav_mod_obs, flux_mod_obs, filters_in
        )

        # Compare warped magnitudes with observed photometry.
        # Require less than 0.05 mag difference
        while False in (np.abs(mag[:][i] - mags_warp) < 0.05):
            maxiter = maxiter - 1
            if maxiter <= 0:
                print("Did not converge...")
                break

            # Calculate new offsets
            offset = []
            for j, band in enumerate(bands):
                delta = mag[j][i] - mag_warp[j]
                offset.append(10 ** (-0.4 * delta))

            mags_warp, flux_warp = warp_spectrum(
                offset, lambda_effs, wav_mod_obs, flux_warp, filters_in
            )

        # Store flux corrected spectra
        wav_mod_warped.append(wav_mod_obs)
        flux_mod_warped.append(flux_warp)

        print("Observed photometry:")
        print(mag[:][i])
        print("Photometry after flux corection:")
        print(mags_warp)

    ### Calculate the actual AKS correction ###
    aks_corr = {}
    for j, band in enumerate(bands):
        aks_corr[band] = []
        for i in range(len(model)):
            # Synthetic magnitude of warped spectrum in observed frame
            m_obs = filters_in[i].calculate_vega_magnitude(wav_mod_obs, flux_mod_warped)

            # Remove Milky Way extinction
            flux_mod_avg = f19_unred(wav_mod_obs, flux_mod_warped, a_v)

            # Put the warped spectrum in the restframe
            wav_mod_rest = wav_mod_obs / (1 + z_hel)
            flux_mod_rest_avg = flux_mod_avg * (1 + z_hel)

            ### K-correction ###
            # At low redshifts probably irrelevant: a photon emmited in the B
            # band will be received in the same band. For example, a photon
            # received in the KAIT-V band will be emitted in the Bessell-V band
            # Get effective wavelengths of output filter systems
            lambda_effs_out = filters_out.calculate_lambda_effs(
                wav_mod_rest, flux_mod_rest_avg
            )

            # Move observed lambda_effs to rest frame
            lambda_eff_rest = lambda_effs[j] / (1 + z_hel)

            ind_lambda_eff_out = np.argmin(np.abs(lambda_effs_out - lambda_eff_rest))
            print(
                "Observed %s band filter corresponds to restframe %s band filter"
                % (band, filter_out[ind_lambda_eff_out])
            )

            # Synthetic magnitude of the warped spectrum in the rest frame
            # corrected for AvG in the output filter
            m_out = filters_out[ind_lambda_eff_out].calculate_vega_magnitude(
                wav_mod_rest, flux_mod_rest_avg
            )

            ### AKS correction ###
            aks = m_obs - m_out
            aks_corr[band].append(aks)

    ### Interpolate AKS corrections back to the photometry epochs

    return
