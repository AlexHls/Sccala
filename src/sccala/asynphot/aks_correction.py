import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib

from dust_extinction.parameter_averages import F19
from specutils import Spectrum1D
from astropy import units as u

from sccala.interplib import interpolators as it
import sccala.asynphot.synphot as base
from sccala.libio import get_paths as pa


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
def get_sn_spectra(
    snname, delimiter=",", modelpath="Models", check_spectra=True, rej_100=True
):
    info = pd.read_csv(os.path.join(modelpath, snname, snname + "_info.csv"))
    model = info["File"].to_list()
    for i in range(len(model)):
        model[i] = os.path.join("Models", str(snname), model[i])
    epoch_mod = info["JD"].to_numpy() - 2400000.5 - info["MJD_Explosion"].to_numpy()
    # Exclude spectra with insufficient wavelength coverage
    if check_spectra:
        mask = []
        for i, mod in enumerate(model):
            data = np.genfromtxt(mod, delimiter=delimiter).T
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


def aks_correction(
    snname,
    photometry_file,
    filter_in,
    filter_out,
    epoch_region=None,
    lsb=[0, 0],
    maxiter=10,
    save_plots=True,
    save_results=True,
    delimiter=",",
    disable_mean_fit=False,
):
    matplotlib.use("TkAgg")
    # SN data
    a_v, z_hel, mjd_explo = get_sn_info(snname)

    # SN photometry
    phot_data = get_sn_phot(photometry_file=photometry_file)

    model, epoch_mod = get_sn_spectra(snname, delimiter=delimiter)

    # Update mjd to time after explosion
    phot_data["mjd"] -= mjd_explo

    # Drop data that is outside the epoch region
    if epoch_region:
        inds = phot_data[
            ((phot_data.mjd < min(epoch_region)) | (phot_data.mjd > max(epoch_region)))
        ].index
        phot_data = phot_data.drop(inds)

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
        if filter_name not in bands_obs:
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
        time = phot_data.mjd.to_numpy()

        # Remove potential NaN values
        # Assumes that if a magnitude is missing, the rest is
        # missing as well and vice versa.
        em = em[~np.isnan(m)]
        time = time[~np.isnan(m)]
        m = m[~np.isnan(m)]

        mag[band], emag[band], mjd[band] = mag_interp(
            time, m, em, epoch_mod, visualize=False
        )

    ### Calibrate spectra ###
    wav_mod_warped = []
    flux_mod_warped = []
    for i in range(len(model)):
        # Load spectrum
        wav_mod, flux_mod = np.genfromtxt(model[i], delimiter=delimiter).T

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
        #        wav_mod = np.concatenate((wav_blank_low, wav_mod, wav_blank_upper), axis=None)

        flux_blank_low = np.zeros_like(wav_blank_low) + 1e-10 * flux_mod[0]
        flux_blank_upper = np.zeros_like(wav_blank_upper) + 1e-10 * flux_mod[-1]
        #        flux_mod = np.concatenate(
        #            (flux_blank_low, flux_mod, flux_blank_upper), axis=None
        #        )

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

        m_v = filters_in.calculate_vega_magnitudes(wav_mod_obs, flux_mod_obs)[
            filter_ind
        ]

        coeff_norm = (10 ** (-0.4 * mag[bands[filter_ind]][i])) / (10 ** (-0.4 * m_v))
        flux_mod_obs = flux_mod_obs * coeff_norm

        # Calculate synthetic magnitudes
        mag_syn = filters_in.calculate_vega_magnitudes(wav_mod_obs, flux_mod_obs)

        # For each filter compare observed photometry to synthetic magnitudes
        offset = []
        for j, band in enumerate(bands):
            delta = mag[band][i] - mag_syn[j]
            offset.append(10 ** (-0.4 * delta))

        mags_warp, flux_warp = warp_spectrum(
            offset, lambda_effs, wav_mod_obs, flux_mod_obs, filters_in
        )

        # Compare warped magnitudes with observed photometry.
        # Require less than 0.05 mag difference
        while False in (
            np.abs(np.array([mag[b][i] for b in bands]) - mags_warp) < 0.05
        ):
            maxiter = maxiter - 1
            if maxiter <= 0:
                print("Did not converge...")
                break

            # Calculate new offsets
            offset = []
            for j, band in enumerate(bands):
                delta = mag[band][i] - mags_warp[j]
                offset.append(10 ** (-0.4 * delta))

            mags_warp, flux_warp = warp_spectrum(
                offset, lambda_effs, wav_mod_obs, flux_warp, filters_in
            )

        # Store flux corrected spectra
        wav_mod_warped.append(wav_mod_obs)
        flux_mod_warped.append(flux_warp)

        print("Observed photometry:")
        print([mag[b][i] for b in bands])
        print("Photometry after flux corection:")
        print(mags_warp)

    ### Calculate the actual AKS correction ###
    aks_corr = {}
    filter_choice = []
    for j, band in enumerate(bands):
        aks_corr[band] = []
        for i in range(len(model)):
            # Synthetic magnitude of warped spectrum in observed frame
            m_obs = filters_in.calculate_vega_magnitudes(
                wav_mod_warped[i], flux_mod_warped[i]
            )[j]

            # Remove Milky Way extinction
            flux_mod_avg = f19_unred(wav_mod_warped[i], flux_mod_warped[i], a_v)

            # Put the warped spectrum in the restframe
            wav_mod_rest = wav_mod_warped[i] / (1 + z_hel)
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
                % (filter_in[j], filter_out[ind_lambda_eff_out])
            )
            if i == 0:
                # Only store the first filter pair. For most SNe, this should be the
                # same for all epochs
                filter_choice.append((filter_in[j], filter_out[ind_lambda_eff_out]))

            # Synthetic magnitude of the warped spectrum in the rest frame
            # corrected for AvG in the output filter
            m_out = filters_out.calculate_vega_magnitudes(
                wav_mod_rest, flux_mod_rest_avg
            )[ind_lambda_eff_out]

            ### AKS correction ###
            aks = m_obs - m_out
            aks_corr[band].append(aks)

    ### Interpolate AKS corrections back to the photometry epochs ###
    aks_corr_phot = {}
    aks_corr_phot_err = {}
    for band in bands:
        aks_interp = it.AKS_Interpolator(
            aks_corr[band], epoch_mod, disable_mean_fit=disable_mean_fit
        )
        aks_interp.sample_posterior(num_live_points=400)
        data_int = aks_interp.predict_from_posterior(phot_data.mjd.to_numpy())
        aks_corr_phot[band] = np.mean(data_int, axis=0)
        aks_corr_phot_err[band] = np.std(data_int, axis=0)

        ### Save diagnostic plots ###
        if save_plots:
            fig, ax = plt.subplots(1, 1)
            ax.plot(
                epoch_mod,
                aks_corr[band],
                label="Spectral epochs",
                linestyle="",
                marker="x",
            )
            ax.errorbar(
                phot_data.mjd.to_numpy(),
                aks_corr_phot[band],
                yerr=aks_corr_phot_err[band],
                marker=".",
                color="tab:orange",
                capsize=0,
                label="Photometry epochs",
            )
            ax.fill_between(
                phot_data.mjd.to_numpy(),
                aks_corr_phot[band] + aks_corr_phot_err[band],
                aks_corr_phot[band] - aks_corr_phot_err[band],
                color="tab:orange",
                alpha=0.3,
            )
            ax.set_title("%s | %s band" % (snname, band))
            ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("AKS correction (mag)")

            diag_path = os.path.join(pa.get_diag_path(), snname)
            if not os.path.exists(diag_path):
                os.mkdir(diag_path)
            fig.savefig(
                os.path.join(diag_path, "{:s}_{:s}_band_AKS.png".format(snname, band)),
                bbox_inches="tight",
            )

    ### Save results ###
    if save_results:
        for i in range(len(filter_choice)):
            _, instr_in, f_in = re.split("/|\.", filter_choice[i][0])
            _, instr_out, f_out = re.split("/|\.", filter_choice[i][1])
            band = bands[i]
            data_out = {}
            data_out["time"] = phot_data.mjd.to_numpy()
            data_out["mag"] = phot_data.mag[band].to_numpy() - aks_corr_phot[band]
            data_out["emag"] = np.sqrt(
                phot_data.emag[band].to_numpy() ** 2 + aks_corr_phot_err[band] ** 2
            )
            data_out["AKS"] = aks_corr_phot[band]
            data_out["AKS_err"] = aks_corr_phot_err[band]

            df = pd.DataFrame(data_out)
            res_path = pa.get_res_path()
            res_path = os.path.join(res_path, "aks_corr")
            if not os.path.exists(res_path):
                os.mkdir(res_path)
            df.to_csv(
                os.path.join(
                    res_path,
                    "%s_%s_to_%s_%s.csv"
                    % (
                        instr_in,
                        f_in,
                        instr_out,
                        f_out,
                    ),
                )
            )

    return aks_corr_phot, aks_corr_phot_err
