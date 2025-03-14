import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sccala.interplib.interpolators import (
    LC_Interpolator,
    Vel_Interpolator,
    AE_Interpolator,
)
from sccala.utillib.aux import convert_to_flux, convert_to_mag

try:
    matplotlib.use("TkAgg")
except ImportError:
    pass


class EpochDataSet:
    """
    Class wrapping data time series and providing
    interpolation functionality
    """

    def __init__(
        self,
        data,
        data_error,
        tkde,
        red,
        mjd,
        snname="",
        errorfloor=0.0,
        errorscale=1.0,
        reg_min=20.0,
        reg_max=60.0,
        extrapolate=5.0,
        size=50,
        num_live_points=800,
        disable_mean_fit=False,
        disable_white_noise_fit=False,
        ignore_toe_uncertainty=False,
    ):
        """
        Initialize EpochDataSet

        Parameters
        ----------
        data : float
            List or array of data
        data_error : float
            List or array of data uncertainties
        tkde : float
            (Resampled) time of explosion KDE
        red : float
            Redshift of the dataset
        mjd : float
            List or array of observation times (NOT in restframe!) in days.
        snname : str
            Name of the SN for which data should be interpolated.
            Default: ""
        errorfloor : float
            Minimum error for velocities. All uncertainties smaller than this
            value will be increased to this value. Default: 0.0
        errorscale : float
            Scales all data uncertainties will be scaled by this factor.
            Default: 1.0
        reg_min : float
            Lower boundary for the data interpolation in days. Data points
            earlier than this value will be excluded from the interpolation
            procedure. Default: 20.0
        reg_max : float
            Upper boundary for the data interpolation in days. Data points
            later than this value will be excluded from the interpolation
            procedure. Default: 60.0
        extrapolate : float
            Allowed extrapolation range in days. Default: 5.0

        """

        assert np.shape(data) == np.shape(data_error) and np.shape(data) == np.shape(
            mjd
        ), "Input data needs to have the same shape"

        self.data = np.array(data)
        self.data_error = np.array(data_error)
        self.tkde = tkde
        self.mjd = np.array(mjd)

        self.snname = snname

        self.errorfloor = errorfloor
        self.errorscale = errorscale
        self.reg_min = reg_min
        self.reg_max = reg_max
        self.extrapolate = extrapolate

        self.toe = float(np.percentile(tkde.resample(10000), 50.0))

        # Confert dates to restframe
        self.time = (mjd - self.toe) / (1 + red)

        # Apply some rules
        for i in range(len(self.data_error)):
            if self.data_error[i] < self.errorfloor:
                self.data_error[i] = self.errorfloor

        if not np.isnan(self.errorscale):
            self.data_error *= self.errorscale

        mask = np.logical_and(self.time > self.reg_min, self.time < self.reg_max)

        self.data_ex = self.data[np.logical_not(mask)]
        self.data_error_ex = self.data_error[np.logical_not(mask)]
        self.time_ex = self.time[np.logical_not(mask)]
        self.mjd_ex = self.mjd[np.logical_not(mask)]

        self.data = self.data[mask]
        self.data_error = self.data_error[mask]
        self.time = self.time[mask]
        self.mjd = self.mjd[mask]

        # results
        self.data_int = None

        # Sampling parameters
        self.size = size
        self.num_live_points = num_live_points
        self.disable_mean_fit = disable_mean_fit
        self.disable_white_noise_fit = disable_white_noise_fit
        self.ignore_toe_uncertainty = ignore_toe_uncertainty

        return

    def get_results(self):
        if self.data_int is None:
            raise ValueError("No interpolated values found")

        self.median = np.percentile(self.data_int, 50, axis=0)
        self.minustwosigma = np.percentile(self.data_int, 2.28, axis=0)
        self.minusonesigma = np.percentile(self.data_int, 15.87, axis=0)
        self.plusonesigma = np.percentile(self.data_int, 84.13, axis=0)
        self.plustwosigma = np.percentile(self.data_int, 97.72, axis=0)

        self.data_int_error_lower = self.median - self.minusonesigma
        self.data_int_error_upper = self.plusonesigma - self.median

        return self.median, self.data_int_error_lower, self.data_int_error_upper

    def exclude_data(self, beginning=True):
        """
        Removes one datapoint from the data set.
        If beginning is True, first element is removed, otherwise last.
        """
        if beginning:
            ind = 0
        else:
            ind = 1
        ex_data = self.data[-1 * ind]
        self.data = np.delete(self.data, -1 * ind, axis=0)
        self.data_ex = np.insert(self.data_ex, ind * len(self.data_ex), ex_data)

        ex_data_err = self.data_error[-1 * ind]
        self.data_error = np.delete(self.data_error, -1 * ind, axis=0)
        self.data_error_ex = np.insert(
            self.data_error_ex, ind * len(self.data_error_ex), ex_data_err
        )

        ex_time = self.time[-1 * ind]
        self.time = np.delete(self.time, -1 * ind, axis=0)
        self.time_ex = np.insert(self.time_ex, ind * len(self.time_ex), ex_time)

        ex_mjd = self.mjd[-1 * ind]
        self.mjd = np.delete(self.mjd, -1 * ind, axis=0)
        self.mjd_ex = np.insert(self.mjd_ex, ind * len(self.mjd_ex), ex_mjd)

        return

    def diagnostic_plot(self, diagnostic, target, flux_interp=False, interactive=False):
        """
        Plots the output of the interpolation
        """

        if flux_interp:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=[6, 9])
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=[6, 6])

        if target == "halpha-ae" or "phot" in target:
            conv = 1
        else:
            conv = 1000  # Conversion factor from m/s to km/s

        if self.data_int is not None:
            plotind = int(len(self.dates) / 2)
            data_int = self.data_int / conv
            ax1.hist(data_int[:, plotind], label="Interpolated")
            ax1.axvspan(
                self.minustwosigma[plotind] / conv,
                self.plustwosigma[plotind] / conv,
                alpha=0.1,
                color="red",
                label=r"2$\sigma$ (95.44%)",
            )
            ax1.axvspan(
                self.minusonesigma[plotind] / conv,
                self.plusonesigma[plotind] / conv,
                alpha=0.3,
                color="red",
                label=r"1$\sigma$ (68.26%)",
            )
            ax1.axvline(
                self.median[plotind] / conv,
                color="red",
                label="Median",
            )
            ax1.set_xlabel("v(t={:.1f}d) (km/s)".format(self.dates[plotind]))
            ax1.set_title(
                r"v({:.1f}) = {:.2f} +{:.2f}/ -{:.2f} km/s | {:s}".format(
                    self.dates[plotind],
                    self.median[plotind] / conv,
                    self.data_int_error_upper[plotind] / conv,
                    self.data_int_error_lower[plotind] / conv,
                    target,
                )
            )
            ax1.legend()

            # Lower subplot
            ax2.axhspan(
                self.minustwosigma[plotind] / conv,
                self.plustwosigma[plotind] / conv,
                alpha=0.1,
                color="red",
            )
            ax2.axhspan(
                self.minusonesigma[plotind] / conv,
                self.plusonesigma[plotind] / conv,
                alpha=0.3,
                color="red",
            )
            ax2.axhline(self.median[plotind] / conv, color="red")
            lower = (
                self.dates[plotind]
                + self.toe
                - np.percentile(self.tkde.resample(10000), 15.87)
            )
            upper = (
                self.dates[plotind]
                + self.toe
                - np.percentile(self.tkde.resample(10000), 84.13)
            )
            ax2.axvspan(
                lower, upper, alpha=0.3, color="blue", label=r"1$\sigma$ (68.26%)"
            )
            ax2.axvline(
                self.dates[plotind],
                color="blue",
                label="{:.1f} days".format(self.dates[plotind]),
            )

            ax2.plot(self.dates, self.median / conv, color="k", label="Prediction")
            ax2.fill_between(
                self.dates,
                self.minusonesigma / conv,
                self.plusonesigma / conv,
                color="k",
                alpha=0.3,
            )
            ax2.fill_between(
                self.dates,
                self.minustwosigma / conv,
                self.plustwosigma / conv,
                color="k",
                alpha=0.1,
            )

            # Plot flux
            if flux_interp:
                ax3.axhspan(
                    convert_to_flux(self.minustwosigma[plotind]) / conv,
                    convert_to_flux(self.plustwosigma[plotind]) / conv,
                    alpha=0.1,
                    color="red",
                )
                ax3.axhspan(
                    convert_to_flux(self.minusonesigma[plotind]) / conv,
                    convert_to_flux(self.plusonesigma[plotind]) / conv,
                    alpha=0.3,
                    color="red",
                )
                ax3.axhline(convert_to_flux(self.median[plotind]) / conv, color="red")
                lower = self.dates[plotind] + self.toe - np.percentile(self.tkde, 15.87)
                upper = self.dates[plotind] + self.toe - np.percentile(self.tkde, 84.13)
                ax3.axvspan(
                    lower, upper, alpha=0.3, color="blue", label=r"1$\sigma$ (68.26%)"
                )
                ax3.axvline(
                    self.dates[plotind],
                    color="blue",
                    label="{:.1f} days".format(self.dates[plotind]),
                )

                ax3.plot(
                    self.dates,
                    convert_to_flux(self.median) / conv,
                    color="k",
                    label="Prediction",
                )
                ax3.fill_between(
                    self.dates,
                    convert_to_flux(self.minusonesigma) / conv,
                    convert_to_flux(self.plusonesigma) / conv,
                    color="k",
                    alpha=0.3,
                )
                ax3.fill_between(
                    self.dates,
                    convert_to_flux(self.minustwosigma) / conv,
                    convert_to_flux(self.plustwosigma) / conv,
                    color="k",
                    alpha=0.1,
                )

        ax2.errorbar(
            x=self.time,
            y=self.data / conv,
            yerr=self.data_error / conv,
            marker="o",
            ls=" ",
            capsize=0,
            color="orange",
            label="Data",
        )
        ax2.errorbar(
            x=self.time_ex,
            y=self.data_ex / conv,
            yerr=self.data_error_ex / conv,
            marker=".",
            ls=" ",
            capsize=0,
            color="tab:red",
            label="Excluded",
        )
        ax2.set_xlabel("Time (days)")
        if "phot" in target:
            ax2.set_ylabel("{:s} (mag)".format(target))
        else:
            ax2.set_ylabel("Velocity (km s$^{-1}$)")
        ax2.legend()
        ax2.minorticks_on()
        ax2.grid(which="major", axis="both", linestyle="-")
        ax2.grid(which="minor", axis="both", linestyle="--")

        ax2.set_xlim([max([ax2.get_xlim()[0], 0]), max([65.0, max(self.time)])])
        if "phot" not in target:
            lims = ax2.get_xlim()
            curr_data = np.where((self.time > lims[0]) & (self.time < lims[1]))[0]
            ax2.set_ylim(
                (self.data - self.data_error)[curr_data].min() / conv * 0.5,
                (self.data - self.data_error)[curr_data].max() / conv * 1.5,
            )

        if "phot" in target:
            ax2.invert_yaxis()

        if flux_interp:
            # Plot flux data
            ax3.errorbar(
                x=self.time,
                y=convert_to_flux(self.data) / conv,
                yerr=convert_to_flux(self.data, self.data_error) / conv,
                marker="o",
                ls=" ",
                capsize=0,
                color="orange",
                label="Data",
            )
            ax3.errorbar(
                x=self.time_ex,
                y=convert_to_flux(self.data_ex) / conv,
                yerr=convert_to_flux(self.data_ex, self.data_error_ex) / conv,
                marker=".",
                ls=" ",
                capsize=0,
                color="tab:red",
                label="Excluded",
            )
            ax3.set_xlabel("Time (days)")
            ax3.set_ylabel(r"{:s} (erg $\AA$)".format(target))
            ax3.legend()
            ax3.minorticks_on()
            ax3.grid(which="major", axis="both", linestyle="-")
            ax3.grid(which="minor", axis="both", linestyle="--")

            ax3.set_xlim([max([ax3.get_xlim()[0], 0]), max([65.0, max(self.time)])])

        plt.tight_layout()

        if interactive:
            plt.show()

        fig.savefig(
            os.path.join(
                diagnostic, "{:s}_Interpolation_{:s}".format(self.snname, target)
            ),
            bbox_inches="tight",
            dpi=100,
        )

        return fig

    def data_interp(
        self,
        target,
        step=0.1,
        date_low=None,
        date_high=None,
        diagnostic=None,
        no_reject=False,
        flux_interp=False,
        interactive=False,
    ):
        """
        Interpolate dataocities using Gaussian Process regression

        Parameters
        ----------
        target : str
            Specifies as what the data is to be interpolated. Determines
            Gaussian Process kernel. Photometry has to contain 'phot' in
            target name.
        step : float
            Resolution of the interpolated data. Default: 0.1
        date_low : int or float
            Lower epoch limit of interpolation.
        date_up : int or float
            Upper epoch limit of interpolation.
        diagnostic : str or None
            Path to directory where diagnostic plots are to be saved.
        no_reject : bool
            If True velocity fits with increasing values will not
            be rejected. Default: False
        flux_interp : bool
            If True, data is converted to flux before interpolating.
            Exported values will be converted back to magnitudes.
            Only works with 'phot' in target. Default: False
        interactive : bool
            If True, diagnostic plots will be shown interactively.

        Returns
        -------
        median : float
            median data value at date days
        data_int_error_lower : float
        data_int_error_upper : float
            return values are in m/s
        date : float
            date to which the magnitudes have been interpolated
        """

        if "phot" not in target and flux_interp:
            raise ValueError("Flux interpolation only works with photometry targets")

        if date_low is None:
            date_low = self.reg_min
        if date_high is None:
            date_high = self.reg_max

        if len(self.data) < 2:
            warnings.warn("Insufficient datapoints for interpolation, skipping...")
            if diagnostic:
                self.diagnostic_plot(diagnostic, target)

            return None

        if max(self.time) + self.extrapolate > date_high:
            date = np.arange(
                start=date_low, stop=max(self.time) + self.extrapolate + step, step=step
            )
        else:
            date = np.arange(start=date_low, stop=date_high + step, step=step)

        self.dates = date

        if flux_interp:
            data = convert_to_flux(self.data)
            data_error = convert_to_flux(self.data, self.data_error)
        else:
            data = self.data
            data_error = self.data_error

        if "phot" in target:
            interpolator = LC_Interpolator(
                data,
                data_error,
                self.time,
                disable_mean_fit=self.disable_mean_fit,
                disable_white_noise_fit=self.disable_white_noise_fit,
            )
            interpolator.sample_posterior(num_live_points=self.num_live_points)

            if self.ignore_toe_uncertainty:
                data_int = interpolator.predict_from_posterior(
                    date, tkde=None, toe=self.toe, size=self.size
                )
            else:
                data_int = interpolator.predict_from_posterior(
                    date, tkde=self.tkde, toe=self.toe, size=self.size
                )
        elif target == "halpha-ae":
            interpolator = AE_Interpolator(
                data,
                data_error,
                self.time,
                disable_mean_fit=self.disable_mean_fit,
                disable_white_noise_fit=self.disable_white_noise_fit,
            )
            interpolator.sample_posterior(num_live_points=self.num_live_points)

            if self.ignore_toe_uncertainty:
                data_int = interpolator.predict_from_posterior(
                    date, tkde=None, toe=self.toe, size=self.size
                )
            else:
                data_int = interpolator.predict_from_posterior(
                    date, tkde=self.tkde, toe=self.toe, size=self.size
                )
        elif target:
            interpolator = Vel_Interpolator(
                data,
                data_error,
                self.time,
                disable_mean_fit=self.disable_mean_fit,
                disable_white_noise_fit=self.disable_white_noise_fit,
            )
            interpolator.sample_posterior(num_live_points=self.num_live_points)

            if self.ignore_toe_uncertainty:
                data_int = interpolator.predict_from_posterior(
                    date, tkde=None, toe=self.toe, size=self.size
                )
            else:
                data_int = interpolator.predict_from_posterior(
                    date, tkde=self.tkde, toe=self.toe, size=self.size
                )
        else:
            raise NotImplementedError("Target has no implemented interpolator")

        # If no matching curves are found, start excluding data
        # until interpolation is successful
        if len(data_int) <= 3:
            warnings.warn("No valid parameters found, excluding datapoints...")
            self.reg_min = self.time[0] + 0.1
            self.exclude_data()

            return self.data_interp(
                target,
                step=step,
                date_low=date_low,
                date_high=date_high,
                diagnostic=diagnostic,
                no_reject=no_reject,
                flux_interp=flux_interp,
            )

        if flux_interp:
            self.data_int = convert_to_mag(np.array(data_int))
        else:
            self.data_int = np.array(data_int)

        self.get_results()

        if diagnostic:
            self.diagnostic_plot(
                diagnostic, target, flux_interp=flux_interp, interactive=interactive
            )

        return (
            self.median,
            self.data_int_error_lower,
            self.data_int_error_upper,
            self.dates,
        )
