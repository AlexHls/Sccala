import os
import warnings

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas as pd

import emcee
import george
from george import kernels


class VelocitySet:
    """
    Class wrapping velocity time series and providing
    interpolation functionality
    """

    def __init__(
        vel,
        vel_error,
        tkde,
        red,
        mjd,
        snname="",
        errorfloor=0.0,
        errorscale=1.0,
        reg_min=20.0,
        reg_max=60.0,
        extrapolate=5.0,
    ):
        """
        Initialize VelocitySet

        Parameters
        ----------
        vel : float
            List or array of velocities in m/s
        vel_error : float
            List or array of velocity uncertainties in m/s
        tkde : float
            (Resampled) time of explosion KDE
        red : float
            Redshift of the dataset
        mjd : float
            List or array of observation times (NOT in restframe!) in days.
        snname : str
            Name of the SN for which velocitys should be interpolated.
            Default: ""
        errorfloor : float
            Minimum error for velocities. All uncertainties smaller than this
            value will be increased to this value. Default: 0.0
        errorscale : float
            Scales all velocity uncertainties will be scaled by this factor.
            Default: 1.0
        reg_min : float
            Lower boundary for the velocity interpolation in days. Data points
            earlier than this value will be excluded from the interpolation
            procedure. Default: 20.0
        reg_max : float
            Upper boundary for the velocity interpolation in days. Data points
            later than this value will be excluded from the interpolation
            procedure. Default: 60.0
        extrapolate : float
            Allowed extrapolation range in days. Default: 5.0

        """

        assert np.shape(vel) == np.shape(vel_error) and np.shape(vel) == np.shape(
            mjd
        ), "Input data needs to have the same shape"

        self.vel = np.array(vel)
        self.vel_error = np.array(vel_error)
        self.tkde = np.array(tkde)
        self.mjd = np.array(mjd)

        self.snname = snname

        self.errorfloor = errorfloor
        self.errorscale = errorscale
        self.reg_min = reg_min
        self.reg_max = reg_max
        self.extrapolate = extrapolate

        self.toe = np.percentile(tkde, 50.0)

        # Confert dates to restframe
        self.time = (mjd - self.toe) / (1 + red)

        # Apply some rules
        for i in range(len(self.vel_error)):
            if self.vel_error[i] < self.errorfloor:
                self.vel_error[i] = self.errorfloor

        if not np.isnan(self.errorscale):
            self.vel_error *= self.errorscale

        mask = np.logical_and(self.time > self.reg_min, self.time < self.reg_max)

        self.vel_ex = self.vel[np.logical_not[mask]]
        self.vel_error_ex = self.vel_error[np.logical_not[mask]]
        self.time_ex = self.time[np.logical_not[mask]]
        self.mjd_ex = self.mjd_ex[np.logical_not[mask]]

        self.vel = self.vel[mask]
        self.vel_error = self.vel_error[mask]
        self.time = self.time[mask]
        self.mjd = self.mjd[mask]

        # results
        self.vel_pred = None
        self.vel_int = None

        return

    def get_results():

        if self.vel_int is None:
            raise ValueError("No interpolated values found")

        self.median = np.percentile(vel_int, 50, axis=0)
        self.minustwosigma = np.percentile(vel_int, 2.28, axis=0)
        self.minusonesigma = np.percentile(vel_int, 15.87, axis=0)
        self.plusonesigma = np.percentile(vel_int, 84.13, axis=0)
        self.plustwosigma = np.percentile(vel_int, 97.72, axis=0)

        self.vel_int_error_lower = self.median - self.minusonesigma
        self.vel_int_error_upper = self.plusonesigma - self.median

        return self.median, self.vel_int_error_lower, self.vel_int_error_upper

    def exclude_data(beginning=True):
        """
        Removes one datapoint from the data set.
        If beginning is True, first element is removed, otherwise last.
        """
        if beginning:
            ind = 0
        else:
            ind = 1
        ex_vel = self.vel[-1 * ind]
        self.vel = np.delete(self.vel, -1 * ind, 0)
        self.vel_ex = np.insert(self.vel_ex, ind * len(self.vel_ex), 0)

        ex_vel_err = self.vel_error[-1 * ind]
        self.vel_error = np.delete(self.vel_error, -1 * ind, 0)
        self.vel_error_ex = np.insert(
            self.vel_error_ex, ind * len(self.vel_error_ex), 0
        )

        ex_time = self.time[-1 * ind]
        self.time = np.delete(self.time, -1 * ind, 0)
        self.time_ex = np.insert(self.time_ex, ind * len(self.time_ex), 0)

        ex_mjd = self.mjd[-1 * ind]
        self.mjd = np.delete(self.mjd, -1 * ind, 0)
        self.mjd_ex = np.insert(self.mjd_ex, ind * len(self.mjd_ex), 0)

        return

    def diagnostic_plot(diagnostic, line):
        """
        Plots the output of the interpolation
        """

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=[6, 6])

        if line == "halpha-ae":
            conv = 1
        else:
            conv = 1000  # Conversion factor from m/s to km/s

        if self.vel_int is not None:
            plotind = int(len(self.dates) / 2)
            vel_int = self.vel_int / conv
            ax1.hist(vel_int[:, plotind], label="Interpolated")
            ax1.axvspan(
                self.minustwosigma[plotind] / conv,
                self.plustwosigma[plotind] / conv,
                alpha=0.1,
                color="red",
                label="2$\sigma$ (95.44%)",
            )
            ax1.axvspan(
                self.minusonesigma[plotind] / conv,
                self.plusonesigma[plotind] / conv,
                alpha=0.3,
                color="red",
                label="1$\sigma$ (68.26%)",
            )
            ax1.axvline(
                self.median[plotind] / conv,
                color="red",
                label="Median",
            )
            ax1.set_xlabel("v(t={:.1f})d) (km/s)".format(self.dates[plotind]))
            ax1.set_title(
                r"v({:.1f}) = {:.2f} +{:.2f}/ -{:.2f} km/s | {:s}".format(
                    self.dates[plotind],
                    self.median[plotind] / conv,
                    self.vel_int_error_upper[plotind] / conv,
                    self.vel_int_error_lower[plotind] / conv,
                    line,
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
            ax2.axhline(median[plotind] / conv, color="red")
            lower = self.dates[plotind] + self.toe - np.percentile(self.tkde, 15.87)
            upper = self.dates[plotind] + self.toe - np.percentile(self.tkde, 84.13)
            ax2.axvspan(
                lower, upper, alpha=0.3, color="blue", label="1$\sigma$ (68.26%)"
            )
            ax2.axvline(
                self.dates[plotind],
                color="blue",
                label="{:.1f} days".format(self.dates[plotind]),
            )

            for v in self.vel_pred[:100]:
                ax2.plot(self.x_pred, np.array(v) / conv, color="")

        ax2.errorbar(
            x=self.time,
            y=self.vel / conv,
            yerr=self.vel_error / conv,
            marker="o",
            ls=" ",
            capsize=0,
            color="orange",
            label="Data",
        )
        ax2.errorbar(
            x=self.time_ex,
            y=self.vel_ex / conv,
            yerr=self.vel_error_ex / conv,
            marker=".",
            ls=" ",
            capsize=0,
            color="tab:red",
            label="Excluded",
        )
        ax2.set_xlabel("Time (days)")
        ax2.set_ylabel("Velocity (km s$^{-1}$)")
        ax2.legend()
        ax2.minorticks_on()
        ax2.grid(which="major", axis="both", linestyle="-")
        ax2.grid(which="minor", axis="both", linestyle="--")

        ax2.set_xlim([max([ax2.get_xlim()[0], 0]), max([65.0, max(self.time)])])

        plt.tight_layout()

        fig.savefig(
            os.path.join(
                diagnostic, "{:s}_VelocityInterpolation_{:s}".format(self.snname, line)
            ),
            bbox_inches="tight",
            dpi=100,
        )

        return fig

    def vel_interp(
        line,
        step=0.1,
        date_low=self.reg_min,
        date_high=self.reg_max,
        diagnostic=None,
    ):
        """
        Interpolate velocities using Gaussian Process regression

        Parameters
        ----------
        line : str
            Specifies which line velocity is to be interpolated. Determines
            Gaussian Process kernel.
        step : float
            Resolution of the interpolated data. Default: 0.1
        date_low : int or float
            Lower epoch limit of interpolation.
        date_up : int or float
            Upper epoch limit of interpolation.
        diagnostic : str or None
            Path to directory where diagnostic plots are to be saved.

        Returns
        -------
        median : float
            median velocity at date days
        vel_int_error_lower : float
        vel_int_error_upper : float
            return values are in m/s
        date : float
            date to which the magnitudes have been interpolated
        """

        if len(self.vel) < 2:
            warnings.warn("Insufficient datapoints for interpolation, skipping...")
            if diagnostic:
                self.diagnostic_plot(diagnostic, line)

            return None

        if max(self.time) + self.extrapolate > date_high:
            date = np.arange(
                start=date_low, stop=max(self.time) + self.extrapolate + step, step=step
            )
        else:
            date = np.arange(start=date_low, stop=date_high + step, step=step)

        self.dates = date

        if line == "halpha-ae":
            kernel = np.var(vel) * kernels.PolynomialKernel(
                log_sigma2=np.var(self.vel), order=3
            )
        else:
            kernel = np.var(self.vel) * kernels.ExpSquaredKernel(1000)

        model = george.GP(
            kernel=kernel,
        )

        try:
            model.compute(self.time, self.vel_err)
        except np.linalg.LinAlgError as err:
            warnings.warn("LinAlgError occured, modifying vel_error...")
            model.compute(self.time, self.vel_err * 2)

        # Emcee sampling
        def lnprob(p):
            model.set_parameter_vector(p)
            return model.log_likelihood(velocity, quiet=True) + model.log_prior()

        initial = model.get_parameter_vector()
        ndim, nwalkers = len(initial), 32
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

        try:
            print("Running first burn-in...")
            p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
            p0, lp, _ = sampler.run_mcmc(p0, 400)

            print("Running second burn-in...")
            p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
            sampler.reset()
            p0, _, _ = sampler.run_mcmc(p0, 400)

            print("Running production...")
            sampler.run_mcmc(p0, 800)
        except ValueError:
            warnings.warn("ValueError occured, skipping second burn-in...")
            ndim, nwalkers = len(initial), 32
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
            print("Running first burn-in...")
            p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
            p0, _, _ = sampler.run_mcmc(p0, 400)
            sampler.reset()
            print("Running production...")
            sampler.run_mcmc(p0, 800)

        x_pred = np.linspace(min(date), max([max(date), 45, max(self.time)]), 100)
        self.x_pred = x_pred

        samples = sampler.flatchain

        # Draw time from toe prior
        size = 125
        rng = default_rng()
        uni_rng = rng.uniform(size=size)

        vel_int = []
        vel_pred = []
        for s in samples[np.random.randint(len(samples), size=size)]:
            model.set_parameter_vector(s)
            v = model.predict(self.vel, x_pred, return_cov=False)

            # Rule to skip "0-fits"
            if np.mean(v) < 1 and line != "halpha-ae":
                continue
            # Reject curve if values increase
            if any(np.sign(np.diff(v)) == 1) and line != "halpha-ae":
                continue

            vel_pred.append(v)
            for num in uni_rng:
                toe_rnd = np.percentile(self.tkde, num * 100)
                t = date + (self.toe - toe_rnd)
                vint = model.predict(self.vel, t, return_cov=False)
                vel_int.append(vint)

            # If no matching curves are found, start excluding data
            # until interpolation is successful
        if len(vel_int) <= 3:
            warnings.warn("No valid parameters found, excluding datapoints...")
            self.reg_min = self.time[0] + 0.1
            self.mask_data()

            return self.interp_velocity(
                line,
                step=step,
                date_low=date_low,
                date_high=date_high,
                diagnostic=diagnostic,
            )

        self.get_results()

        if diagnostic:
            self.diagnostic_plot(diagnostic, line)

        return (
            self.median,
            self.vel_int_error_lower,
            self.vel_int_error_upper,
            self.dates,
        )
