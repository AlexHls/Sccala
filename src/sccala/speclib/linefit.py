import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import george
from george import kernels
import emcee


def velocity_conversion(x, rest=4861):
    """Converts wavelength into velocity with relativistic
    Doppler formula

    Parameters
    ----------
    x : float
        wavelength to convert
    rest : float
        restwavelength w.r.t to which to convert

    Returns
    -------
    vel : float
        velocity in m/s
    """
    return 299792458 * (rest**2 - x**2) / (rest**2 + x**2)


class LineFit:
    """
    Base class encapsulating line fits of a spectrum
    """

    def __init__(self, wav, flux, error, numcode=100):
        """
        Initializes line fit object.

        Parameters
        ----------
        wav : float
        flux : float
        error : float or None
        numcode : int
            Numeric specifier of spectrum. Used while exporting data.
            Default: 100
        """

        # Check if error has correct format
        assert (
            len(error) == len(flux) or len(error) == 1
        ), "Length of error does not match length of flux: %d <-> %d" % (
            len(error),
            len(flux),
        )

        self.wav = wav
        self.norm_factor = max(flux)
        self.flux = flux / self.norm_factor
        self.error = error / self.norm_factor

        self.numcode = numcode

        self.fits = {}  # Dictionary containing all linefits

        # [cr_low, cr_high, rest, ae_feature]
        self.lines = {
            "halpha-ae": [6250, 6650, 6563, True],
            "hbeta": [4550, 4900, 4861, False],
        }

    def __get_builtin_lines__(self):
        for key in list(self.lines.keys()):
            print(
                "%s: %e | %e | %e | %r \n"
                % (
                    key,
                    self.lines[key][0],
                    self.lines[key][1],
                    self.lines[key][2],
                    self.lines[key][3],
                )
            )
        return

    def __reset_builtin_lines__(self):
        """
        Resets builtin lines in case their properties have
        been overwritten
        """
        self.lines = {
            "halpha-ae": [6250, 6650, 6563, True],
            "hbeta": [4550, 4900, 4861, False],
        }
        return

    def fit_line(
        self,
        line,
        cr_low=None,
        cr_high=None,
        rest=None,
        ae_feature=False,
        noisefit=True,
        hodlrsolver=True,
        size=10000,
        diagnostic=None,
    ):
        """
        Fits a specific line to the spectrum. Lines can either be
        choosen from the predefined set or specified manually

        Parameters
        ----------
        line : str
            Specifies the name of the line. Has to match the predefined
            line names, unless cr_low, cr_high and rest are specified.
        cr_low : float or None
            Lower boundary of the line feature. Default: None
        cr_high : float or None
            Upper boundary of the line feature. Default: None
        rest : float or None
            Rest wavelength of the feature. Default: None
        ae_feature : bool
            Specifies if absorption to emission ratio is to be fit.
            Default: False
        noisefit : bool
            Determines if correlated noise is to be fit. If false, only one
            kernel is used, otherwise two will be used. Default: True
        hodlrsolver : bool
            Determines if george.HODLRSolver is to be used. Default: True
        size : int
            Number of flux predictions to be drawn. Default: 10000
        diagnostic : str
            Determines if diagnostic plots are to be saved in specified
            directory. Default: False

        Returns
        -------
        results : dict
            Dictonary containing all the different fit quantities
        """

        # Check if specified line is defined
        if line in list(self.lines.keys()):
            cr_low = self.lines[line][0]
            cr_high = self.lines[line][1]
            rest = self.lines[line][2]
            ae_feature = self.lines[line][3]
        else:
            if cr_low is None or cr_high is None or rest is None:
                raise ValueError(
                    "Line not found in built in set and no range specifications given..."
                )
            else:
                # Add user defined line to line dict
                self.lines[line] = [cr_low, cr_high, rest, ae_feature]

        print("Fitting %s line...")

        # Cut off any additional secondary feature (particularly for hbeta features)
        while True:
            cutting_range = np.logical_and(self.wav > cr_low, self.wav < cr_high)
            wav = self.wav[cutting_range]
            flux = self.flux[cutting_range]
            if self.error is not None:
                error = self.error[cutting_range]
            if wavelength[np.argmin(flux)] < rest:
                break
            cr_high -= 20

        self.lines[line][1] = cr_high

        # Determine scale lengths for Gaussian Process
        scale = max(wav) - min(wav)
        # Length scale of correlations spanning 5 bins
        corr_scale = np.mean(np.diff(wav))

        # Define kernels
        if noisefit:
            # Kernel1 for the actual long scale peak/ absorbtion minimum
            kernel1 = np.var(flux) * kernels.ExpSquaredKernel(scale)
            # Kernel2 for the (correlated) noise
            kernel2 = np.var(flux) * kernels.ExpSquaredKernel(corr_scale)
            kernel = kernel1 + kernel2
        else:
            kernel = np.var(flux) * kernels.ExpSquaredKernel(scale)

        # Set-up GP
        if hodlrsolver:
            gp = george.GP(kernel, solver=george.HODLRSolver)
        else:
            gp = george.GP(kernel)
        gp.compute(wav, error)

        # Emcee sampling
        def lnprob(p):
            if noisefit:
                if p[3] > corr_scale:
                    return -np.inf
            gp.set_parameter_vector(p)
            return gp.log_likelihood(flux, quiet=True) + gp.log_prior()

        initial = gp.get_parameter_vector()
        ndim, nwalkers = len(initial), 32
        sampler = emcee.EnsambleSampler(nwalkers, ndim, lnprob)
        sampler.reset()

        try:
            print("Running first burn-in...")
            p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
            p0, lp, _ = sampler.run_mcmc(p0, 400)

            print("Running second burn-in...")
            p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
            sampler.reset()
            p0, _, _ = sampler.run_mcmc(p0, 800)
            sampler.reset()

            print("Running production...")
            sampler.run_mcmc(p0, 1000)
        except ValueError:
            warnings.warn("ValueError occured, skipping second burn-in...")
            ndim, nwalkers = len(initial), 32
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
            print("Running first burn-in...")
            p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
            p0, _, _ = sampler.run_mcmc(p0, 400)
            sampler.reset()
            print("Running production...")
            sampler.run_mcmc(p0, 1000)

        # The positions where the prediction should be computed.
        x = np.linspace(min(wav), max(wav), len(wav) * 10)

        # Draw from samples to get peak estimate
        samples = sampler.flatchain
        minima = []
        flux_pred = []
        avg_noise = []
        ae_avg = []
        mins = []
        maxs = []

        # Minima cutoff from wavelenth minimum
        lowcut = 30

        print("Predicting flux...")
        # Mask to ensure only minima in the middle region are selected
        mask = np.logical_and(x > min(wav) + lowcut, x < rest)
        xmin = x[mask]
        for s in samples[np.random.randint(len(samples), size=size)]:
            gp.set_parameter_vector(s)
            if noisefit:
                mu = gp.predict(flux, x, return_cov=False, kernel=kernel1)
                mu_noise = gp.predict(flux, wav, return_cov=False, kernel=kernel2)
            else:
                mu = gp.predict(flux, x, return_cov=False, kernel=kernel)
                mu_noise = np.zeros_like(flux)
            try:
                minind = np.argmin(mu[mask])
                # Calculate absorption to emission line ratio
                if ae - feature:
                    mi = np.min(mu[mask])
                    ma = np.max(mu[np.logical_and(x > min(wav) + lowcut, x < cr_high)])
                    mins.append(mi)
                    maxs.append(ma)
                    ae = mi / ma
                    ae_avg.append(ae)
                minima.append(xmin[minind])
            except ValueError:
                warnings.warn("ValueError occured during minima detection. Skipping...")
            flux_pred.append(mu)
            avg_noise.append(mu_noise)

        # Calculate median and sigmas
        median = np.percentile(minima, 50)
        minustwosigma = np.percentile(minima, 2.28)
        minusonesigma = np.percentile(minima, 15.87)
        plusonesigma = np.percentile(minima, 84.13)
        plustwosigma = np.percentile(minima, 97.72)

        min_error_lower = median - minusonesigma
        min_error_upper = plusonesigma - median

        results = {
            "flux_pred": flux_pred,
            "avg_noise": avg_noise,
            "minima": minima,
            "mins": mins,
            "maxs": maxs,
            "ae_avg": ae_avg,
            "x": x,
            "median": median,
            "minustwosigma": minustwosigma,
            "minusonesigma": minusonesigma,
            "plusonesigma": plusonesigma,
            "plustwosigma": plustwosigma,
            "min_error_lower": min_error_lower,
            "min_error_upper": min_error_upper,
        }
        self.fits[line] = results

        print("Finished fitting %s line..." % line)

        if diagnostic:
            self.diagnostic_plot(line, diagnostic)

        return results

    def diagnostic_plot(self, line, save):
        """
        Creates a diagnostic plot of a line fit.

        Parameters
        ----------
        line : str
            Specifies the name of the line. Has to match the predefined
            line names, unless cr_low, cr_high and rest are specified.
        save : str or None
            Directory in which the figure is saved. If figure should not be saved,
            None or False can be passed.

        Returns
        -------
        fig : matplotlib figure object
            Diagnostic figure
        """

        # Check if specified line is defined
        if line in list(self.lines.keys()):
            cr_low = self.lines[line][0]
            cr_high = self.lines[line][1]
            rest = self.lines[line][2]
            ae_feature = self.lines[line][3]
        else:
            if cr_low is None or cr_high is None or rest is None:
                raise ValueError(
                    "Line not found in built in set and no range specifications given..."
                )
            else:
                # Add user defined line to line dict
                self.lines[line] = [cr_low, cr_high, rest, ae_feature]

        cutting_range = np.logical_and(self.wav > cr_low, self.wav < cr_high)

        # Unpack results
        flux_pred = self.fits[line]["flux_pred"]
        avg_noise = self.fits[line]["avg_noise"]
        minima = self.fits[line]["minima"]
        mins = self.fits[line]["mins"]
        maxs = self.fits[line]["maxs"]
        ae_avg = self.fits[line]["ae_avg"]
        x = self.fits[line]["x"]
        median = self.fits[line]["median"]
        minustwosigma = self.fits[line]["minustwosigma"]
        minusonesigma = self.fits[line]["minusonesigma"]
        plusonesigma = self.fits[line]["plusonesigma"]
        plustwosigma = self.fits[line]["plustwosigma"]
        min_error_lower = self.fits[line]["min_error_lower"]
        min_error_upper = self.fits[line]["min_error_upper"]

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=[19.2, 4.8])
        if ae_feature:
            ax1.hist(ae_avg, label="a/e ratios")
            ax1.axvspan(
                np.percentile(ae_avg, 2.28),
                np.percentile(ae_avg, 97.72),
                alpha=0.1,
                color="red",
                label="2$\sigma$ (95.44%)",
            )
            ax1.axvspan(
                np.percentile(ae_avg, 15.87),
                np.percentile(ae_avg, 84.13),
                alpha=0.3,
                color="red",
                label="1$\sigma$ (68.26%)",
            )
            ax1.axvline(np.percentile(ae_avg, 50), color="red", label="Median")

            ax1.set_xlabel("a/e")
            ax1.set_ylabel("Number of found a/e")

            ax1.set_title(
                r"$a/e = %.4f^{+%.4f}_{-%.4f}$"
                % (
                    np.around(np.percentile(ae_avg, 50), 4),
                    np.around(
                        np.percentile(ae_avg, 84.13) - np.percentile(ae_avg, 50), 4
                    ),
                    np.around(
                        np.percentile(ae_avg, 50) - np.percentile(ae_avg, 15.87), 4
                    ),
                )
            )
            ax1.legend()

            # Plot fit with error band for peak position
            ax2.errorbar(
                self.wav[cutting_range],
                self.flux[cutting_range],
                yerr=self.error[cutting_range],
                fmt=".k",
                capsize=0,
                label="Data",
                mec="none",
                alpha=0.3,
            )

            for ind in range(min([len(flux_pred), 50])):
                if ind == min([len(flux_pred), 50]) - 1:
                    ax2.plot(
                        x, flux_pred[ind], color="#4682b4", label="Flux prediction"
                    )
                    ax2.axhline(mins[ind], color="red", label="Absorbtion minima")
                    ax2.axhline(maxs[ind], color="tab:green", label="Emission maxima")
                else:
                    ax2.plot(x, flux_pred[ind], color="#4682b4", alpha=0.1)
                    ax2.axhline(mins[ind], color="red", alpha=0.1)
                    ax2.axhline(maxs[ind], color="tab:green", alpha=0.1)

            ax2.errorbar(
                self.wav[cutting_range],
                self.flux[cutting_range] - avg_noise,
                yerr=self.error,
                fmt=".k",
                capsize=0,
                label="Noise subtracted data",
            )
            ax2.axvline(
                min(self.wav[cutting_range]) + lowcut, color="k", ls="--", alpha=0.3
            )
            ax2.axvline(4861, color="k", ls="--", alpha=0.3)
            ax2.set_title(r"H$_\alpha$ line fit")
            ax2.set_xlabel("Wavelength ($\AA$)")
            ax2.set_ylabel("Flux (arb. unit)")
            ax2.legend()
            ax2.tight_layout()
            ax2.xlim([min(x), max(x)])
            ax2.grid(which="major")

            ax3.plot(
                self.wav[cutting_range],
                avg_noise,
                "k",
                label="Average Subtracted Noise",
            )
            ax3.legend()
            ax3.set_xlabel(r"Wavelength($\AA$)")
            ax3.set_ylabel("Flux (arb. unit)")
        else:
            ax1.hist(minima, label="Minima")
            ax1.axvspan(
                minustwosigma,
                plustwosigma,
                alpha=0.1,
                color="red",
                label=r"2$\sigma$ (95.44%)",
            )
            ax1.axvspan(
                minusonesigma,
                plusonesigma,
                alpha=0.3,
                color="red",
                label=r"1$\sigma$ (68.26%)",
            )
            ax1.axvline(median, color="red", label="Weighted Median")
            ax1.set_ylabel("Found minima")

            ax1.legend()
            axes2 = ax1.twiny()
            ax1_ticks = ax1.get_xticks()
            axes2_ticks = []
            for X in ax1_ticks:
                # Velocity in km/s
                vel_value = (
                    299792458 * (4861**2 - X**2) / (4861**2 + X**2) / 1000
                )
                axes2_ticks.append("%.0f" % vel_value)

            axes2.set_xticks(ax1_ticks)
            axes2.set_xbound(ax1.get_xbound())
            axes2.set_xticklabels(axes2_ticks)
            ax1.set_xlabel("Wavelength ($\AA$)")
            axes2.set_xlabel("Velocity (km/s)")

            # Plot fit with error band for peak position
            ax2.errorbar(
                self.wav[cutting_range],
                self.flux[cutting_range],
                yerr=self.error[cutting_range],
                fmt=".k",
                capsize=0,
                label="Data",
                mec="none",
                alpha=0.3,
            )

            for ind in range(min([len(flux_pred), 50])):
                ax2.plot(x, flux_pred[ind], color="#4682b4", alpha=0.1)
                ax2.axvline(minima[ind], color="red", alpha=0.1)
            ax2.errorbar(
                self.wav[cutting_range],
                self.flux[cutting_range] - avg_noise,
                yerr=self.error[cutting_range],
                fmt=".k",
                capsize=0,
                label="Denoised",
            )
            ax2.axvline(min(self.wav) + lowcut, color="k", ls="--", alpha=0.3)
            ax2.axvline(4861, color="k", ls="--", alpha=0.3)
            ax2.set_xlabel(r"Wavelength ($\AA$)")
            ax2.set_ylabel("Flux (arb. unit)")
            ax2.legend()
            ax2.tight_layout()
            ax2.set_xlim([min(x), max(x)])
            ax2.grid(which="major")

            ax2.set_title(
                "MinWavelength: {:.2f} +{:.2f}/ -{:.2f} $\AA$\n Velocity: {:.2f} +{:.2f}/ -{:.2f} km/s".format(
                    median,
                    min_error_upper,
                    min_error_lower,
                    velocity / 1000,
                    vel_err_upper / 1000,
                    vel_err_lower / 1000,
                )
            )

            ax3 = plt.subplot(133)
            ax3.plot(wavelength, avg_noise, "k", label="Average Subtracted Noise")
            ax3.legend()
            ax3.set_xlabel(r"Wavelength($\AA$)")
            ax3.set_ylabel("Flux (arb. unit)")

        plt.tight_layout()

        # Save figure
        if save:
            plt.savefig(
                os.path.join(save, "Fit_{:s}_{:s}.pdf".format(line, str(self.numcode))),
                bbox_inches="tight",
                dpi=100,
            )

        plt.close()

        return fig

    def get_results(self, line):
        """
        Function that returns either the velocities or a/e values for a specific lines

        Parameters
        ----------
        line : str
            Specifies the name of the line. Has to match the predefined
            line names, unless cr_low, cr_high and rest are specified.

        Returns
        -------
        velocity : float
        vel_err_lower : float
        vel_err_upper : float

            or

        ae : float
        ae_err_lower : float
        ae_err_upper : float
        """
        assert line in list(self.fist.keys()), "Specified line not found in results"

        ae_feature = self.lines[line][3]

        if ae_feature:
            ae_avg = self.fits[line]["ae_avg"]
            ae = np.percentile(ae_avg, 50)
            ae_err_lower = ae - np.percentile(ae_avg, 15.87)
            ae_err_upper = np.percentile(ae_avg, 84.13)

            return ae, ae_err_lower, ae_err_upper
        else:
            median = self.fits[line]["median"]
            min_error_lower = self.fits[line]["min_error_lower"]
            min_error_upper = self.fits[line]["min_error_upper"]

            # Conversion to velocity
            velocity = velocity_conversion(median, rest)

            # Calculate velocity error
            vel_err_lower = (
                4
                * 299792458
                * rest**2
                * median
                * min_error_lower
                / (rest**2 + median**2) ** 2
            )
            vel_err_upper = (
                4
                * 299792458
                * rest**2
                * median
                * min_error_upper
                / (rest**2 + median**2) ** 2
            )
            return velocity, vel_err_lower, vel_err_upper
