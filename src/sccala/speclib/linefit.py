import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import george
from george import kernels
from ultranest import ReactiveNestedSampler
from ultranest.plot import PredictionBand
from tqdm import tqdm

from sccala.utillib.aux import velocity_conversion


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
            "si-ii": [5900, 6450, 6355, False],
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

    def __set_builtin_lines__(self, line, cr_low, cr_high, rest, ae_feature):
        """
        Sets a new builtin line if it does not already exist
        Parameters
        ----------
        line : str
            Specifies the name of the line. Has to be in the list
            of existing lines
        cr_low : float
            Lower boundary of the line feature.
        cr_high : float
            Upper boundary of the line feature.
        rest : float
            Rest wavelength of the feature.
        ae_feature : bool
            Specifies if absorption to emission ratio is to be fit.
        """

        # Run some checks to make sure the input is sensible
        assert line not in self.lines.keys(), "Line exists already"
        assert cr_low < rest, "cr_low cannot be higher than rest"
        assert rest < cr_high, "cr_high cannot be lower than rest"
        assert type(ae_feature) is bool, "ae_feature has to be bool"

        self.lines[line] = [cr_low, cr_high, rest, ae_feature]

        print("Added %s line to lines" % line)

        return

    def __modify_builtin_lines__(self, line, **kwargs):
        """
        Modifies the builtin lines, e.g. changes the cr_ranges
        Parameters
        ----------
        line : str
            Specifies the name of the line. Has to be in the list
            of existing lines
        kwargs:
            cr_low : float
                Lower boundary of the line feature.
            cr_high : float
                Upper boundary of the line feature.
            rest : float
                Rest wavelength of the feature.
            ae_feature : bool
                Specifies if absorption to emission ratio is to be fit.
        """

        if line not in self.lines.keys():
            raise ValueError("Line not in list of existing lines.")

        for arg in kwargs:
            if arg == "cr_low":
                self.lines[line][0] = kwargs[arg]
                print("Modified cr_low for %s line" % line)
            elif arg == "cr_high":
                self.lines[line][1] = kwargs[arg]
                print("Modified cr_high for %s line" % line)
            elif arg == "rest":
                self.lines[line][2] = kwargs[arg]
                print("Modified rest for %s line" % line)
            elif arg == "ae_feature":
                self.lines[line][3] = kwargs[arg]
                print("Modified ae_feature for %s line" % line)

        return

    def __reset_builtin_lines__(self):
        """
        Resets builtin lines in case their properties have
        been overwritten
        """
        self.lines = {
            "halpha-ae": [6250, 6650, 6563, True],
            "hbeta": [4550, 4900, 4861, False],
            "si-ii": [5900, 6450, 6355, False],
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
        size=1000,
        diagnostic=None,
        num_live_points=800,
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
            Number of flux predictions to be drawn. Total number of samples
            drawn is <size>^2. Default: 100
        diagnostic : str
            Determines if diagnostic plots are to be saved in specified
            directory. Default: False
        num_live_points : float
            Number of live points used in sampling routine. Default: 800

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

        print("Fitting %s line..." % line)

        # Cut off any additional secondary feature (particularly for hbeta features)
        while True:
            cutting_range = np.logical_and(self.wav > cr_low, self.wav < cr_high)
            wav = self.wav[cutting_range]
            flux = self.flux[cutting_range]
            if self.error is not None:
                error = self.error[cutting_range]
            else:
                error = None
            try:
                if wav[np.argmin(flux)] < rest:
                    break
            except ValueError as e:
                warnings.warn(
                    "No feature could be identified (spectral range possibly insufficent)"
                )
                raise e
            cr_high -= 20

        self.lines[line][1] = cr_high

        # Determine scale lengths for Gaussian Process
        scale = max(wav) - min(wav)
        # Length scale of correlations spanning 5 bins
        corr_scale = np.mean(np.diff(wav))

        # Check that all data is defined correctly
        assert error is not None, "No error found..."
        assert size > 1, "Size needs to be larger than 1..."

        # Define kernels
        if noisefit:
            # Kernel1 for the actual long scale peak/ absorbtion minimum
            kernel1 = np.var(flux) * kernels.ExpSquaredKernel(scale)
            # Kernel2 for the (correlated) noise
            kernel2 = np.var(flux) * kernels.Matern32Kernel(corr_scale)
            kernel = kernel1 + kernel2
        else:
            kernel = np.var(flux) * kernels.ExpSquaredKernel(scale)
            kernel1 = None
            kernel2 = None

        # Set-up GP
        if hodlrsolver:
            solver = george.HODLRSolver
        else:
            solver = george.BasicSolver
        gp = george.GP(kernel, solver=solver)
        gp.compute(wav, error)
        if noisefit:
            gp_signal = george.GP(kernel1, solver=solver)
            gp_signal.compute(wav, error)
            parameters = ["log_var", "l", "log_var_corr", "l_corr"]
        else:
            gp_signal = george.GP(kernel, solver=solver)
            gp_signal.compute(wav, error)
            parameters = ["log_var", "l"]

        if noisefit:

            def prior_transform(cube):
                params = cube.copy()
                # Variance prior -> Half Gaussian prior
                params[0] = np.log(st.halfnorm.ppf(cube[0], scale=1) ** 2)
                # l prior -> InvGamma prior
                params[1] = np.log(
                    st.invgamma.ppf(
                        cube[1], 2.9385366780066273, scale=207.54330815610572
                    )
                    ** 2
                )
                # Variance prior -> Half Gaussian prior
                params[2] = np.log(st.halfnorm.ppf(cube[2], scale=0.1) ** 2)
                # l prior -> InvGamma prior
                params[3] = np.log(
                    st.loguniform.ppf(cube[3], corr_scale, 4 * corr_scale) * 2
                )
                return params

        else:

            def prior_transform(cube):
                params = cube.copy()
                # Variance prior -> Half Gaussian prior
                params[0] = np.log(st.halfnorm.ppf(cube[0], scale=1) ** 2)
                # l prior -> InvGamma prior
                params[1] = np.log(
                    st.invgamma.ppf(
                        cube[1], 2.9385366780066273, scale=207.54330815610572
                    )
                    ** 2
                )
                return params

        def log_likelihood(params):
            # Update the kernel and compute the lnlikelihood.
            gp.set_parameter_vector(params)
            return gp.log_likelihood(flux, quiet=True)

        sampler = ReactiveNestedSampler(parameters, log_likelihood, prior_transform)
        sampler.run(
            min_num_live_points=num_live_points, viz_callback=False, show_status=False
        )

        # The positions where the prediction should be computed.
        x = np.linspace(min(wav), max(wav), len(wav) * 10)

        # Draw from samples to get peak estimate
        minima = []
        fits = []
        fits_noise = []
        ae_avg = []
        mins = []
        maxs = []

        # Minima cutoff from wavelenth minimum
        lowcut = 30

        # Subsamples from which results are drawn
        subsample = np.random.randint(len(sampler.results["samples"][:, 0]), size=size)

        print("Predicting flux...")
        # Mask to ensure only minima in the middle region are selected
        mask = np.logical_and(x > min(wav) + lowcut, x < rest)
        xmin = x[mask]
        for s in tqdm(sampler.results["samples"][subsample]):
            gp.set_parameter_vector(s)
            if noisefit:
                gp_signal.set_parameter_vector(s[:2])
            else:
                gp_signal.set_parameter_vector(s)

            fit = gp_signal.sample_conditional(flux, x, size=10)
            if noisefit:
                fit_full = gp.sample_conditional(flux, x, size=10)
                fits_noise.extend(fit_full - fit)
            else:
                fits_noise.extend(np.zeros_like(fit))
            try:
                minind = np.argmin(fit[:, mask], axis=1)
                if len(minind) == 0:
                    continue
                # Calculate absorption to emission line ratio
                if ae_feature:
                    mi = np.min(fit[:, mask], axis=1)
                    ma = np.max(
                        fit[:, np.logical_and(x > min(wav) + lowcut, x < cr_high)],
                        axis=1,
                    )
                    mins.extend(mi)
                    maxs.extend(ma)
                    ae = mi / ma
                    ae_avg.extend(ae)
                for ind in minind:
                    minima.append(xmin[ind])
            except ValueError:
                warnings.warn("ValueError occured during minima detection. Skipping...")
            fits.extend(fit)

        # Calculate median and sigmas
        median = np.percentile(minima, 50)
        minustwosigma = np.percentile(minima, 2.28)
        minusonesigma = np.percentile(minima, 15.87)
        plusonesigma = np.percentile(minima, 84.13)
        plustwosigma = np.percentile(minima, 97.72)

        min_error_lower = median - minusonesigma
        min_error_upper = plusonesigma - median

        results = {
            "flux_pred": fits,
            "avg_noise": fits_noise,
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
            raise ValueError("Line not found in built in set...")

        lowcut = 30
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
            )

            plt.sca(ax=ax2)
            band = PredictionBand(x)
            for i in range(len(flux_pred)):
                band.add(flux_pred[i])
            for i in range(min([len(flux_pred), 50])):
                if i == min([len(flux_pred), 50]) - 1:
                    ax2.axhline(mins[i], color="red", label="Absorbtion minima")
                    ax2.axhline(maxs[i], color="tab:green", label="Emission maxima")
                else:
                    ax2.axhline(mins[i], color="red", alpha=0.1)
                    ax2.axhline(maxs[i], color="tab:green", alpha=0.1)
            band.line(color="#4682b4", label="Flux prediction")
            band.shade(color="#4682b4", alpha=0.3)
            band.shade(q=0.49, color="#4682b4", alpha=0.2)

            ax2.axvline(
                min(self.wav[cutting_range]) + lowcut, color="k", ls="--", alpha=0.3
            )
            ax2.axvline(4861, color="k", ls="--", alpha=0.3)
            ax2.set_title(r"H$_\alpha$ line fit")
            ax2.set_xlabel("Wavelength ($\AA$)")
            ax2.set_ylabel("Flux (arb. unit)")
            ax2.legend()
            ax2.set_xlim([min(x), max(x)])
            ax2.grid(which="major")

            velocity, vel_err_lower, vel_err_upper = self.get_results(line)
            ax2.set_title(
                "MinWavelength: {:.2f} +{:.2f}/ -{:.2f} $\AA$\n a/e: {:.2e} +{:.2e}/ -{:.2e}".format(
                    median,
                    min_error_upper,
                    min_error_lower,
                    velocity,
                    vel_err_upper,
                    vel_err_lower,
                )
            )

            plt.sca(ax=ax3)
            band_noise = PredictionBand(x)
            for i in range(len(avg_noise)):
                band_noise.add(avg_noise[i])
            band_noise.line(color="k", label="Average Subtracted Noise")
            band_noise.shade(color="k", alpha=0.3)
            band_noise.shade(q=0.49, color="k", alpha=0.2)
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
            ax1.axvline(median, color="red", label="Median")
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
            )

            plt.sca(ax=ax2)
            band = PredictionBand(x)
            for i in range(len(flux_pred)):
                band.add(flux_pred[i])
            for ind in range(min([len(flux_pred), 50])):
                ax2.axvline(minima[ind], color="red", alpha=0.1)
            band.line(color="#4682b4", label="Flux prediction")
            band.shade(color="#4682b4", alpha=0.3)
            band.shade(q=0.49, color="#4682b4", alpha=0.2)

            ax2.axvline(min(self.wav) + lowcut, color="k", ls="--", alpha=0.3)
            ax2.axvline(4861, color="k", ls="--", alpha=0.3)
            ax2.set_xlabel(r"Wavelength ($\AA$)")
            ax2.set_ylabel("Flux (arb. unit)")
            ax2.legend()
            ax2.set_xlim([min(x), max(x)])
            ax2.grid(which="major")

            velocity, vel_err_lower, vel_err_upper = self.get_results(line)

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

            plt.sca(ax=ax3)
            band_noise = PredictionBand(x)
            for i in range(len(avg_noise)):
                band_noise.add(avg_noise[i])
            band_noise.line(color="k", label="Average Subtracted Noise")
            band_noise.shade(color="k", alpha=0.3)
            band_noise.shade(q=0.49, color="k", alpha=0.2)
            ax3.legend()
            ax3.set_xlabel(r"Wavelength($\AA$)")
            ax3.set_ylabel("Flux (arb. unit)")

        plt.tight_layout()

        # Save figure
        if not os.path.exists(save):
            os.makedirs(save)
        if save:
            plt.savefig(
                os.path.join(save, "Fit_{:s}_{:s}.png".format(line, str(self.numcode))),
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
        assert line in list(self.lines.keys()), "Specified line not found in results"

        if line in list(self.lines.keys()):
            cr_low = self.lines[line][0]
            cr_high = self.lines[line][1]
            rest = self.lines[line][2]
            ae_feature = self.lines[line][3]
        else:
            raise ValueError("Line not found in built in set...")

        if ae_feature:
            ae_avg = self.fits[line]["ae_avg"]
            ae = np.percentile(ae_avg, 50)
            ae_err_lower = ae - np.percentile(ae_avg, 15.87)
            ae_err_upper = np.percentile(ae_avg, 84.13) - ae

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
