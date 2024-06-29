import warnings

import numpy as np
import scipy.stats as st

import george
from george import kernels
from george.modeling import ConstantModel
from ultranest import ReactiveNestedSampler
from tqdm import tqdm


class Interpolator:
    def __init__():
        pass

    def get_toe_rnd(self, tkde, toe_samples, toe=0.0):
        if tkde is not None:
            return np.array(tkde.resample(size=toe_samples)[0])
        else:
            return np.ones(toe_samples) * toe

    def sample_posterior(self, num_live_points=800):
        self.sampler = ReactiveNestedSampler(
            self.parameters, self.log_likelihood, self.prior_transform
        )
        self.result = self.sampler.run(
            min_num_live_points=num_live_points, viz_callback=False, show_status=False
        )

        return self.sampler

    def log_likelihood(self, params):
        # Update the kernel and compute the log_likelihood.
        self.gp.set_parameter_vector(params)
        like = self.gp.log_likelihood(self.data, quiet=True)
        if np.isinf(like):
            return -1e100
        return like

    def reject_invalid(self, data_int, dint, no_reject):
        data_int.extend(dint)

    def check_sample_size(self, size, toe_samples):
        if size**2 < 1000:
            warnings.warn(
                "The sample size is too small, consider increasing the sample size."
            )
        if size % toe_samples != 0:
            warnings.warn(
                "The sample size is not divisible by the number of toe_samples."
            )

    def predict_from_posterior(
        self, t_pred, tkde=None, toe=0.0, size=100, no_reject=False, toe_samples=25
    ):
        # This makes sure that data_int.append() works as intended
        assert size > 1, "Insufficient size"
        assert size > toe_samples, "Insufficient size"
        self.check_sample_size(size, toe_samples)

        data_int = []

        subsample = np.random.randint(
            len(self.sampler.results["samples"][:, 0]), size=size
        )
        self._print_predicting()
        for s in tqdm(self.sampler.results["samples"][subsample]):
            self.gp.set_parameter_vector(s)
            toe_rnd = self.get_toe_rnd(tkde, toe_samples, toe)
            for t_rnd in toe_rnd:
                t = t_pred + (toe - t_rnd)
                dint = self.gp.sample_conditional(
                    self.data, t, size=int(size / toe_samples)
                )
            self.reject_invalid(data_int, dint, no_reject)

        return np.array(data_int)

    def _print_predicting(self):
        print("Predicting values...")


class LC_Interpolator(Interpolator):
    def __init__(
        self,
        data,
        data_error,
        t_grid,
        num_live_points=800,
        disable_mean_fit=False,
        disable_white_noise_fit=False,
    ):
        self.data = data
        self.data_error = data_error
        self.t_grid = t_grid
        self.num_live_points = num_live_points
        self.disable_mean_fit = disable_mean_fit
        self.disable_white_noise_fit = disable_white_noise_fit

        # Setup kernel
        kernel = np.var(data) * kernels.ExpSquaredKernel(25**2)
        self.parameters = []
        if not disable_mean_fit:
            self.parameters.append("mean")
            mean_model = ConstantModel(data.mean())
        else:
            mean_model = None
        if not disable_white_noise_fit:
            self.parameters.append("log_var_white")
            white_noise = np.log(0.01**2)
        else:
            white_noise = None
        self.parameters.extend(["log_var", "l"])
        self.gp = george.GP(
            kernel,
            mean=mean_model,
            fit_mean=(not disable_mean_fit),
            fit_white_noise=(not disable_white_noise_fit),
            white_noise=white_noise,
        )
        try:
            self.gp.compute(self.t_grid, self.data_error)
        except np.linalg.LinAlgError:
            warnings.warn("LinAlgError occured, modifying data_error...")
            self.gp.compute(self.t_grid, self.data_error * 2)

    def prior_transform(self, cube):
        # the argument, cube, consists of values from 0 to 1
        # we have to convert them to physical scales

        params = cube.copy()
        if self.disable_mean_fit and self.disable_white_noise_fit:
            params[0] = np.log(st.halfnorm.ppf(cube[0], scale=1.5) ** 2)
            params[1] = np.log(
                st.invgamma.ppf(cube[1], 4.62908952, scale=110.33659801) ** 2
            )
        elif self.disable_mean_fit:
            params[0] = np.log(st.loguniform.ppf(np.array(cube[0]), 1e-5, 0.2) * 2)
            params[1] = np.log(st.halfnorm.ppf(cube[1], scale=1.5) ** 2)
            params[2] = np.log(
                st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2
            )
        elif self.disable_white_noise_fit:
            params[0] = cube[0] * 20 + 10
            params[1] = np.log(st.halfnorm.ppf(cube[1], scale=1.5) ** 2)
            params[2] = np.log(
                st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2
            )
        else:
            # Mean prior -> Uniform prior from 10 to 30
            params[0] = cube[0] * 20 + 10
            # White noise prior -> log-Uniform prior
            params[1] = np.log(st.loguniform.ppf(np.array(cube[1]), 1e-5, 0.2) * 2)
            # Variance prior -> Half Gaussian prior
            params[2] = np.log(st.halfnorm.ppf(cube[2], scale=1.5) ** 2)
            # l prior -> InvGamma prior
            params[3] = np.log(
                st.invgamma.ppf(cube[3], 4.62908952, scale=110.33659801) ** 2
            )

        return params

    def _print_predicting(self):
        print("Predicting lightcurves...")


class AKS_Interpolator(Interpolator):
    def __init__(
        self,
        data,
        t_grid,
        uncertainty=0.004,
        num_live_points=400,
        disable_mean_fit=False,
    ):
        self.data = np.array(data)
        self.t_grid = np.array(t_grid)
        self.uncertainty = uncertainty
        self.num_live_points = num_live_points
        self.disable_mean_fit = disable_mean_fit

        # Setup kernel
        kernel = np.var(data) * kernels.ExpSquaredKernel(25**2)
        mean_model = ConstantModel(self.data.mean())
        self.parameters = []
        if not self.disable_mean_fit:
            self.parameters.append("mean")
            mean_model = ConstantModel(self.data.mean())
        else:
            mean_model = None
        self.parameters.extend(["log_var", "l"])
        self.gp = george.GP(
            kernel,
            mean=mean_model,
            fit_mean=(not self.disable_mean_fit),
        )
        self.gp.compute(self.t_grid, self.uncertainty)

    def prior_transform(self, cube):
        # the argument, cube, consists of values from 0 to 1
        # we have to convert them to physical scales

        params = cube.copy()
        if self.disable_mean_fit:
            params[0] = np.log(st.halfnorm.ppf(cube[1], scale=0.2) ** 2)
            params[1] = np.log(
                st.invgamma.ppf(cube[1], 4.62908952, scale=110.33659801) ** 2
            )
        else:
            params[0] = cube[0] * 6 - 3
            params[1] = np.log(st.halfnorm.ppf(cube[1], scale=0.2) ** 2)
            params[2] = np.log(
                st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2
            )

        return params

    def _print_predicting(self):
        print("Predicting AKS values...")

    def predict_from_posterior(self, t_pred, size=1000):
        data_int = []
        subsample = np.random.randint(
            len(self.sampler.results["samples"][:, 0]), size=size
        )
        self._print_predicting()
        for s in tqdm(self.sampler.results["samples"][subsample]):
            self.gp.set_parameter_vector(s)
            dint = self.gp.sample_conditional(self.data, t_pred, size=10)
            data_int.extend(dint)

        return np.array(data_int)


class Vel_Interpolator(Interpolator):
    def __init__(
        self,
        data,
        data_error,
        t_grid,
        num_live_points=800,
        disable_mean_fit=True,
        disable_white_noise_fit=True,
    ):
        self.data = data
        self.data_error = data_error
        self.t_grid = t_grid
        self.num_live_points = num_live_points
        self.disable_mean_fit = disable_mean_fit
        self.disable_white_noise_fit = disable_white_noise_fit

        # Setup kernel
        kernel = np.var(data) * kernels.ExpSquaredKernel(25**2)
        self.parameters = []
        if not disable_mean_fit:
            self.parameters.append("mean")
            mean_model = ConstantModel(data.mean())
        else:
            mean_model = None
        if not disable_white_noise_fit:
            self.parameters.append("log_var_white")
            white_noise = np.log(100e3**2)
        else:
            white_noise = None
        self.parameters.extend(["log_var", "l"])
        self.gp = george.GP(
            kernel,
            mean=mean_model,
            fit_mean=(not disable_mean_fit),
            fit_white_noise=(not disable_white_noise_fit),
            white_noise=white_noise,
        )
        try:
            self.gp.compute(self.t_grid, self.data_error)
        except np.linalg.LinAlgError:
            warnings.warn("LinAlgError occured, modifying data_error...")
            self.gp.compute(self.t_grid, self.data_error * 2)

    def prior_transform(self, cube):
        # the argument, cube, consists of values from 0 to 1
        # we have to convert them to physical scales

        params = cube.copy()
        if self.disable_mean_fit and self.disable_white_noise_fit:
            params[0] = np.log(st.halfnorm.ppf(cube[0], scale=2500e3) ** 2)
            params[1] = np.log(
                st.invgamma.ppf(cube[1], 4.62908952, scale=110.33659801) ** 2
            )
        elif self.disable_mean_fit:
            params[0] = np.log(st.loguniform.ppf(np.array(cube[0]), 10e3, 1000e3) * 2)
            params[1] = np.log(st.halfnorm.ppf(cube[1], scale=2500e3) ** 2)
            params[2] = np.log(
                st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2
            )
        elif self.disable_white_noise_fit:
            params[0] = cube[0] * 1000e4 + 1000e3
            params[1] = np.log(st.halfnorm.ppf(cube[1], scale=2500e3) ** 2)
            params[2] = np.log(
                st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2
            )
        else:
            # Mean prior -> Uniform prior
            params[0] = cube[0] * 1000e4 + 1000e3
            # White noise prior -> log-Uniform prior
            params[1] = np.log(st.loguniform.ppf(np.array(cube[1]), 10e3, 1000e3) * 2)
            # Variance prior -> Half Gaussian prior
            params[2] = np.log(st.halfnorm.ppf(cube[2], scale=2500e3) ** 2)
            # l prior -> InvGamma prior
            params[3] = np.log(
                st.invgamma.ppf(cube[3], 4.62908952, scale=110.33659801) ** 2
            )

        return params

    def reject_invalid(self, data_int, dint, no_reject):
        for d in dint:
            if any(np.sign(np.diff(d)) == 1) and not no_reject:
                continue
            else:
                data_int.append(d)

    def _print_predicting(self):
        print("Predicting velocities...")


class AE_Interpolator(Interpolator):
    def __init__(
        self,
        data,
        data_error,
        t_grid,
        num_live_points=800,
        disable_mean_fit=True,
        disable_white_noise_fit=True,
    ):
        self.data = data
        self.data_error = data_error
        self.t_grid = t_grid
        self.num_live_points = num_live_points
        self.disable_mean_fit = disable_mean_fit
        self.disable_white_noise_fit = disable_white_noise_fit
        if disable_mean_fit and not disable_white_noise_fit:
            warnings.warn(
                "It is not recommended to use noise fit"
                "without a mean fit for halpha-ae."
            )

        # Setup kernel
        kernel = np.var(data) * kernels.PolynomialKernel(log_sigma2=6.0, order=3)
        self.parameters = []
        if not disable_mean_fit:
            self.parameters.append("mean")
            mean_model = ConstantModel(data.mean())
        else:
            mean_model = None
        if not disable_white_noise_fit:
            self.parameters.append("log_var_white")
            white_noise = np.log(0.01**2)
        else:
            white_noise = None
        self.parameters.extend(["log_var", "log_sigma2"])
        self.gp = george.GP(
            kernel,
            mean=mean_model,
            fit_mean=(not disable_mean_fit),
            fit_white_noise=(not disable_white_noise_fit),
            white_noise=white_noise,
        )
        try:
            self.gp.compute(self.t_grid, self.data_error)
        except np.linalg.LinAlgError:
            warnings.warn("LinAlgError occured, modifying data_error...")
            self.gp.compute(self.t_grid, self.data_error * 2)

    def reject_invalid(self, data_int, dint, no_reject):
        for d in dint:
            if any(np.sign(d) < 0.0) and not no_reject:
                continue
            else:
                data_int.append(d)

    def prior_transform(self, cube):
        # the argument, cube, consists of values from 0 to 1
        # we have to convert them to physical scales

        params = cube.copy()
        if self.disable_mean_fit and self.disable_white_noise_fit:
            params[0] = np.log(st.halfnorm.ppf(cube[0], scale=0.5) ** 2)
            params[1] = np.log(
                st.invgamma.ppf(cube[1], 4.62908952, scale=110.33659801) ** 2
            )
        elif self.disable_mean_fit:
            params[0] = np.log(st.loguniform.ppf(np.array(cube[0]), 1e-6, 0.2) * 2)
            params[1] = np.log(st.halfnorm.ppf(cube[1], scale=0.5) ** 2)
            params[2] = np.log(
                st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2
            )
        elif self.disable_white_noise_fit:
            params[0] = cube[0] * 2.0
            params[1] = np.log(st.halfnorm.ppf(cube[1], scale=0.5) ** 2)
            params[2] = np.log(
                st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2
            )
        else:
            # Mean prior -> Uniform prior
            params[0] = cube[0] * 2.0
            # White noise prior -> log-Uniform prior
            params[1] = np.log(st.loguniform.ppf(np.array(cube[1]), 1e-6, 0.2) * 2)
            # Variance prior -> Half Gaussian prior
            params[2] = np.log(st.halfnorm.ppf(cube[2], scale=0.2) ** 2)
            # l prior -> InvGamma prior
            params[3] = np.log(
                st.invgamma.ppf(cube[3], 4.62908952, scale=110.33659801) ** 2
            )

        return params

    def _print_predicting(self):
        print("Predicting a/e values...")
