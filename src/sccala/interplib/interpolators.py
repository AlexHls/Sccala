import numpy as np
from numpy.random import default_rng
import scipy.stats as st

import george
from george import kernels
from george.modeling import ConstantModel
from ultranest import ReactiveNestedSampler
from tqdm import tqdm


class LC_Interpolator:
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
            white_noise = np.log(0.01 ** 2)
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
        self.gp.compute(self.t_grid, self.data_error)

    def log_likelihood(self, params):
        # Update the kernel and compute the log_likelihood.
        self.gp.set_parameter_vector(params)

        return self.gp.log_likelihood(self.data, quiet=True)

    def prior_transform(self, cube):
        # the argument, cube, consists of values from 0 to 1
        # we have to convert them to physical scales

        params = cube.copy()
        if self.disable_mean_fit and self.disable_white_noise_fit:
            params[0] = np.log(st.halfnorm.ppf(cube[0], scale=1.5) ** 2)
            params[1] = np.log(st.invgamma.ppf(cube[1], 4.62908952, scale=110.33659801) ** 2)
        elif self.disable_mean_fit:
            params[0] = np.log(st.loguniform.ppf(np.array(cube[0]), 1e-5, 0.2) * 2)
            params[1] = np.log(st.halfnorm.ppf(cube[1], scale=1.5) ** 2)
            params[2] = np.log(st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2)
        elif self.disable_white_noise_fit:
            params[0] = cube[0] * 20 + 10
            params[1] = np.log(st.halfnorm.ppf(cube[1], scale=1.5) ** 2)
            params[2] = np.log(st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2)
        else:
            # Mean prior -> Uniform prior from 10 to 30
            params[0] = cube[0] * 20 + 10
            # White noise prior -> log-Uniform prior
            params[1] = np.log(st.loguniform.ppf(np.array(cube[1]), 1e-5, 0.2) * 2)
            # Variance prior -> Half Gaussian prior
            params[2] = np.log(st.halfnorm.ppf(cube[2], scale=1.5) ** 2)
            # l prior -> InvGamma prior
            params[3] = np.log(st.invgamma.ppf(cube[3], 4.62908952, scale=110.33659801) ** 2)
        
        return params

    def sample_posterior(self, num_live_points=800):
        self.sampler = ReactiveNestedSampler(
            self.parameters, self.log_likelihood, self.prior_transform
        )
        self.result = self.sampler.run(
            min_num_live_points=num_live_points, viz_callback=False, show_status=False
        )

        return self.sampler

    def predict_from_posterior(self, t_pred, tkde=None, toe=0.0, size=50):

        # This makes sure that data_int.extend() works as intended
        assert size > 1, "Insufficient size"

        data_int = []

        # Draw time from toe prior
        rng = default_rng()
        uni_rng = rng.uniform(size=size)

        subsample = np.random.randint(len(self.sampler.results["samples"][:,0]), size=size)
        print("Predicting lightcurves...")
        for s in tqdm(self.sampler.results["samples"][subsample]):
            self.gp.set_parameter_vector(s)
            if tkde is not None:
                for num in uni_rng:
                    toe_rnd = np.percentile(tkde, num * 100)
                    t = t_pred + (toe - toe_rnd)
                    dint = self.gp.sample_conditional(self.data, t, size=size)
                    data_int.extend(dint)
            else:
                dint = self.gp.sample_conditional(self.data, t_pred, size=size)
                data_int.extend(dint)

        return np.array(data_int)


class Vel_Interpolator:
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

        # Setup kernel
        kernel = np.var(data) * kernels.ExpSquaredKernel(25**2)
        self.parameters = ["mean", "log_var", "l"]
        self.gp = george.GP(
            kernel,
            mean=ConstantModel(data.mean()),
            fit_mean=True,
            # fit_white_noise=True,
            # white_noise=np.log(0.01 ** 2)
        )
        self.gp.compute(self.t_grid, self.data_error)

    def log_likelihood(self, params):
        # Update the kernel and compute the log_likelihood.
        self.gp.set_parameter_vector(params)

        return self.gp.log_likelihood(self.data, quiet=True)

    def prior_transform(self, cube):
        # the argument, cube, consists of values from 0 to 1
        # we have to convert them to physical scales

        params = cube.copy()
        # Mean prior -> Uniform prior from 10 to 30
        params[0] = cube[0] * 1000e4 + 1000e3
        # Variance prior -> Half Gaussian prior
        params[1] = np.log(st.halfnorm.ppf(cube[1], scale=1000e3) ** 2)
        # l prior -> InvGamma prior
        params[2] = np.log(
            st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2
        )

        return params

    def sample_posterior(self, num_live_points=800):
        self.sampler = ReactiveNestedSampler(
            self.parameters, self.log_likelihood, self.prior_transform
        )
        self.result = self.sampler.run(
            min_num_live_points=num_live_points, viz_callback=False, show_status=False
        )

        return self.sampler

    def predict_from_posterior(self, t_pred, tkde=None, toe=0.0, size=50):

        # This makes sure that data_int.extend() works as intended
        assert size > 1, "Insufficient size"

        data_int = []

        # Draw time from toe prior
        rng = default_rng()
        uni_rng = rng.uniform(size=size)

        subsample = np.random.randint(len(self.sampler.results["samples"][:,0]), size=size)
        print("Predicting lightcurves...")
        for s in tqdm(self.sampler.results["samples"][subsample]):
            self.gp.set_parameter_vector(s)
            if tkde is not None:
                for num in uni_rng:
                    toe_rnd = np.percentile(tkde, num * 100)
                    t = t_pred + (toe - toe_rnd)
                    dint = self.gp.sample_conditional(self.data, t, size=size)
                    data_int.extend(dint)
            else:
                dint = self.gp.sample_conditional(self.data, t_pred, size=size)
                data_int.extend(dint)

        return np.array(data_int)


class AE_Interpolator:
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

        # Setup kernel
        kernel = np.var(self.data) * kernels.PolynomialKernel(
            log_sigma2=np.var(self.data), order=3
        )
        self.parameters = ["mean", "log_var", "log_sigma2"]
        self.gp = george.GP(
            kernel,
            mean=ConstantModel(data.mean()),
            fit_mean=True,
            # fit_white_noise=True,
            # white_noise=np.log(0.01 ** 2)
        )
        self.gp.compute(self.t_grid, self.data_error)
        print(self.gp.get_parameter_names())

    def log_likelihood(self, params):
        # Update the kernel and compute the log_likelihood.
        self.gp.set_parameter_vector(params)

        return self.gp.log_likelihood(self.data, quiet=True)

    def prior_transform(self, cube):
        # the argument, cube, consists of values from 0 to 1
        # we have to convert them to physical scales

        params = cube.copy()
        # Mean prior -> Uniform prior from 10 to 30
        params[0] = 0.5 * cube[0] + 0.1
        # Variance prior -> Half Gaussian prior
        params[1] = np.log(st.halfnorm.ppf(cube[1], scale=0.5) ** 2)
        # l prior -> InvGamma prior
        #params[2] = np.log(
        #    st.invgamma.ppf(cube[2], 4.62908952, scale=110.33659801) ** 2
        #)
        params[2] = np.log(st.norm.ppf(cube[2]) ** 2)

        return params

    def sample_posterior(self, num_live_points=800):
        self.sampler = ReactiveNestedSampler(
            self.parameters, self.log_likelihood, self.prior_transform
        )
        self.result = self.sampler.run(
            min_num_live_points=num_live_points, viz_callback=False, show_status=False
        )

        return self.sampler

    def predict_from_posterior(self, t_pred, tkde=None, toe=0.0, size=50):

        # This makes sure that data_int.extend() works as intended
        assert size > 1, "Insufficient size"

        data_int = []

        # Draw time from toe prior
        rng = default_rng()
        uni_rng = rng.uniform(size=size)

        subsample = np.random.randint(len(self.sampler.results["samples"][:,0]), size=size)
        print("Predicting lightcurves...")
        for s in tqdm(self.sampler.results["samples"][subsample]):
            self.gp.set_parameter_vector(s)
            if tkde is not None:
                for num in uni_rng:
                    toe_rnd = np.percentile(tkde, num * 100)
                    t = t_pred + (toe - toe_rnd)
                    dint = self.gp.sample_conditional(self.data, t, size=size)
                    data_int.extend(dint)
            else:
                dint = self.gp.sample_conditional(self.data, t_pred, size=size)
                data_int.extend(dint)

        return np.array(data_int)
