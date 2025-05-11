import os
import warnings

import numpy as np


from sccala.utillib.aux import NumpyEncoder
from sccala.utillib.const import VS_INIT, RV_INIT, VTRUE_INIT


class SCM_Model:
    def __init__(self):
        if self.__class__.__name__ == "SCM_Model":
            raise NotImplementedError("Cannot instantiate abstract class")
        if self.__name__ is None:
            self.__name__ = self.__class__.__name__
        self.model = self._load_model_from_file()
        self.init = {}
        self.data = {}
        self.hubble = False  # Specifies if model fits Hubble constant
        return

    @property
    def name(self):
        return self.__name__

    @property
    def file(self):
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "models",
            self.__name__ + ".stan",
        )

    def _load_model_from_file(self):
        try:
            with open(self.file) as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError("Model file not found")

    def get_data(self):
        return list(self.data.keys())

    def print_data(self):
        print(self.get_data())
        return

    def print_model(self):
        print(self.model)
        return

    def set_initial_conditions(self, init=None):
        pass

    def print_results(self, df, blind=True):
        for key in list(df.keys()):
            print("%s = %.2e +/- %.2e" % (np.mean(df[key][0]), np.std(df[key][0])))
        return

    def write_json(self, filename, path=""):
        try:
            import json
        except ImportError:
            print("json module not available")
            return

        with open(os.path.join(path, filename), "w") as f:
            json.dump(self.data, f, cls=NumpyEncoder)

        return os.path.join(path, filename)

    def write_stan(self, filename, path=""):
        ## Deprecated, raise warning
        warnings.warn(
            "write_stan is deprecated, it will be removed in future versions.",
            DeprecationWarning,
        )

        # Check if file exists and if the contents are identical
        # to the current to avoid re-compilation
        if os.path.exists(os.path.join(path, filename)):
            with open(os.path.join(path, filename), "r") as f:
                if f.read() == self.model:
                    print("Model already exists, skipping compilation...")
                    return os.path.join(path, filename)

        with open(os.path.join(path, filename), "w") as f:
            f.write(self.model)
        return os.path.join(path, filename)


class NHHubbleFreeSCM(SCM_Model):
    def __init__(self):
        self.__name__ = "nh-hubble-free-scm"
        super().__init__()

    def print_results(self, df, blind=True):
        keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        for key in keys:
            print("%s = %.2g +/- %.2g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class NHHubbleSCM(SCM_Model):
    def __init__(self):
        self.__name__ = "nh-hubble-scm"
        super().__init__()
        self.hubble = True

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class NHHubbleSCMSimple(SCM_Model):
    def __init__(self):
        self.__name__ = "nh-hubble-scm-simple"
        super().__init__()
        self.hubble = True

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class HubbleFreeSCM(SCM_Model):
    def __init__(self):
        self.__name__ = "hubble-free-scm"
        super().__init__()

    def set_initial_conditions(self, init=None):
        if init is None:
            self.init = {
                "vs": VS_INIT,
                "rv": RV_INIT,
                "v_true": [VTRUE_INIT] * self.data["sn_idx"],
            }
        else:
            self.init = init
        return

    def print_results(self, df, blind=True):
        keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        for key in keys:
            print("%s = %.2g +/- %.2g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class HubbleFreeSCMOutlier(SCM_Model):
    def __init__(self):
        self.__name__ = "hubble-free-scm-outlier"
        super().__init__()

    def set_initial_conditions(self, init=None):
        if init is None:
            self.init = {
                "vs": VS_INIT,
                "rv": RV_INIT,
                "v_true": [VTRUE_INIT] * self.data["sn_idx"],
            }
        else:
            self.init = init
        return

    def print_results(self, df, blind=True):
        keys = ["alpha", "beta", "gamma", "sigma_int", "Mi", "outl_frac"]
        for key in keys:
            print("%s = %.2g +/- %.2g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class HubbleSCM(SCM_Model):
    def __init__(self):
        self.__name__ = "hubble-scm"
        super().__init__()
        self.hubble = True

    def set_initial_conditions(self, init=None):
        if init is None:
            self.init = {
                "vs": VS_INIT,
                "rv": RV_INIT,
                "v_true": [VTRUE_INIT] * self.data["sn_idx"],
                "calib_v_true": [VTRUE_INIT] * self.data["calib_sn_idx"],
            }
            for i in range(self.data["num_calib_dset"]):
                self.init["calib_vs.%d" % (i + 1)] = VS_INIT
                self.init["calib_rv.%d" % (i + 1)] = RV_INIT
        else:
            self.init = init
        return

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class HubbleSCMOutlier(SCM_Model):
    def __init__(self):
        self.__name__ = "hubble-scm-outlier"
        super().__init__()
        self.hubble = True

    def set_initial_conditions(self, init=None):
        if init is None:
            self.init = {
                "vs": VS_INIT,
                "rv": RV_INIT,
                "v_true": [VTRUE_INIT] * self.data["sn_idx"],
                "calib_v_true": [VTRUE_INIT] * self.data["calib_sn_idx"],
            }
            for i in range(self.data["num_calib_dset"]):
                self.init["calib_vs.%d" % (i + 1)] = VS_INIT
                self.init["calib_rv.%d" % (i + 1)] = RV_INIT
        else:
            self.init = init
        return

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi", "outl_frac"]
        else:
            keys = ["alpha", "beta", "gamma", "sigma_int", "Mi", "H0", "outl_frac"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class ClassicNHHubbleFreeSCM(SCM_Model):
    def __init__(self):
        self.__name__ = "classic-nh-hubble-free-scm"
        super().__init__()

    def print_results(self, df, blind=True):
        keys = ["alpha", "beta", "sigma_int", "Mi"]
        for key in keys:
            print("%s = %.2g +/- %.2g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class ClassicNHHubbleSCM(SCM_Model):
    def __init__(self):
        self.__name__ = "classic-nh-hubble-scm"
        super().__init__()
        self.hubble = True

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class ClassicNHHubbleSCMSimple(SCM_Model):
    # This model resemples the model used by de Jaeger 2022
    def __init__(self):
        self.__name__ = "classic-nh-hubble-scm-simple"
        super().__init__()
        self.hubble = True

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class ClassicHubbleFreeSCM(SCM_Model):
    def __init__(self):
        self.__name__ = "classic-hubble-free-scm"
        super().__init__()

    def set_initial_conditions(self, init=None):
        if init is None:
            self.init = {
                "vs": VS_INIT,
                "rv": RV_INIT,
                "v_true": [VTRUE_INIT] * self.data["sn_idx"],
            }
        else:
            self.init = init
        return

    def print_results(self, df, blind=True):
        keys = ["alpha", "beta", "sigma_int", "Mi"]
        for key in keys:
            print("%s = %.2g +/- %.2g" % (key, np.mean(df[key]), np.std(df[key])))
        return


class ClassicHubbleSCM(SCM_Model):
    def __init__(self):
        self.__name__ = "classic-hubble-scm"
        super().__init__()
        self.hubble = True

    def set_initial_conditions(self, init=None):
        if init is None:
            self.init = {
                "vs": VS_INIT,
                "rv": RV_INIT,
                "v_true": [VTRUE_INIT] * self.data["sn_idx"],
                "calib_v_true": [VTRUE_INIT] * self.data["calib_sn_idx"],
            }
            for i in range(self.data["num_calib_dset"]):
                self.init["calib_vs.%d" % (i + 1)] = VS_INIT
                self.init["calib_rv.%d" % (i + 1)] = RV_INIT
        else:
            self.init = init
        return

    def print_results(self, df, blind=True):
        if blind:
            keys = ["alpha", "beta", "sigma_int", "Mi"]
        else:
            keys = ["alpha", "beta", "sigma_int", "Mi", "H0"]
        for key in keys:
            print("%s = %.4g +/- %.4g" % (key, np.mean(df[key]), np.std(df[key])))
        return
