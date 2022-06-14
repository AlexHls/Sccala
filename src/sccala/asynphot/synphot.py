import os

import numpy as np
from scipy import interpolate
from scipy import integrate

from sccala.asynphot.calibration import get_vega_spectrum
from sccala.asynphot.io import load_transmission_data
from sccala.asynphot import integrate as err_integrate


def calculate_vega_zp(filter):

    c_AA = 299792458 * 1.0e10  # in AA/s
    h_erg = 6.62607015e-27  # Plancks constant (erg.s)

    vega_wav, vega_flux = get_vega_spectrum()

    return (
        2.5
        * np.log10(
            1
            / h_erg
            / c_AA
            * integrate.simpson(
                vega_flux * filter.interpolate(vega_wav) * vega_wav, vega_wav
            )
        )
        + 0.03
    )


def calculate_vega_magnitude(spec_wav, spec_flux, filter, spec_err=None):
    """
    Calculate Vega magnitudes from spectral flux. If spectral uncertainties are
    supplied, a magnitude uncertainty will be calculated by propagating the
    uncertainty through the numerical integration (Simpson's rule).

    :param spec_wav: np.array of floats
        wavelength values of spectrum
    :param spec_flux: np.array of floats
        spectral flux
    :param filter: BaseFilter
        filter object on which the magnitude is to be calculated
    :param spec_err: np.array of floats or None
        spectral uncertainty. Default=None
    :return vega_magnitude: np.array of floats
        calculated vega_magnitude
    :return vega_magnitude_error: np.array of floats, optional
        calculated magnitude uncertainty
    """

    h_erg = 6.62607015e-27  # Plancks constant (erg.s)
    c_AA = 299792458 * 1.0e10  # in AA/s

    vega_zp = calculate_vega_zp(filter)

    vega_magnitude = (
        -2.5
        * np.log10(
            1
            / h_erg
            / c_AA
            * integrate.simpson(
                spec_flux * filter.interpolate(spec_wav) * spec_wav, spec_wav
            )
        )
        + vega_zp
    )

    if spec_err is None:
        return vega_magnitude
    else:
        vega_magnitude_error = (
            2.5
            / np.log(10)
            / integrate.simpson(
                spec_flux * filter.interpolate(spec_wav) * spec_wav, spec_wav
            )
            * np.sqrt(
                err_integrate.mod_simpson(
                    (spec_err * filter.interpolate(spec_wav) * spec_wav) ** 2, spec_wav
                )
            )
        )
        return [vega_magnitude, vega_magnitude_error]


class BaseFilterCurve(object):
    @classmethod
    def load_filter(cls, filter_id=None, interpolation_kind="linear"):
        if filter_id is None:
            raise AttributeError("No filter specified...")
        else:
            filter = load_transmission_data(filter_id)
            wav, trans = filter["Wavelength"], filter["Transmission"]

        return cls(
            wav, trans, interpolation_kind=interpolation_kind, filter_id=filter_id
        )

    def __init__(self, wav, trans, interpolation_kind="linear", filter_id=None):
        self.wav = wav
        self.trans = trans

        self.interpolation_object = interpolate.interp1d(
            self.wav,
            self.trans,
            kind=interpolation_kind,
            bounds_error=False,
            fill_value=0.0,
        )
        self.filter_id = filter_id

    def interpolate(self, wavelength):
        return self.interpolation_object(wavelength)

    def calculate_vega_magnitude(self, spec_wav, spec_flux, spec_err=None):
        __doc__ = calculate_vega_magnitude.__doc__
        return calculate_vega_magnitude(spec_wav, spec_flux, self, spec_err=spec_err)


class FilterCurve(BaseFilterCurve):
    def __repr__(self):
        if self.filter_id is None:
            filter_id = "{0:x}".format(self.__hash__())
        else:
            filter_id = self.filter_id
        return "FilterCurve <{0}>".format(filter_id)


class FilterSet(object):
    def __init__(self, filter_set, interpolation_kind="linear"):

        if hasattr(filter_set[0], "wavelength"):
            self.filter_set = filter_set
        else:
            self.filter_set = [
                FilterCurve.load_filter(
                    filter_id, interpolation_kind=interpolation_kind
                )
                for filter_id in filter_set
            ]

    def __iter__(self):
        self.current_filter_idx = 0
        return self

    def __next__(self):
        try:
            item = self.filter_set[self.current_filter_idx]
        except IndexError:
            raise StopIteration

        self.current_filter_idx += 1
        return item

    next = __next__

    def __getitem__(self, item):
        return self.filter_set.__getitem__(item)

    def __repr__(self):
        return "<{0} \n{1}>".format(
            self.__class__.__name__,
            "\n".join([item.filter_id for item in self.filter_set]),
        )

    def calculate_vega_magnitudes(self, spec_wav, spec_flux, spec_err=None):
        if spec_err is None:
            mags = [
                item.calculate_vega_magnitude(spec_wav, spec_flux, spec_err=spec_err)
                for item in self.filter_set
            ]
            return mags
        else:
            mags, mags_err = np.array(
                [
                    item.calculate_vega_magnitude(
                        spec_wav, spec_flux, spec_err=spec_err
                    )
                    for item in self.filter_set
                ]
            ).T
            return mags, mags_err


class MagnitudeSet(FilterSet):
    def __init__(
        self,
        filter_set,
        magnitudes,
        magnitude_uncertainties=None,
        interpolation_kind="linear",
    ):
        super(MagnitudeSet, self).__init__(
            filter_set, interpolation_kind=interpolation_kind
        )
        self.magnitudes = np.array(magnitudes)
        if magnitude_uncertainties is not None:
            self.magnitude_uncertainties = np.array(magnitude_uncertainties)
        else:
            self.magnitude_uncertainties = None

    def __repr__(self):
        mag_str = "{0} {1:.4f} +/- {2:.4f}"
        mag_data = []
        for i, filter in enumerate(self.filter_set):
            unc = (
                np.nan
                if self.magnitude_uncertainties is None
                else self.magnitude_uncertainties[i]
            )
            mag_data.append(mag_str.format(filter.filter_id, self.magnitudes[i], unc))

        return "<{0} \n{1}>".format(self.__class__.__name__, "\n".join(mag_data))
