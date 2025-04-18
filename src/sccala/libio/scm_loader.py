import os
import warnings
import glob

import pandas as pd
import numpy as np

from sccala.libio import get_paths as pa
from sccala.utillib.aux import calc_single_error


def load_data(
    sne_list,
    date,
    mag="I",
    col=("V", "I"),
    calib_sne_list=None,
    calib_date=None,
    calib_mag="I",
    calib_col=("V", "I"),
    instrument="Bessell12",
    export=False,
    mag_sys=None,
    vel_sys=None,
    col_sys=None,
    ae_sys=None,
    rho=1.0,
    rho_calib=0.0,
    error_mode="mean",
    m_cut_nom=18.5,
    sig_cut_nom=0.5,
    pv_red_file=None,
):
    """
    Loads all the necessary data for the scm and bundles it
    into a pandas DataFrame

    Parameters
    ----------
    sne_list : str or list
        Which SNe to load. Can either be a string, list of string or
        a filename where a detailed list is stored. If 'all' is passed,
        all SNe found in the data directory will be loaded.
    mag : str
        Filterband which is used as magnitude. Will be ignored for input if
        found in sne file and only used for export. Default: "I"
    col : tuple
        Filterbands to be used for colors. Needs to be given as (a, b) where
        the color is calculated as 'a-b'. Will be ignored for input if
        found in sne file and only used for export. Default: ("V", "I")
    date : float
        Epoch at which data is loaded. If date column exists in sne file,
        input will be ignored.
    calib_sne_list : str or list
        Same as sne, but for calibrators (optional).
    calib_mag : str
        Same as sne, but for calibrators (optional).
    calib_col : tuple
        Same as sne, but for calibrators (optional).
    calib_date : float
        Same as sne, but for calibrators (optional).
    instrument : str
        Instrument system of the photometry. If multiple systems are used, this
        has to be specified via the sne and calib_sne file
    export : bool or str
        Specifies if DataFrame is to be exported. If True is passed,
        exported file will be stored in results directory. If str is passed,
        file will be stored in specified location. Default: False
    mag_sys : float or None
        Systematic magnitude uncertainty added to all SNe. Default: None
    vel_sys : float or None
        Systematic velocity uncertainty added to all SNe. Default: None
    col_sys : float or None
        Systematic color uncertainty added to all SNe. Default: None
    ae_sys : float or None
        Systematic ae uncertainty added to all SNe. Default: None
    rho : float
        Correlation between the color and magnitude uncertainties. Default: 1.0
    rho_calib : float
        Correlation between the color and magnitude uncertainties for calibrator SNe. Default: 0.0
    error_mode : str
        Mode used to calculate single value error from asymmetric errors. Default: "mean"
    m_cut_nom : float
    sig_cut_nom : float
        Nominal values for the magnitude cut and its uncertainty. Default: 18.5, 0.5
    pv_red_file : str
        Path to a file containing the peculiar velocity corrected redshifts of the SNe.
        If None, redshifts are taken from the info file. Default: None

    Returns
    -------
    scm_data : pd.DataFrame
        DataFrame containing all the data necessary for the standardisation
    """

    assert error_mode in ["mean", "max", "min"], "Invalid error mode"

    datadict = {
        "SN": [],
        "dataset": [],
        "mag": [],
        "mag_err": [],
        "mag_sys": [],
        "col": [],
        "col_err": [],
        "col_sys": [],
        "vel": [],
        "vel_err": [],
        "vel_sys": [],
        "ae": [],
        "ae_err": [],
        "ae_sys": [],
        "red": [],
        "red_err": [],
        "epoch": [],
    }

    if pv_red_file is not None:
        pv_red = pd.read_csv(pv_red_file)

    # Check format of sne input
    if not isinstance(sne_list, list) or len(sne_list) == 1:
        if isinstance(sne_list, list):
            sne_list = sne_list[0]
        if os.path.exists(sne_list):
            df = pd.read_csv(sne_list)
            sne = df["SN"].tolist()
            if "mag" in df.columns:
                mag = df["mag"].tolist()
            else:
                mag = [mag] * len(sne)
            if "col0" in df.columns and "col1" in df.columns:
                col = list(zip(df["col0"].tolist(), df["col1"].tolist()))
            else:
                col = [col] * len(sne)
            if "date" in df.columns:
                date = df["date"].tolist()
            else:
                date = [date] * len(sne)
            if "dataset" in df.columns:
                dataset = df["dataset"].tolist()
            else:
                dataset = None
            if "instrument" in df.columns:
                instrument = df["instrument"].tolist()
            else:
                instrument = [instrument] * len(sne)
        else:
            # Turn input into list of only SN name was passed
            sne = [sne]
            mag = [mag]
            col = [col]
            date = [date]
            dataset = None
    elif sne == "all":
        sne = glob.glob(os.path.join(pa.get_data_path(), "*/"))
        mag = [mag] * len(sne)
        col = [col] * len(sne)
        date = [date] * len(sne)
        instrument = [instrument] * len(sne)
        dataset = None
    else:
        mag = [mag] * len(sne)
        col = [col] * len(sne)
        date = [date] * len(sne)
        instrument = [instrument] * len(sne)
        dataset = None

    # Check format of calib_sne input
    if calib_sne_list is not None:
        if not isinstance(calib_sne_list, list) or len(calib_sne_list) == 1:
            if isinstance(calib_sne_list, list):
                calib_sne_list = calib_sne_list[0]
            if os.path.exists(calib_sne_list):
                df = pd.read_csv(calib_sne_list)
                calib_sne = df["SN"].tolist()
                if "mag" in df.columns:
                    calib_mag = df["mag"].tolist()
                else:
                    calib_mag = [calib_mag] * len(calib_sne)
                if "col0" in df.columns and "col1" in df.columns:
                    calib_col = list(zip(df["col0"].tolist(), df["col1"].tolist()))
                else:
                    calib_col = [calib_col] * len(calib_sne)
                if "date" in df.columns:
                    calib_date = df["date"].tolist()
                else:
                    calib_date = [calib_date] * len(calib_sne)
                if "dataset" in df.columns:
                    calib_dataset = df["dataset"].tolist()
                else:
                    calib_dataset = None
                if "instrument" in df.columns:
                    calib_instrument = df["instrument"].tolist()
                else:
                    calib_instrument = [instrument] * len(calib_sne)
            else:
                # Turn input into list of only SN name was passed
                calib_sne = [calib_sne_list]
                calib_mag = [calib_mag]
                calib_col = [calib_col]
                calib_date = [calib_date]
                calib_instrument = [instrument]
                calib_dataset = None
        elif calib_sne_list == "all":
            raise ValueError("Calibrators need to be specified manually")
        else:
            calib_dataset = None
            calib_mag = [calib_mag] * len(calib_sne_list)
            calib_col = [calib_col] * len(calib_sne_list)
            calib_date = [calib_date] * len(calib_sne_list)
            calib_instrument = [instrument] * len(calib_sne_list)

    for i, sn in enumerate(sne):
        # Check if sn refers is a path to a SN or just a SN name
        if os.path.exists(sn):
            datapath = sn
            datadict["SN"].append(os.path.split(os.path.dirname(sn))[-1])
        else:
            datapath = os.path.join(pa.get_data_path(), sn)
            datadict["SN"].append(sn)

        # Load info file
        info = pd.read_csv(os.path.join(datapath, "{:s}_info.csv".format(sn)))
        red = np.mean(info["Redshift"].to_numpy())
        red_err = np.mean(info["Redshift_Error"])

        # If SN is in peculiar velocity redshift file, use that redshift
        if pv_red_file is not None:
            if sn in pv_red["SN"].tolist():
                red = pv_red[pv_red["SN"] == sn]["Redshift"].to_numpy()[0]
            else:
                warnings.warn("No peculiar velocity redshift found for %s" % sn)

        datadict["red"].append(red)
        datadict["red_err"].append(red_err)
        if dataset is None:
            if "Dataset" in df.columns:
                datasets = list(set(info["Dataset"].tolist()))
                # Check if SN consists of more than one dataset
                if len(datasets) > 1:
                    warnings.warn(
                        "More than one dataset found for %s, can only use one." % sn
                    )
                dset = datasets[0]
                datadict["dataset"].append(dset)
            else:
                datadict["dataset"].append("UNKNOWN")
        else:
            datadict["dataset"].append(dataset[i])

        # Load results
        respath = pa.get_res_path()
        # Magnitudes
        df = pd.read_csv(
            os.path.join(
                respath,
                "%s_%s_%s_InterpolationResults.csv" % (sn, instrument[i], mag[i]),
            )
        )
        try:
            mags = df[df["Date"] == date[i]][mag[i]].to_numpy()[0]
        except IndexError:
            raise ValueError(
                "No data found for %s at date %f in %s" % (sn, date[i], mag[i])
            )
        mags_err_lower = df[df["Date"] == date[i]]["%s_err_lower" % mag[i]].to_numpy()[
            0
        ]
        mags_err_upper = df[df["Date"] == date[i]]["%s_err_upper" % mag[i]].to_numpy()[
            0
        ]
        datadict["mag"].append(mags)
        datadict["mag_err"].append(
            calc_single_error(mags_err_lower, mags_err_upper, error_mode)
        )

        # Color 0
        df = pd.read_csv(
            os.path.join(
                respath,
                "%s_%s_%s_InterpolationResults.csv" % (sn, instrument[i], col[i][0]),
            )
        )
        col0 = df[df["Date"] == date[i]][col[i][0]].to_numpy()[0]
        col0_err_lower = df[df["Date"] == date[i]][
            "%s_err_lower" % col[i][0]
        ].to_numpy()[0]
        col0_err_upper = df[df["Date"] == date[i]][
            "%s_err_upper" % col[i][0]
        ].to_numpy()[0]
        col0_err = calc_single_error(col0_err_lower, col0_err_upper, error_mode)

        # Color 1
        df = pd.read_csv(
            os.path.join(
                respath,
                "%s_%s_%s_InterpolationResults.csv" % (sn, instrument[i], col[i][1]),
            )
        )
        col1 = df[df["Date"] == date[i]][col[i][1]].to_numpy()[0]
        col1_err_lower = df[df["Date"] == date[i]][
            "%s_err_lower" % col[i][1]
        ].to_numpy()[0]
        col1_err_upper = df[df["Date"] == date[i]][
            "%s_err_upper" % col[i][1]
        ].to_numpy()[0]
        col1_err = calc_single_error(col1_err_lower, col1_err_upper, error_mode)

        # Color
        datadict["col"].append(col0 - col1)
        datadict["col_err"].append(
            np.max([
                np.sqrt(col0_err**2 + col1_err**2 - 2 * rho * col0_err * col1_err),
                0.001,
            ])
        )

        # Velocity
        df = pd.read_csv(
            os.path.join(
                respath,
                "%s_hbeta_InterpolationResults.csv"
                % (sn),  # TODO Allow for other lines
            )
        )
        vel = df[df["Date"] == date[i]]["VelInt"].to_numpy()[0]
        vel_err_lower = df[df["Date"] == date[i]]["ErrorLower"].to_numpy()[0]
        vel_err_upper = df[df["Date"] == date[i]]["ErrorUpper"].to_numpy()[0]
        vel_err = calc_single_error(vel_err_lower, vel_err_upper, error_mode)
        datadict["vel"].append(vel)
        datadict["vel_err"].append(vel_err)

        df = pd.read_csv(
            os.path.join(
                respath,
                "%s_halpha-ae_InterpolationResults.csv"
                % (sn),  # TODO Allow for other lines
            )
        )
        ae = df[df["Date"] == date[i]]["VelInt"].to_numpy()[0]
        ae_err_lower = df[df["Date"] == date[i]]["ErrorLower"].to_numpy()[0]
        ae_err_upper = df[df["Date"] == date[i]]["ErrorUpper"].to_numpy()[0]
        ae_err = calc_single_error(ae_err_lower, ae_err_upper, error_mode)
        datadict["ae"].append(ae)
        datadict["ae_err"].append(ae_err)

        # Epoch
        datadict["epoch"].append(date[i])

    if calib_sne_list is not None:
        # Create new distance modulus columns and fill them with 0 values
        datadict["mu"] = [0] * len(datadict["SN"])
        datadict["mu_err"] = [0] * len(datadict["SN"])

        for i, sn in enumerate(calib_sne):
            # Check if sn refers is a path to a SN or just a SN name
            if os.path.exists(sn):
                datapath = sn
                datadict["SN"].append(os.path.split(os.path.dirname(sn))[-1])
            else:
                datapath = os.path.join(pa.get_data_path(), sn)
                datadict["SN"].append(sn)

            # Load info file
            info = pd.read_csv(os.path.join(datapath, "{:s}_info.csv".format(sn)))
            red = np.mean(info["Redshift"].to_numpy())
            red_err = np.mean(info["Redshift_Error"])
            datadict["red"].append(red)
            datadict["red_err"].append(red_err)
            if calib_dataset is None:
                if "Dataset" in df.columns:
                    datasets = list(set(info["Dataset"].tolist()))
                    # Check if SN consists of more than one dataset
                    if len(datasets) > 1:
                        warnings.warn(
                            "More than one dataset found for %s, can only use one." % sn
                        )
                    dset = datasets[0]
                    datadict["dataset"].append("CALIB_%s" % dset)
                else:
                    datadict["dataset"].append("CALIB_UNKNOWN")
            else:
                datadict["dataset"].append("CALIB_%s" % calib_dataset[i])

            # Load results
            respath = pa.get_res_path()
            # Magnitudes
            df = pd.read_csv(
                os.path.join(
                    respath,
                    "%s_%s_%s_InterpolationResults.csv"
                    % (sn, calib_instrument[i], calib_mag[i]),
                )
            )
            mags = df[df["Date"] == calib_date[i]][calib_mag[i]].to_numpy()[0]
            mags_err_lower = df[df["Date"] == calib_date[i]][
                "%s_err_lower" % calib_mag[i]
            ].to_numpy()[0]
            mags_err_upper = df[df["Date"] == calib_date[i]][
                "%s_err_upper" % calib_mag[i]
            ].to_numpy()[0]
            datadict["mag"].append(mags)
            datadict["mag_err"].append(
                calc_single_error(mags_err_lower, mags_err_upper, error_mode)
            )

            # Color 0
            df = pd.read_csv(
                os.path.join(
                    respath,
                    "%s_%s_%s_InterpolationResults.csv"
                    % (sn, calib_instrument[i], calib_col[i][0]),
                )
            )
            col0 = df[df["Date"] == calib_date[i]][calib_col[i][0]].to_numpy()[0]
            col0_err_lower = df[df["Date"] == calib_date[i]][
                "%s_err_lower" % calib_col[i][0]
            ].to_numpy()[0]
            col0_err_upper = df[df["Date"] == calib_date[i]][
                "%s_err_upper" % calib_col[i][0]
            ].to_numpy()[0]
            col0_err = calc_single_error(col0_err_lower, col0_err_upper, error_mode)

            # Color 1
            df = pd.read_csv(
                os.path.join(
                    respath,
                    "%s_%s_%s_InterpolationResults.csv"
                    % (sn, calib_instrument[i], calib_col[i][1]),
                )
            )
            col1 = df[df["Date"] == calib_date[i]][calib_col[i][1]].to_numpy()[0]
            col1_err_lower = df[df["Date"] == calib_date[i]][
                "%s_err_lower" % calib_col[i][1]
            ].to_numpy()[0]
            col1_err_upper = df[df["Date"] == calib_date[i]][
                "%s_err_upper" % calib_col[i][1]
            ].to_numpy()[0]
            col1_err = calc_single_error(col1_err_lower, col1_err_upper, error_mode)

            # Color
            datadict["col"].append(col0 - col1)
            datadict["col_err"].append(
                np.max([
                    np.sqrt(
                        col0_err**2 + col1_err**2 - 2 * rho_calib * col0_err * col1_err
                    ),
                    0.001,
                ])
            )

            # Velocity
            df = pd.read_csv(
                os.path.join(
                    respath,
                    "%s_hbeta_InterpolationResults.csv"
                    % sn,  # TODO Allow for other lines
                )
            )
            vel = df[df["Date"] == calib_date[i]]["VelInt"].to_numpy()[0]
            vel_err_lower = df[df["Date"] == calib_date[i]]["ErrorLower"].to_numpy()[0]
            vel_err_upper = df[df["Date"] == calib_date[i]]["ErrorUpper"].to_numpy()[0]
            vel_err = calc_single_error(vel_err_lower, vel_err_upper, error_mode)
            datadict["vel"].append(vel)
            datadict["vel_err"].append(vel_err)

            df = pd.read_csv(
                os.path.join(
                    respath,
                    "%s_halpha-ae_InterpolationResults.csv"
                    % sn,  # TODO Allow for other lines
                )
            )
            ae = df[df["Date"] == calib_date[i]]["VelInt"].to_numpy()[0]
            ae_err_lower = df[df["Date"] == calib_date[i]]["ErrorLower"].to_numpy()[0]
            ae_err_upper = df[df["Date"] == calib_date[i]]["ErrorUpper"].to_numpy()[0]
            ae_err = calc_single_error(ae_err_lower, ae_err_upper, error_mode)
            datadict["ae"].append(ae)
            datadict["ae_err"].append(ae_err)

            # Epoch
            datadict["epoch"].append(calib_date[i])

            # Distance modulus
            mu = np.mean(info["DistMod"].to_numpy())
            mu_err = np.mean(info["DistMod_Error"])
            datadict["mu"].append(mu)
            datadict["mu_err"].append(mu_err)

    if mag_sys is None:
        datadict["mag_sys"] = [0] * len(datadict["SN"])
    else:
        datadict["mag_sys"] = [mag_sys] * len(datadict["SN"])

    if vel_sys is None:
        datadict["vel_sys"] = [0] * len(datadict["SN"])
    else:
        datadict["vel_sys"] = [vel_sys] * len(datadict["SN"])

    if col_sys is None:
        datadict["col_sys"] = [0] * len(datadict["SN"])
    else:
        datadict["col_sys"] = [col_sys] * len(datadict["SN"])

    if ae_sys is None:
        datadict["ae_sys"] = [0] * len(datadict["SN"])
    else:
        datadict["ae_sys"] = [ae_sys] * len(datadict["SN"])

    scm_data = pd.DataFrame(datadict)

    scm_data["m_cut_nom"] = m_cut_nom * np.ones(len(scm_data))
    scm_data["sig_cut_nom"] = sig_cut_nom * np.ones(len(scm_data))

    if export:
        if isinstance(export, bool):
            scm_data.to_csv(
                os.path.join(
                    pa.get_res_path(), "SCM_Data_%s_%d.csv" % (instrument[0], date[0])
                )
            )
        else:
            scm_data.to_csv(export)

    return datadict
