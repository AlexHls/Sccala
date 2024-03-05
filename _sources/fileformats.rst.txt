.. _fileformats:

************
File Formats
************

Sccala uses several different files as input, all of which need to have to correct format to be correctly read. Here you can find detailed informations on the different input files and how they should be formatted.

==========
Info files
==========

Info files collect all the metadata connected to the spectra a specific SN and are loosely based on the format exported by the `WiSERep <https://www.wiserep.org/>`_ metadata format. This means that you can use the metadata files that come with `WiSERep <https://www.wiserep.org/>`_ SNe as a basis and you will only need to add the missing columns.

+-------------------+-------------------------------------------------------------------------------------------------------------------+
| Column            | Explanation                                                                                                       |
+===================+===================================================================================================================+
| `Redshift`        | Redshift of each spectrum. Although different values can be given, Sccala uses the average value of all lines.    |
+-------------------+-------------------------------------------------------------------------------------------------------------------+
| `Redshift_Error`  | Uncertainty of the redshift. Although different values can be given, Sccala uses the average value of all lines.  |
+-------------------+-------------------------------------------------------------------------------------------------------------------+
| `ID`              | ID of spectrum. Used to identify spectra of SN in diagnostic/ result files.                                       |
+-------------------+-------------------------------------------------------------------------------------------------------------------+
| `MJD`             | MJD of spectrum in observer frame of reference. Used to calculate time since explosion.                           |
+-------------------+-------------------------------------------------------------------------------------------------------------------+
| `File`            | (Absolute) path of the file containing the spectrum.                                                              |
+-------------------+-------------------------------------------------------------------------------------------------------------------+

There are some optional columns which can be added, but are not essential for the Sccala workflow:


+-----------------------+---------------------------------------------------+
| Column                | Explanation                                       |
+=======================+===================================================+
| `IAU name`            | IAU name of the SN.                               |
+-----------------------+---------------------------------------------------+
| `Internal name`       | Internally used name of the SN.                   |
+-----------------------+---------------------------------------------------+
| `A_V`                 | A\ :sub:`V`\  value of the host galaxy of the SN. |
+-----------------------+---------------------------------------------------+
| `MJD_Explosion`       | Time of explosion in MJD.                         |
+-----------------------+---------------------------------------------------+
| `MJD Explosion Error` | Uncertainty of the time of explosion.             |
+-----------------------+---------------------------------------------------+

=============
ToE KDE files
=============

Although the Time of Explosion (ToE) could be stored as a plain number, Sccala uses a kernel density estimate (KDE) of the ToE as prior for inferring, e.g., the velocity at a certain phase. For this, the ToE KDE needs to be stored in each SN directory. This has the additional benefit that in cases where the ToE is obtained through statistical inference, the posterior can be directly stored and used as a prior for Sccala.

Here, Sccala expects a `scipy.stats.gaussian_kde` object stored with `cloudpickle` adhering to the following naming scheme:
::

    <snname>TimeKDE.pkl

In case you only have the ToE date without any probability, you can manually create the KDE using e.g. the following function:
::

    import os
    import cloudpickle
    import numpy as np
    from scipy import stats

    def manual_toe(snname, toe, toeerr):
        """Function to manually create *TimeKDE.pkl if e.g. phase matching fails

        Parameters
        ----------
        snname : String
            Name of SN to be analyzed
        toe : float
            Time of explosion in MJD
        toeerr : float
            Uncertainty of time of explosion

        Returns
        -------
        none
        """
        minima = np.random.normal(toe, toeerr, 100000)
        kernel = stats.gaussian_kde(minima, bw_method="silverman")

        # Adapt the path if needed
        with open(os.path.join("Data", snname, snname + "_TimeKDE.pkl"), "wb") as f:
            cloudpickle.dump(kernel, f)

        return None

.. note::
   The ToE has to be stored as the MJD in the observer frame. The ToE will be subtracted, e.g., from the stored MJD of the spectra.

=======
Spectra
=======

Spectra are should be stored as plain text files (ending with `.dat`), consisting of two columns separated by two white spaces. The first column corresponds to the **wavelength** and the second column to the **flux**. The flux units do not matter, unless you intend to do synthetic photometry, in which case the flux needs to be given in `cgs` units. The spectral file must not contain any header. Generally speaking, any file which is readable by
::

    import numpy as np

    wavelength, flux = np.genfromtxt("<fluxfile>.dat").T

is acceptable.

The flux uncertainty needs to be stored in a separate file, which has the same filename as the corresponding spectral flux file but appended by `_error` (e.g.: `fluxfile.dat` `->` `fluxfile_error.dat`). It should have the same format as the flux file, i.e. to columns containing **wavelength** and **flux_uncertainty**, respectively.

In case there does not exist a `<fluxfile>_error.dat` file, Sccala will automatically calculate a standard deviation spectrum. It is also possible to manually generate the error spectrum by running
::

    from sccala.speclib import spectools

    spectools.calculate_flux_error("<fluxfile>.dat")

This function call will automatically save the appropriate uncertainty file. For more details, see the API documentation.

==========
Photometry
==========

Photometry files should store the pre-processed photometry data of a SN, i.e. after all corrections such as K-corrections etc. have been applied. All photometry should be stored in one file per photometric system, following the naming scheme `<SN name>_<photometric system>_Photometry.csv>` in a CSV file. This file must contain the following columns:

+-------------+------------------------------------------------------------------------+
| Column      | Explanation                                                            |
+=============+========================================================================+
| `MJD`       | Modified Julian Date in the observer frame of the photometry.          |
+-------------+------------------------------------------------------------------------+
| `<band>`    | Observed magnitude in the `<band>` filter passband.                    |
+-------------+------------------------------------------------------------------------+
| `<band>err` | Uncertainty of the observed magnitude in the `<band>` filter passband. |
+-------------+------------------------------------------------------------------------+

Here, a `<band>` and `<band>err` column has to be given for each photometric passband. 

============
Runner files
============

Runner files can be used as inputs for the various Sccala tools. Although it is possible to pass most of the inputs individually via the command line, it is recommended to use runner files provide somewhat more verbosity and persistence (e.g. in case you need to re-run certain steps).

Line fit
========

The line fit runner files can be used with the `sccala-linefit` command. They should contain for plain text columns separated by two white spaces, see the example below.

::

    SN ID line noisefit
    1999em 1 hbeta True
    1999em 2 hbeta True
    1999em 2 halpha-ae False
    1999gi 1 hbeta True
    1999gi 2 halpha-ae True
    ...
    
+------------+---------------------------------------------------------------------------------+
| `SN`       | Internal name of the SN, i.e. name of the SN directory in the `Data` directory. |
+------------+---------------------------------------------------------------------------------+
| `ID`       | ID of the spectrum to be fit.                                                   |
+------------+---------------------------------------------------------------------------------+
| `line`     | Specifies which line/feature is to be fit.                                      |
+------------+---------------------------------------------------------------------------------+
| `noisefit` | Specifies if noise is to be included in the fitting procedure.                  |
+------------+---------------------------------------------------------------------------------+

Synthetic photometry
====================

The synthetic photometry runner files can be used with the `sccala-photometry` command. They should contain for plain text columns separated by two white spaces, see the example below.

::

    SN  ID  CalibErr  AddErr
    1999em  1  0.00  0.03
    1999em  2  0.00  0.03
    1999em  3  0.04  0.03
    1999gi  1  0.30  0.02
    1999gi  2  0.00  0.02
    ...

+------------+---------------------------------------------------------------------------------+
| `SN`       | Internal name of the SN, i.e. name of the SN directory in the `Data` directory. |
+------------+---------------------------------------------------------------------------------+
| `ID`       | ID of the spectrum to be fit.                                                   |
+------------+---------------------------------------------------------------------------------+
| `CalibErr` | Calibration error of absolute flux. Added linearly to integration uncertainty.  |
+------------+---------------------------------------------------------------------------------+
| `AddErr`   | Additional errors. Added quadratically to integration uncertainty.              |
+------------+---------------------------------------------------------------------------------+

Filter lists
============

The filter runner files can be used with the `sccala-photometry` command in addition to the synthetic photometry files. Each line should contain the path to a filter (following the naming scheme of the `SVO filter service <http://svo2.cab.inta-csic.es/theory/fps/>`_) see the example below. Several filters are already built into Sccala and can be found in `asynphot/filters/`.

::
 
    Generic/Bessell12.U
    Generic/Bessell12.B
    Generic/Bessell12.V
    Generic/Bessell12.R
    Generic/Bessell12.I

Interpolation rules
===================

The interpolation rules file is to be used with `sccala-photometry-Interpolation` and `sccala-velocity-interpolation`. Here, the idea is to have one persistent file per interpolation target. In this file, all the possible interpolation rules are collected. The file itself is a regular CSV file, see the example below.

::

    SN,errorfloor,errorscale,region_min,region_max,extrapolate
    1999em,0.0,1.0,20.0,60.0,5.0
    1999gi,0.0,1.0,15.0,60.0,2.0
    ...

+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| `SN`          | Internal name of the SN, i.e. name of the SN directory in the `Data` directory.                                                    |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| `errorfloor`  | Minimum uncertainty for all datapoints. All datapoints with an uncertainty smaller than this will have it increased to this value. |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| `errorscale`  | Scales the uncertainty of all datapoints by this value.                                                                            |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| `region_min`  | Minimum epoch for interpolation. Datapoints earlier than this value will not be considered for the interpolation.                  |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| `region_max`  | Maximum epoch for interpolation. Datapoints later than this value will not be considered for the interpolation.                    |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+
| `extrapolate` | Number of days after the last valid datapoint until which the fit will extrapolate, even if a later date is specified as target.   |
+---------------+------------------------------------------------------------------------------------------------------------------------------------+

SN lists
========

The SN list file is to be used with `collect-scm-data`. It lists all the SNe you want to collect, as well as some specifics about which data to collect, see the example below. It should be given as a regular CSV file.

::

    SN,mag,col0,col1,date,dataset,instrument
    1999em,I,V,I,35.0,KAIT,Bessell12
    2004ay,I,V,I,35.0,KAIT,Bessell12
    2005cs,I,V,I,35.0,KAIT_CALIB,Bessell12
    1999gi,I,V,I,35.0,KAIT_CALIB,Bessell12
    ...

+-----------------+--------------------------------------------------------------------------------------------+
| `SN`            | Internal name of the SN, i.e. name of the SN directory in the `Data` directory.            |
+-----------------+--------------------------------------------------------------------------------------------+
| `mag`           | Filter which is to be used for magnitudes.                                                 |
+-----------------+--------------------------------------------------------------------------------------------+
| `col0` & `col1` | Filters which are to be used as colors. Color is calculated as `col0 - col1`.              |
+-----------------+--------------------------------------------------------------------------------------------+
| `date`          | Epoch from which data is to be taken.                                                      |
+-----------------+--------------------------------------------------------------------------------------------+
| `dataset`       | Dataset to which the SN is to be assigned to. Calibrators should have the `_CALIB` suffix .|
+-----------------+--------------------------------------------------------------------------------------------+
| `instrument`    | Instrument from which the photometry is to be taken.                                       |
+-----------------+--------------------------------------------------------------------------------------------+

