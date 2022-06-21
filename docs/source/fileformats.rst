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

    import numpy as np
    from scipy import stats
    import cloudpickle

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
    with open("Data/" + snname + "/" + snname + "TimeKDE.pkl", "wb") as f:
        cloudpickle.dump(kernel, f)

    return None

.. note::
   The ToE has to be stored as the MJD in the observer frame. The ToE will be subtracted, e.g., from the stored MJD of the spectra.

