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

