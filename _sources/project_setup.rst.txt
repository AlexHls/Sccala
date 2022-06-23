.. _projectsetup:

************
Project Setup
************

.. note::
   Setting up a project is always necessary (even when using Sccala e.g. through a Jupyter notebook) since Sccala requires certain settings. There might be workarounds, but they will not be addressed in this notebook.


Setting up your project directory
=================================

Although it is not strictly necessary to collect all your data into any specific directory structure, it is highly recommended to collect your SN data into a directory structure as follows:
::

    project_root
    │   runner_files.txt    
    │
    └───Data
        │
        ├───supernova1
        │   │   supernova1_info.csv
        │   │   supernova1_TimeKDE.pkl
        │   │   spec1.dat
        │   │   spec1_error.dat
        │   │   spec2.dat
        │   │   ... 
        │
        └───supernova2
            │   supernova2_info.csv
            │   supernova2_TimeKDE.pkl
            │   spec1.dat
            │   spec1_error.dat
            │   spec2.dat
            │   ... 

Here, `project_root` refers to the main directory, where everything related to a specific project *could* live. As an example, all the `runner_files.txt` should be stored here (for more details on the runner files, see :doc:`fileformats`).

The important part here is the `Data` directory. Here is where you collect all your SN data. Each SN should have its own subdirectory where all the spectra, photometry etc. should be collected. For specifics on the file formats, contents on the `supernova_info.txt` files etc., see :doc:`fileformats`.

Initialising the project
========================

After you have finished setting up your directory structure, it is time to initialise the project. You can do so by running
::

    sccala-project-init

from within your `project_root` directory. This will give you the opportunity to set several paths to various directories relevant for the project. The default values correspond to the ones used here, but it is also possible to add custom paths. In this case it is recommended to give absolute paths.

If you used the default values, your project directory should now look like this:
::

    project_root
    │   runner_files.txt    
    │
    ├───Data
    │   │
    │   ├───supernova1
    │   │   │   supernova1_info.csv
    │   │   │   supernova1_TimeKDE.pkl
    │   │   │   spec1.dat
    │   │   │   spec1_error.dat
    │   │   │   spec2.dat
    │   │   │   ... 
    │   │
    │   └───supernova2
    │       │   supernova2_info.csv
    │       │   supernova2_TimeKDE.pkl
    │       │   spec1.dat
    │       │   spec1_error.dat
    │       │   spec2.dat
    │       │   ... 
    │
    └───.sccala_proj
        │   .sccala_paths
        │
        ├───results
        │   │   ... 
        │
        └───diagnostic
            │   ...

The `.sccala_proj` directory is where most of the output files will be stored per default, e.g. diagnostic plots in the `diagnostic` directory or interpolation results in the `results` directory. Here you can also find the `.sccala_paths` file, which stores the paths settings for Sccala.

After this, everything is set up and you can now start standardising!
