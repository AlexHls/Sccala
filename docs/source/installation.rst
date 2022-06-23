.. _installation:

************
Installation
************

.. note::
   - Sccala is only compatible with Python >=3.6 and it is recommended to use Python 3.10.
   - Sccala only supports Linux. It will most likely work on MacOS and on WSL on Windows, but this has not been tested
   - The instructions here use a virtual environment, but an Anaconda environment will most likely also work. But since not all required packages are available through Anaconda, it is recommended to use a virtual environment to avoid interoperability issues with pip and Anaconda.


Requirements
============
A list of packages required by Sccala can be found in `environment definition file <https://raw.githubusercontent.com/AlexHls/Sccala/master/sccala-env.txt>`_.

Installing Sccala with pip
==========================

First, create a virtual environment using `venv <https://docs.python.org/3/library/venv.html#module-venv>`_. This will use the most recent Python version you have installed on your system.
::

    python3 -m venv sccala-env

After you have created the virtual environment, you need to activate it
::

    source sccala-env/bin/activate

This can also be set as an alias in your `.bashrc` etc.
::

    alias sccala-env="source <path-to-sccala-env>/sccala-env/bin/activate"

Next, make sure your pip is up to date and install the required packages
::

    pip install --upgrade pip
    pip install -r sccala-env.txt

Now your python environment is set up and you can proceed to install Sccala:

From source
-----------

Clone the GitHub repository locally and change into the cloned directory:
::

    git clone https://github.com/AlexHls/Sccala
    cd Sccala

Make sure that you have activated the `sccala-env` defined previously and install Sccala:
::

    pip install .


Congratulations, you have successfully installed Sccala to your system!
