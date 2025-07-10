.. highlight:: shell

============
Installation
============

Shell scripts to install Pace on specific machines such as Gaea can be found in `examples/build_scripts/`.

When cloning Pace you will need to update the repository's submodules as well:

.. code-block:: console

    $ git clone --recursive https://github.com/NOAA-GFDL/pace.git

or if you have already cloned the repository:

.. code-block:: console

    $ git submodule update --init --recursive


Pace requires GCC > 9.2, MPI, and Python 3.8 on your system, and CUDA is required to run with a GPU backend.

We recommend creating a python `venv` or conda environment specifically for Pace.

.. code-block:: console

    $ python3 -m venv venv_name
    $ source venv_name/bin/activate

Inside of your pace `venv` or conda environment pip install the Python requirements, GT4Py, and Pace:

.. code-block:: console

    $ pip3 install -r requirements_dev.txt -c constraints.txt

There are also separate requirements files which can be installed for linting (`requirements_lint.txt`) and building documentation (`requirements_docs.txt`).
