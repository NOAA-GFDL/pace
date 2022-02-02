# Driver Examples

Here we have example scripts and configuration for running the Pace driver.
Currently this contains one example, to run the model on a baroclinic test case using the numpy backend.
You will find this runs fairly slowly, since the "compiled" code is still Python.
In the future, we will add examples for the compiled backends we support.

Note that on the baroclinic test case example you will see significant grid imprinting in the first hour time evolution.
Rest assured that this duplicates the behavior of the original Fortran code.

We have also included a utility to convert the zarr output of the run to netcdf, for convenience. To convert `output.zarr` to `output.nc`, you would run:

```bash
$ python3 zarr_to_nc.py output.zarr output.nz
```

## Docker

To run a baroclinic c12 case with Docker in a single command, run `run_docker.sh`.
This example will start from the Python 3.8 docker image, install extra dependencies and Python packages, and execute the example, leaving the output in this directory.

There is also a script `plot_output.py` which will show the output of the run. To use it, you must install matplotlib (e.g. with `pip install matplotlib`).

## Host Machine

To run examples on your host machine, you will need to have an MPI library on your machine suitable for installing mpi4py.
For example, on Ubuntu 20.04 this could be the libopenmpi3 and libopenmpi-dev libraries.

With these requirements installed, set up the virtual environment with

```bash
$ create_venv.sh
$ . venv/bin/activate
```

With the environment activated, the model itself can be run with `python3 -m pace.driver.run <config yaml>`.
Currently this must be done using some kind of mpirun, though there are plans to enable single-tile runs without MPI.
The exact command will vary based on your MPI implementation, but you can try running

```bash
$ mpirun -n 6 python3 -m pace.driver.run ./configs/baroclinic_c12.yaml
```

To run the example at C48 resolution instead of C12, you can update the value of `nx_tile` in the configuration file from 12 to 48.
Here you can also change the timestep in seconds `dt_atmos`, as well as the total run duration with `minutes`, or by adding values for `hours` or `days`.
