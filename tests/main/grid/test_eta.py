#!/usr/bin/env python3

import os

import numpy as np
import pytest
import xarray as xr
import yaml

import pace.driver


input_dir="./"

"""
This test checks to ensure that ak and bk
values are read-in and stored properly.
In addition, this test checks to ensure that
the function set_hybrid_pressure_coefficients
fail as expected if the computed eta values
vary non-mononitically and if the eta_file
is not provided.
"""


def set_answers(eta_file):

    """
    Read in the expected values of ak and bk
    arrays from the input eta NetCDF files.
    """

    data = xr.open_dataset(eta_file)
    return data["ak"].values, data["bk"].values


def write_eta_79_file(eta_file) :

    #Assign ak data
    ak=np.array(
        [ 3.000000e+02,  6.467159e+02,  1.045222e+03,  1.469188e+03,  1.897829e+03,
          2.325385e+03,  2.754396e+03,  3.191294e+03,  3.648332e+03,  4.135675e+03,
          4.668282e+03,  5.247940e+03,  5.876271e+03,  6.554716e+03,  7.284521e+03,
          8.066738e+03,  8.902188e+03,  9.791482e+03,  1.073499e+04,  1.162625e+04,
          1.237212e+04,  1.299041e+04,  1.349629e+04,  1.390277e+04,  1.422098e+04,
          1.446058e+04,  1.462993e+04,  1.473633e+04,  1.478617e+04,  1.478511e+04,
          1.473812e+04,  1.464966e+04,  1.452370e+04,  1.436382e+04,  1.417324e+04,
          1.395491e+04,  1.371148e+04,  1.344540e+04,  1.315890e+04,  1.285407e+04,
          1.253280e+04,  1.219685e+04,  1.184788e+04,  1.148739e+04,  1.111682e+04,
          1.073748e+04,  1.035062e+04,  9.957395e+03,  9.558875e+03,  9.156069e+03,
          8.749922e+03,  8.341315e+03,  7.931065e+03,  7.519942e+03,  7.108648e+03,
          6.698281e+03,  6.290007e+03,  5.884984e+03,  5.484372e+03,  5.089319e+03,
          4.700960e+03,  4.320421e+03,  3.948807e+03,  3.587201e+03,  3.236666e+03,
          2.898237e+03,  2.572912e+03,  2.261667e+03,  1.965424e+03,  1.685079e+03,
          1.421479e+03,  1.175419e+03,  9.476516e+02,  7.388688e+02,  5.497130e+02,
          3.807626e+02,  2.325417e+02,  1.054810e+02, -8.381903e-04,  0.000000e+00] )
    
    #Assign bk data
    bk=np.array(
        [ 0.,         0.,         0.,         0.,         0.,         0.,
          0.,         0.,         0.,         0.,         0.,         0.,
          0.,         0.,         0.,         0.,         0.,         0.,
          0.,         0.00106595, 0.00412866, 0.00900663, 0.01554263, 0.02359921,
          0.03305481, 0.0438012,  0.05574095, 0.06878554, 0.08285347, 0.09786981,
          0.1137643,  0.130471,   0.1479275,  0.1660746,  0.1848558,  0.2042166,
          0.2241053,  0.2444716,  0.2652672,  0.286445,   0.3079604,  0.3297701,
          0.351832,   0.3741062,  0.3965532,  0.4191364,  0.4418194,  0.4645682,
          0.48735,    0.5101338,  0.5328897,  0.5555894,  0.5782067,  0.6007158,
          0.6230936,  0.6452944,  0.6672683,  0.6889648,  0.7103333,  0.7313231,
          0.7518838,  0.7719651,  0.7915173,  0.8104913,  0.828839,   0.846513,
          0.8634676,  0.8796583,  0.8950421,  0.9095779,  0.9232264,  0.9359506,
          0.9477157,  0.9584892,  0.9682413,  0.9769447,  0.9845753,  0.9911126,
          0.9965372,  1.        ] )
    
    #Create a Dataset instance
    coefficients = xr.Dataset(
        { "ak": (["km1"], ak),
          "bk": (["km1"], bk) 
        })
    
    #Set attributes for each variable
    coefficients["ak"].attrs["units"]=""
    coefficients["bk"].attrs["units"]=""
    
    #Write netcdf file
    coefficients.to_netcdf(eta_file)


def write_eta_91_file(eta_file) :
    #Assign ak data
    ak = np.array(
        [1.00000000e+00, 1.75000000e+00, 2.75000000e+00, 4.09999990e+00,
         5.98951054e+00, 8.62932968e+00, 1.22572632e+01, 1.71510906e+01,
         2.36545467e+01, 3.21627693e+01, 4.31310921e+01, 5.71100426e+01,
         7.46595764e+01, 9.64470978e+01, 1.23169769e+02, 1.55601318e+02,
         1.94594009e+02, 2.41047531e+02, 2.95873840e+02, 3.60046967e+02,
         4.34604828e+02, 5.20628723e+02, 6.19154846e+02, 7.31296021e+02,
         8.58240906e+02, 1.00106561e+03, 1.16092859e+03, 1.33903992e+03,
         1.53650012e+03, 1.75448938e+03, 1.99417834e+03, 2.25667407e+03,
         2.54317139e+03, 2.85476392e+03, 3.19258569e+03, 3.55775366e+03,
         3.95135107e+03, 4.37428662e+03, 4.82711084e+03, 5.31022168e+03,
         5.82387793e+03, 6.36904248e+03, 6.94875244e+03, 7.56691992e+03,
         8.22634277e+03, 8.93120996e+03, 9.68446191e+03, 1.04822725e+04,
         1.13182793e+04, 1.21840771e+04, 1.30655674e+04, 1.39532207e+04,
         1.48307285e+04, 1.56872617e+04, 1.65080645e+04, 1.72810996e+04,
         1.79942988e+04, 1.86363223e+04, 1.91961797e+04, 1.96640723e+04,
         2.00301914e+04, 2.02853691e+04, 2.04215254e+04, 2.04300684e+04,
         2.03028730e+04, 2.00323711e+04, 1.96110664e+04, 1.90313848e+04,
         1.82866426e+04, 1.73777930e+04, 1.63224639e+04, 1.51444033e+04,
         1.38725674e+04, 1.25404785e+04, 1.11834170e+04, 9.83532715e+03,
         8.52630664e+03, 7.28224512e+03, 6.12326074e+03, 5.06350684e+03,
         4.11124902e+03, 3.27000122e+03, 2.53922729e+03, 1.91530762e+03,
         1.39244995e+03, 9.63134766e+02, 6.20599365e+02, 3.57989502e+02,
         1.69421387e+02, 5.10314941e+01, 2.48413086e+00, 0.00000000e+00] )
    
    #Assign bk data
    bk = np.array(
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.50123992e-06,
         2.81484008e-05, 9.38666999e-05, 2.28561999e-04, 5.12343016e-04,
         1.04712998e-03, 1.95625005e-03, 3.42317997e-03, 5.58632007e-03,
         8.65428988e-03, 1.27844000e-02, 1.81719996e-02, 2.49934997e-02,
         3.34198996e-02, 4.36249003e-02, 5.57769015e-02, 7.00351968e-02,
         8.65636021e-02, 1.05520003e-01, 1.27051994e-01, 1.51319996e-01,
         1.78477004e-01, 2.08675995e-01, 2.42069006e-01, 2.78813988e-01,
         3.19043010e-01, 3.62558991e-01, 4.08596009e-01, 4.56384987e-01,
         5.05111992e-01, 5.53902984e-01, 6.01903021e-01, 6.48333013e-01,
         6.92534983e-01, 7.33981013e-01, 7.72292018e-01, 8.07236016e-01,
         8.38724971e-01, 8.66774976e-01, 8.91497016e-01, 9.13065016e-01,
         9.31702971e-01, 9.47658002e-01, 9.61175978e-01, 9.72495019e-01,
         9.81844008e-01, 9.89410996e-01, 9.95342016e-01, 1.00000000e+00] )
    
    #Create a Dataset instance
    coefficients = xr.Dataset(
        { "ak": (["km1"], ak),
          "bk": (["km1"], bk) 
        })
    
    #Set attributes for each variable
    coefficients["ak"].attrs["units"]=""
    coefficients["bk"].attrs["units"]=""
    
    #Write netcdf file
    coefficients.to_netcdf(eta_file)


def write_non_mono_eta_file(in_eta_file, out_eta_file):
    """
    Reads in file eta79.nc and alters randomly chosen ak/bk values
    This tests the expected failure of set_eta_hybrid_coefficients
    for coefficients that lead to non-monotonically increasing
    eta values
    """

    data = xr.open_dataset(in_eta_file)
    data["ak"].values[10]=data["ak"].values[0]
    data["bk"].values[20]=0.0
    
    data.to_netcdf(out_eta_file)


@pytest.mark.parametrize("km", [79,91])
def test_set_hybrid_pressure_coefficients_correct(km):

    """This test checks to see that the ak and bk arrays
    are read-in correctly and are stored as
    expected.  Both values of km=79 and km=91 are
    tested and both tests are expected to pass
    with the stored ak and bk values agreeing with the
    values read-in directly from the NetCDF file.
    """

    dirname = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(
        dirname, "../../../driver/examples/configs/baroclinic_c12.yaml"
    )

    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    yaml_config["nz"] = km
    eta_file = f"{input_dir}/eta{km}.nc"
    yaml_config["grid_config"]["config"]["eta_file"] = eta_file
    if km == 79 : write_eta_79_file(eta_file)
    if km == 91 : write_eta_91_file(eta_file)

    driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
    driver_config.comm_config = pace.driver.NullCommConfig(rank=0, total_ranks=6)
    driver = pace.driver.Driver(config=driver_config)

    p_results = driver.state.grid_data.p.data
    ak_results = driver.state.grid_data.ak.data
    bk_results = driver.state.grid_data.bk.data
    ak_answers, bk_answers = set_answers(f"{input_dir}/eta{km}.nc")

    if ak_answers.size != ak_results.size:
        raise ValueError("Unexpected size of bk")
    if bk_answers.size != bk_results.size:
        raise ValueError("Unexpected size of ak")

    if not np.array_equal(ak_answers, ak_results):
        raise ValueError("Unexpected value of ak")
    if not np.array_equal(bk_answers, bk_results):
        raise ValueError("Unexpected value of bk")

    driver.safety_checker.clear_all_checks()


@pytest.mark.parametrize( "eta_file",["file_is_not_here",
                                      f"{input_dir}/eta_not_mono.nc"])
@pytest.mark.xfail
def test_set_hybrid_pressure_coefficients_fail(eta_file):

    """This test checks to see that the program
    fails when (1) the eta_file is not specified in the yaml
    configuration file; and (2), the computed eta values
    increase non-monotonically.  For the latter test, the eta_file
    is specified in test_config_not_mono.yaml file and
    the ak and bk values in the eta_file have been changed nonsensically
    to result in erronenous eta values.
    """

    dirname = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(
        dirname, "../../../driver/examples/configs/baroclinic_c12.yaml"
    )

    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    yaml_config["grid_config"]["config"]["eta_file"] = eta_file
    if "not_mono" in eta_file :
        in_eta_file= f"{input_dir}/eta79.nc"
        if not os.path.isfile(in_eta_file) : write_eta_79_file(in_eta_file)
        write_non_mono_eta_file( in_eta_file, eta_file )

    driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
    driver_config.comm_config = pace.driver.NullCommConfig(rank=0, total_ranks=6)
    driver = pace.driver.Driver(config=driver_config)

