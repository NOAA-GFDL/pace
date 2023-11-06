import math

import numpy as np

eta_0 = 0.252
surface_pressure = 1.0e5  # units of (Pa), from Table VI of DCMIP2016

def vertical_coordinate(eta_value):
    """
    Equation (1) JRMS2006
    computes eta_v, the auxiliary variable vertical coordinate
    """
    return (eta_value - eta_0) * math.pi * 0.5


def compute_eta(ak, bk):
    """
    Equation (1) JRMS2006
    eta is the vertical coordinate and eta_v is an auxiliary vertical coordinate
    """
    eta = 0.5 * ((ak[:-1] + ak[1:]) / surface_pressure + bk[:-1] + bk[1:])
    eta_v = vertical_coordinate(eta)
    return eta, eta_v


