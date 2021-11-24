from gt4py.gtscript import BACKWARD, PARALLEL, computation, interval

from fv3gfs.physics.global_config import *


def get_phi_fv3(
    gt0: FIELD_FLT, gq0: FIELD_FLT, del_gz: FIELD_FLT, phii: FIELD_FLT, phil: FIELD_FLT
):
    with computation(PARALLEL), interval(0, -1):
        del_gz = (
            del_gz[0, 0, 0] * gt0[0, 0, 0] * (1.0 + con_fvirt * max(0.0, gq0[0, 0, 0]))
        )

    with computation(BACKWARD):
        with interval(-1, None):
            phii = 0.0
        with interval(-2, -1):
            phil = 0.5 * (phii[0, 0, 1] + phii[0, 0, 1] + del_gz[0, 0, 0])
            phii = phii[0, 0, 1] + del_gz[0, 0, 0]
        with interval(0, -2):
            phil = 0.5 * (phii[0, 0, 1] + phii[0, 0, 1] + del_gz[0, 0, 0])
            phii = phii[0, 0, 1] + del_gz[0, 0, 0]
