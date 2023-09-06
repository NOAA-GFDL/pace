import numpy as np

import pace.dsl.gt4py_utils as utils
import pace.fv3core.initialization.init_utils as init_utils
import pace.util as fv3util
from pace.fv3core.initialization.dycore_state import DycoreState
from pace.util.grid import GridData


nhalo = fv3util.N_HALO_DEFAULT


def init_baroclinic_state(
    grid_data: GridData,
    quantity_factory: fv3util.QuantityFactory,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.CubedSphereCommunicator,
) -> DycoreState:
    """
    Create a DycoreState object with quantities initialized to the Jablonowski &
    Williamson baroclinic test case perturbation applied to the cubed sphere grid.
    """
    sample_quantity = grid_data.lat
    shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    nx, ny, nz = init_utils.local_compute_size(shape)
    numpy_state = init_utils.empty_numpy_dycore_state(shape)
    # Initializing to values the Fortran does for easy comparison
    numpy_state.delp[:] = 1e30
    numpy_state.delp[:nhalo, :nhalo] = 0.0
    numpy_state.delp[:nhalo, nhalo + ny :] = 0.0
    numpy_state.delp[nhalo + nx :, :nhalo] = 0.0
    numpy_state.delp[nhalo + nx :, nhalo + ny :] = 0.0
    numpy_state.pe[:] = 0.0
    numpy_state.pt[:] = 1.0
    numpy_state.ua[:] = 1e35
    numpy_state.va[:] = 1e35
    numpy_state.uc[:] = 1e30
    numpy_state.vc[:] = 1e30
    numpy_state.w[:] = 1.0e30
    numpy_state.delz[:] = 1.0e25
    numpy_state.phis[:] = 1.0e25
    numpy_state.ps[:] = init_utils.surface_pressure
    eta = np.zeros(nz)
    eta_v = np.zeros(nz)
    islice, jslice, slice_3d, slice_2d = init_utils.compute_slices(nx, ny)
    # Slices with extra buffer points in the horizontal dimension
    # to accomodate averaging over shifted calculations on the grid
    _, _, slice_3d_buffer, slice_2d_buffer = init_utils.compute_slices(nx + 1, ny + 1)

    init_utils.setup_pressure_fields(
        eta=eta,
        eta_v=eta_v,
        delp=numpy_state.delp[slice_3d],
        ps=numpy_state.ps[slice_2d],
        pe=numpy_state.pe[slice_3d],
        peln=numpy_state.peln[slice_3d],
        pk=numpy_state.pk[slice_3d],
        pkz=numpy_state.pkz[slice_3d],
        ak=utils.asarray(grid_data.ak.data),
        bk=utils.asarray(grid_data.bk.data),
        ptop=grid_data.ptop,
    )

    init_utils.baroclinic_initialization(
        eta=eta,
        eta_v=eta_v,
        peln=numpy_state.peln[slice_3d_buffer],
        qvapor=numpy_state.qvapor[slice_3d_buffer],
        delp=numpy_state.delp[slice_3d_buffer],
        u=numpy_state.u[slice_3d_buffer],
        v=numpy_state.v[slice_3d_buffer],
        pt=numpy_state.pt[slice_3d_buffer],
        phis=numpy_state.phis[slice_2d_buffer],
        delz=numpy_state.delz[slice_3d_buffer],
        w=numpy_state.w[slice_3d_buffer],
        lon=utils.asarray(grid_data.lon.data[slice_2d_buffer]),
        lat=utils.asarray(grid_data.lat.data[slice_2d_buffer]),
        lon_agrid=utils.asarray(grid_data.lon_agrid.data[slice_2d_buffer]),
        lat_agrid=utils.asarray(grid_data.lat_agrid.data[slice_2d_buffer]),
        ee1=utils.asarray(grid_data.ee1.data[slice_3d_buffer]),
        ee2=utils.asarray(grid_data.ee2.data[slice_3d_buffer]),
        es1=utils.asarray(grid_data.es1.data[slice_3d_buffer]),
        ew2=utils.asarray(grid_data.ew2.data[slice_3d_buffer]),
        ptop=grid_data.ptop,
        adiabatic=adiabatic,
        hydrostatic=hydrostatic,
        nx=nx,
        ny=ny,
    )

    init_utils.p_var(
        delp=numpy_state.delp[slice_3d],
        delz=numpy_state.delz[slice_3d],
        pt=numpy_state.pt[slice_3d],
        ps=numpy_state.ps[slice_2d],
        qvapor=numpy_state.qvapor[slice_3d],
        pe=numpy_state.pe[slice_3d],
        peln=numpy_state.peln[slice_3d],
        pkz=numpy_state.pkz[slice_3d],
        ptop=grid_data.ptop,
        moist_phys=moist_phys,
        make_nh=(not hydrostatic),
    )
    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    comm.halo_update(state.phis, n_points=nhalo)

    comm.vector_halo_update(state.u, state.v, n_points=nhalo)

    return state
