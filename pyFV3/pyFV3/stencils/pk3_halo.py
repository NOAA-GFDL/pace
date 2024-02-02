from gt4py.cartesian.gtscript import FORWARD, computation, horizontal, interval, region
from ndsl.constants import X_DIM, Y_DIM
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.initialization.allocator import QuantityFactory


# TODO merge with pe_halo? reuse partials?
# NOTE: This is different from pyFV3.stencils.pe_halo.edge_pe
def edge_pe_update(
    pe: FloatFieldIJ, delp: FloatField, pk3: FloatField, ptop: Float, akap: Float
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(
                region[local_is - 2 : local_is, local_js : local_je + 1],
                region[local_ie + 1 : local_ie + 3, local_js : local_je + 1],
                region[local_is - 2 : local_ie + 3, local_js - 2 : local_js],
                region[local_is - 2 : local_ie + 3, local_je + 1 : local_je + 3],
            ):
                pe = ptop
        with interval(1, None):
            with horizontal(
                region[local_is - 2 : local_is, local_js : local_je + 1],
                region[local_ie + 1 : local_ie + 3, local_js : local_je + 1],
                region[local_is - 2 : local_ie + 3, local_js - 2 : local_js],
                region[local_is - 2 : local_ie + 3, local_je + 1 : local_je + 3],
            ):
                pe = pe + delp[0, 0, -1]
                pk3 = pe ** akap


class PK3Halo:
    """
    Fortran name is pk3_halo
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        grid_indexing = stencil_factory.grid_indexing
        origin = grid_indexing.origin_full()
        domain = grid_indexing.domain_full(add=(0, 0, 1))
        ax_offsets = grid_indexing.axis_offsets(origin, domain)
        self._edge_pe_update = stencil_factory.from_origin_domain(
            func=edge_pe_update,
            externals={
                **ax_offsets,
            },
            origin=origin,
            domain=domain,
        )
        self._pe_tmp = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )

    def __call__(self, pk3: FloatField, delp: FloatField, ptop: Float, akap: Float):
        """Update pressure raised to the kappa (pk3) in halo region.

        Args:
            pk3: 3D interface pressure raised to power of kappa using constant kappa
            delp: Vertical delta in pressure
            ptop: The pressure level at the top of atmosphere
            akap: Poisson constant (KAPPA)
        """
        self._edge_pe_update(self._pe_tmp, delp, pk3, ptop, akap)
