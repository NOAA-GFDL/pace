from typing import Dict

import ndsl.dsl.gt4py_utils as utils
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import Float, FloatField
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Quantity

from pyFV3.stencils.fillz import FillNegativeTracerValues
from pyFV3.stencils.map_single import MapSingle


class MapNTracer:
    """
    Fortran code is mapn_tracer, test class is MapN_Tracer_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        kord: int,
        nq: int,
        fill: bool,
        tracers: Dict[str, Quantity],
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["tracers"],
        )
        self._nq = int(nq)
        self._qs = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )

        kord_tracer = [kord] * self._nq
        kord_tracer[5] = 9  # qcld

        self._list_of_remap_objects = [
            MapSingle(
                stencil_factory,
                quantity_factory,
                kord_tracer[i],
                0,
                dims=[X_DIM, Y_DIM, Z_DIM],
            )
            for i in range(len(kord_tracer))
        ]

        if fill:
            self._fill_negative_tracers = True
            self._fillz = FillNegativeTracerValues(
                stencil_factory,
                quantity_factory,
                self._nq,
                tracers,
            )
        else:
            self._fill_negative_tracers = False

    def __call__(
        self,
        pe1: FloatField,
        pe2: FloatField,
        dp2: FloatField,
        tracers: Dict[str, Quantity],
    ):
        """
        Remaps the tracer species onto the Eulerian grid
        and optionally fills negative values in the tracer fields
        Assumes the minimum value is 0 for each tracer

        Args:
            pe1 (in): Lagrangian pressure levels
            pe2 (in): Eulerian pressure levels
            dp2 (in): Difference in pressure between Eulerian levels
            tracers (inout): tracers to be remapped
        """
        for i, q in enumerate(utils.tracer_variables[0 : self._nq]):
            self._list_of_remap_objects[i](tracers[q], pe1, pe2, self._qs)

        if self._fill_negative_tracers is True:
            self._fillz(dp2, tracers)
