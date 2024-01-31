from typing import Dict, Tuple

import ndsl.dsl
import ndsl.util
from ndsl.util import X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM
from pace.fv3core.stencils.map_single import MapSingle


class MapSingleFactory:
    _object_pool: Dict[Tuple[int, int, Tuple[str, ...]], MapSingle] = {}
    """Pool of MapSingle objects."""

    def __init__(
        self,
        stencil_factory: ndsl.dsl.StencilFactory,
        quantity_factory: ndsl.util.QuantityFactory,
    ):
        self.stencil_factory = stencil_factory
        self.quantity_factory = quantity_factory

    def __call__(
        self,
        kord: int,
        mode: int,
        *args,
        **kwargs,
    ):
        key_tuple = (kord, mode, (X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM))
        if key_tuple not in self._object_pool:
            self._object_pool[key_tuple] = MapSingle(
                self.stencil_factory, self.quantity_factory, *key_tuple
            )
        return self._object_pool[key_tuple](*args, **kwargs)
