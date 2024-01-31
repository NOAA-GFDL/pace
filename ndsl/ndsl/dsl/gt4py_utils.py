from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gt4py
import numpy as np

from ndsl.dsl.typing import DTypes, Field, Float
from ndsl.util.constants import N_HALO_DEFAULT
from ndsl.util.logging import pace_log


try:
    import cupy as cp
except ImportError:
    cp = None

# If True, automatically transfers memory between CPU and GPU (see gt4py.storage)
managed_memory = True

# Number of halo lines for each field and default origin
origin = (N_HALO_DEFAULT, N_HALO_DEFAULT, 0)

# TODO: Both pyFV3 and pySHiELD need to know what is being advected
#       but the actual value should come from outside of `ndsl`.
#       There should be a set of API to deal with tracers, that lives in `ndsl`
#       but their call doesn't.
# TODO get from field_table
tracer_variables = [
    "qvapor",
    "qliquid",
    "qrain",
    "qice",
    "qsnow",
    "qgraupel",
    "qo3mr",
    "qsgs_tke",
    "qcld",
]


def mark_untested(msg="This is not tested"):
    def inner(func) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            print(f"{func.__name__}: {msg}")
            func(*args, **kwargs)

        return wrapper

    return inner


def _mask_to_dimensions(
    mask: Tuple[bool, ...], shape: Sequence[int]
) -> List[Union[str, int]]:
    assert len(mask) == 3
    dimensions: List[Union[str, int]] = []
    for i, axis in enumerate(("I", "J", "K")):
        if mask[i]:
            dimensions.append(axis)
    offset = int(sum(mask))
    dimensions.extend(shape[offset:])
    return dimensions


def _translate_origin(origin: Sequence[int], mask: Tuple[bool, ...]) -> Sequence[int]:
    if len(origin) == int(sum(mask)):
        # Correct length. Assumedd to be correctly specified.
        return origin

    assert len(mask) == 3
    final_origin: List[int] = []
    for i, has_axis in enumerate(mask):
        if has_axis:
            final_origin.append(origin[i])

    final_origin.extend(origin[len(mask) :])
    return final_origin


def make_storage_data(
    data: Field,
    shape: Optional[Tuple[int, ...]] = None,
    origin: Tuple[int, ...] = origin,
    *,
    backend: str,
    dtype: DTypes = Float,
    mask: Optional[Tuple[bool, ...]] = None,
    start: Tuple[int, ...] = (0, 0, 0),
    dummy: Optional[Tuple[int, ...]] = None,
    axis: int = 2,
    max_dim: int = 3,
    read_only: bool = True,
) -> Field:
    """Create a new gt4py storage from the given data.

    Args:
        data: Data array for new storage
        shape: Shape of the new storage. Number of indices should be equal
            to number of unmasked axes
        origin: Default origin for gt4py stencil calls
        dtype: Data type
        mask: Tuple indicating the axes used when initializing the storage.
            True indicates a masked axis, False is a used axis.
        start: Starting points for slices in data copies
        dummy: Dummy axes
        axis: Axis for 2D to 3D arrays
        backend: gt4py backend to use

    Returns:
        Field[..., dtype]: New storage

    Examples:
        1) ptop = utils.make_storage_data(top_p, q4_1.shape)
        2) ws3 = utils.make_storage_data(ws3[:, :, -1], shape, origin=(0, 0, 0))
        3) data_dict[names[i]] = make_storage_data(
               data[:, :, :, i],
               shape,
               origin=origin,
               start=start,
               dummy=dummy,
               axis=axis,
           )

    """
    n_dims = len(data.shape)
    if shape is None:
        shape = data.shape

    if mask is None:
        if not read_only:
            default_mask: Tuple[bool, ...] = (True, True, True)
        else:
            if n_dims == 1:
                if axis == 1:
                    # Convert J-fields to IJ-fields
                    default_mask = (True, True, False)
                    shape = (1, shape[axis])
                else:
                    default_mask = tuple(
                        [i == axis for i in range(max_dim)]
                    )  # type: ignore
            elif dummy or axis != 2:
                default_mask = (True, True, True)
            else:
                default_mask = (n_dims * (True,)) + ((max_dim - n_dims) * (False,))
        mask = default_mask

    if n_dims == 1:
        data = _make_storage_data_1d(
            data, shape, start, dummy, axis, read_only, backend=backend
        )
    elif n_dims == 2:
        data = _make_storage_data_2d(
            data, shape, start, dummy, axis, read_only, backend=backend
        )
    else:
        data = _make_storage_data_3d(data, shape, start, backend=backend)

    storage = gt4py.storage.from_array(
        data,
        dtype,
        backend=backend,
        aligned_index=_translate_origin(origin, mask),
        dimensions=_mask_to_dimensions(mask, data.shape),
    )
    return storage


def _make_storage_data_1d(
    data: Field,
    shape: Tuple[int, ...],
    start: Tuple[int, ...] = (0, 0, 0),
    dummy: Optional[Tuple[int, ...]] = None,
    axis: int = 2,
    read_only: bool = True,
    *,
    backend: str,
) -> Field:
    # axis refers to a repeated axis, dummy refers to a singleton axis
    axis = min(axis, len(shape) - 1)
    buffer = zeros(shape[axis], backend=backend)
    if dummy:
        axis = list(set((0, 1, 2)).difference(dummy))[0]

    kstart = start[2]
    buffer[kstart : kstart + len(data)] = asarray(data, type(buffer))

    if not read_only:
        tile_spec = list(shape)
        tile_spec[axis] = 1
        if axis == 2:
            buffer = tile(buffer, tuple(tile_spec))
        elif axis == 1:
            x = repeat(buffer[np.newaxis, :], shape[0], axis=0)
            buffer = repeat(x[:, :, np.newaxis], shape[2], axis=2)
        else:
            y = repeat(buffer[:, np.newaxis], shape[1], axis=1)
            buffer = repeat(y[:, :, np.newaxis], shape[2], axis=2)
    elif axis == 1:
        buffer = buffer.reshape((1, buffer.shape[0]))

    return buffer


def _make_storage_data_2d(
    data: Field,
    shape: Tuple[int, ...],
    start: Tuple[int, ...] = (0, 0, 0),
    dummy: Optional[Tuple[int, ...]] = None,
    axis: int = 2,
    read_only: bool = True,
    *,
    backend: str,
) -> Field:
    # axis refers to which axis should be repeated (when making a full 3d data),
    # dummy refers to a singleton axis
    do_reshape = dummy or axis != 2
    if do_reshape:
        d_axis = dummy[0] if dummy else axis
        shape2d = shape[:d_axis] + shape[d_axis + 1 :]
    else:
        shape2d = shape[0:2]

    start1, start2 = start[0:2]
    size1, size2 = data.shape
    buffer = zeros(shape2d, backend=backend)
    buffer[start1 : start1 + size1, start2 : start2 + size2] = asarray(
        data, type(buffer)
    )

    if not read_only:
        buffer = repeat(buffer[:, :, np.newaxis], shape[axis], axis=2)
        if axis != 2:
            buffer = moveaxis(buffer, 2, axis)
    elif do_reshape:
        buffer = buffer.reshape(shape)

    return buffer


def _make_storage_data_3d(
    data: Field,
    shape: Tuple[int, ...],
    start: Tuple[int, ...] = (0, 0, 0),
    *,
    backend: str,
) -> Field:
    istart, jstart, kstart = start
    isize, jsize, ksize = data.shape
    buffer = zeros(shape, backend=backend)
    buffer[
        istart : istart + isize,
        jstart : jstart + jsize,
        kstart : kstart + ksize,
    ] = asarray(data, type(buffer))
    return buffer


def make_storage_from_shape(
    shape: Tuple[int, ...],
    origin: Tuple[int, ...] = origin,
    *,
    backend: str,
    dtype: DTypes = Float,
    mask: Optional[Tuple[bool, ...]] = None,
) -> Field:
    """Create a new gt4py storage of a given shape filled with zeros.

    Args:
        shape: Shape of the new storage
        origin: Default origin for gt4py stencil calls
        dtype: Data type
        mask: Tuple indicating the axes used when initializing the storage
        backend: gt4py backend to use when making the storage

    Returns:
        Field[..., dtype]: New storage

    Examples:
        1) utmp = utils.make_storage_from_shape(ua.shape)
        2) qx = utils.make_storage_from_shape(
               qin.shape, origin=(grid().is_, grid().jsd, kstart)
           )
        3) q_out = utils.make_storage_from_shape(q_in.shape, origin,)
    """
    if not mask:
        n_dims = len(shape)
        if n_dims == 1:
            mask = (False, False, True)  # Assume 1D is a k-field
        else:
            mask = (n_dims * (True,)) + ((3 - n_dims) * (False,))
    storage = gt4py.storage.zeros(
        shape,
        dtype,
        backend=backend,
        aligned_index=_translate_origin(origin, mask),
        dimensions=_mask_to_dimensions(mask, shape),
    )
    return storage


def make_storage_dict(
    data: Field,
    shape: Optional[Tuple[int, ...]] = None,
    origin: Tuple[int, ...] = origin,
    start: Tuple[int, ...] = (0, 0, 0),
    dummy: Optional[Tuple[int, ...]] = None,
    names: Optional[List[str]] = None,
    axis: int = 2,
    *,
    backend: str,
) -> Dict[str, "Field"]:
    assert names is not None, "for 4d variable storages, specify a list of names"
    if shape is None:
        shape = data.shape
    data_dict: Dict[str, Field] = dict()
    for i in range(data.shape[3]):
        data_dict[names[i]] = make_storage_data(
            squeeze(data[:, :, :, i]),
            shape,
            origin=origin,
            start=start,
            dummy=dummy,
            axis=axis,
            backend=backend,
        )
    return data_dict


def storage_dict(st_dict, names, shape, origin, *, backend: str):
    for name in names:
        st_dict[name] = make_storage_from_shape(shape, origin, backend=backend)


def get_kstarts(column_info, npz):
    compare = None
    kstarts = []
    for k in range(npz):
        column_vals = {}
        for q, v in column_info.items():
            if k < len(v):
                column_vals[q] = v[k]
        if column_vals != compare:
            kstarts.append(k)
            compare = column_vals
    for i in range(len(kstarts) - 1):
        kstarts[i] = (kstarts[i], kstarts[i + 1] - kstarts[i])
    kstarts[-1] = (kstarts[-1], npz - kstarts[-1])
    return kstarts


def k_split_run(func, data, k_indices, splitvars_values):
    for ki, nk in k_indices:
        splitvars = {}
        for name, value_array in splitvars_values.items():
            splitvars[name] = value_array[ki]
        data.update(splitvars)
        data["kstart"] = ki
        data["nk"] = nk
        pace_log.debug(
            "Running kstart: {}, num k:{}, variables:{}".format(ki, nk, splitvars)
        )
        func(**data)


def asarray(array, to_type=np.ndarray, dtype=None, order=None):
    if cp and (isinstance(array, list)):
        if to_type is np.ndarray:
            order = "F" if order is None else order
            return cp.asnumpy(array, order=order)
        else:
            return cp.asarray(array, dtype, order)
    elif isinstance(array, list):
        if to_type is np.ndarray:
            return np.asarray(array, dtype, order)
        else:
            return cp.asarray(array, dtype, order)
    if cp and (
        isinstance(array, memoryview)
        or (
            hasattr(array, "data")
            and isinstance(array.data, (cp.ndarray, cp.cuda.memory.MemoryPointer))
        )
    ):
        if to_type is np.ndarray:
            order = "F" if order is None else order
            return cp.asnumpy(array, order=order)
        else:
            return cp.asarray(array, dtype, order)
    else:
        if to_type is np.ndarray:
            return np.asarray(array, dtype, order)
        else:
            return cp.asarray(array, dtype, order)


def is_gpu_backend(backend: str) -> bool:
    return gt4py.cartesian.backend.from_name(backend).storage_info["device"] == "gpu"


def zeros(shape, dtype=Float, *, backend: str):
    storage_type = cp.ndarray if is_gpu_backend(backend) else np.ndarray
    xp = cp if cp and storage_type is cp.ndarray else np
    return xp.zeros(shape, dtype=dtype)


def sum(array, axis=None, dtype=Float, out=None, keepdims=False):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.sum(array, axis, dtype, out, keepdims)


def repeat(array, repeats, axis=None):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.repeat(array, repeats, axis)


def index(array, key):
    return asarray(array, type(key))[key]


def moveaxis(array, source: int, destination: int):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.moveaxis(array, source, destination)


def tile(array, reps: Union[int, Tuple[int, ...]]):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.tile(array, reps)


def squeeze(array, axis: Union[int, Tuple[int]] = None):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.squeeze(array, axis)


def reshape(array, new_shape):
    if array.shape != new_shape:
        old_dims = len(array.shape)
        new_dims = len(new_shape)
        if old_dims < new_dims:
            # Upcast using repeat...
            if old_dims == 2:  # IJ -> IJK
                return repeat(array[:, :, np.newaxis], new_shape[2], axis=2)
            else:  # K -> IJK
                arr_2d = repeat(array[:, np.newaxis], new_shape[1], axis=1)
                return repeat(arr_2d[:, :, np.newaxis], new_shape[2], axis=2)
        else:
            return array.reshape(new_shape)
    return array


def unique(
    array,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Union[int, Tuple[int]] = None,
):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.unique(array, return_index, return_inverse, return_counts, axis)


def stack(tup, axis: int = 0, out=None):
    array_tup = []
    for array in tup:
        array_tup.append(array)
    xp = cp if cp and type(array_tup[0]) is cp.ndarray else np
    return xp.stack(array_tup, axis, out)


def device_sync(backend: str) -> None:
    if cp and is_gpu_backend(backend):
        cp.cuda.Device(0).synchronize()


def split_cartesian_into_storages(var: np.ndarray) -> Sequence[np.ndarray]:
    """
    Provided a storage of dims [X_DIM, Y_DIM, CARTESIAN_DIM]
         or [X_INTERFACE_DIM, Y_INTERFACE_DIM, CARTESIAN_DIM]
    Split it into separate 2D storages for each cartesian
    dimension, and return these in a list.
    """
    var_data = []
    for cart in range(3):
        var_data.append(
            asarray(var, type(var))[:, :, cart],
        )
    return var_data
