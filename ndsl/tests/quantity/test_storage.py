import numpy as np
import pytest

from ndsl.quantity import Quantity


try:
    import gt4py
except ImportError:
    gt4py = None
try:
    import cupy as cp
except ImportError:
    cp = None


@pytest.fixture
def extent_1d():
    return 5


@pytest.fixture(params=[0, 3])
def n_halo(request):
    return request.param


@pytest.fixture(params=[3])
def n_dims(request):
    return request.param


@pytest.fixture
def extent(extent_1d, n_dims):
    return (extent_1d,) * n_dims


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture
def units():
    return "m"


@pytest.fixture
def dims(n_dims):
    return tuple(f"dimension_{dim}" for dim in range(n_dims))


@pytest.fixture
def origin(n_halo, n_dims):
    return (n_halo,) * n_dims


@pytest.fixture
def data(n_halo, extent_1d, n_dims, numpy, dtype):
    shape = (n_halo * 2 + extent_1d,) * n_dims
    return numpy.zeros(shape, dtype=dtype)


@pytest.fixture
def quantity(data, origin, extent, dims, units):
    return Quantity(data, origin=origin, extent=extent, dims=dims, units=units)


def test_numpy(quantity, backend):
    if "cupy" in backend:
        assert quantity.np is cp
    else:
        assert quantity.np is np


@pytest.mark.skipif(gt4py is None, reason="requires gt4py")
def test_modifying_numpy_data_modifies_view():
    shape = (6, 6)
    data = np.zeros(shape, dtype=float)
    quantity = Quantity(
        data,
        origin=(0, 0),
        extent=shape,
        dims=["dim1", "dim2"],
        units="units",
        gt4py_backend="numpy",
    )
    assert np.all(quantity.data == 0)
    quantity.data[0, 0] = 1
    quantity.data[2, 2] = 5
    quantity.data[4, 4] = 3
    assert quantity.view[0, 0] == 1
    assert quantity.view[2, 2] == 5
    assert quantity.view[4, 4] == 3
    assert quantity.data[0, 0] == 1
    assert quantity.data[2, 2] == 5
    assert quantity.data[4, 4] == 3


@pytest.mark.parametrize("backend", ["gt4py_numpy", "gt4py_cupy"], indirect=True)
def test_data_exists(quantity, backend):
    if "numpy" in backend:
        assert isinstance(quantity.data, np.ndarray)
    else:
        assert isinstance(quantity.data, cp.ndarray)


@pytest.mark.parametrize("backend", ["numpy", "cupy"], indirect=True)
def test_accessing_data_does_not_break_view(
    data, origin, extent, dims, units, gt4py_backend
):
    quantity = Quantity(
        data,
        origin=origin,
        extent=extent,
        dims=dims,
        units=units,
        gt4py_backend=gt4py_backend,
    )
    quantity.data[origin] = -1.0
    assert quantity.data[origin] == quantity.view[tuple(0 for _ in origin)]


# run using cupy backend even though unused, to mark this as a "gpu" test
@pytest.mark.parametrize("backend", ["cupy"], indirect=True)
def test_numpy_data_becomes_cupy_with_gpu_backend(
    data, origin, extent, dims, units, gt4py_backend
):
    cpu_data = np.zeros(data.shape)
    quantity = Quantity(
        cpu_data,
        origin=origin,
        extent=extent,
        dims=dims,
        units=units,
        gt4py_backend=gt4py_backend,
    )
    assert isinstance(quantity.data, cp.ndarray)
