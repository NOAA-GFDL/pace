import numpy as np
import pytest

from ndsl.constants import X_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM
from ndsl.halo.rotate import rotate_scalar_data, rotate_vector_data


@pytest.fixture
def start_data(request, numpy):
    if isinstance(request.param, tuple):
        return tuple(numpy.asarray(item) for item in request.param)
    else:
        return numpy.asarray(request.param)


@pytest.mark.parametrize(
    "start_data, n_clockwise_rotations, dims, target_data",
    [
        pytest.param(
            np.array([1.0]),
            0,
            [Z_DIM],
            np.array([1.0]),
            id="1_value_no_rotation",
        ),
        pytest.param(
            np.array([1.0]),
            1,
            [Z_DIM],
            np.array([1.0]),
            id="1_value_1_rotation",
        ),
        pytest.param(
            np.array([1.0, 2.0]),
            1,
            [X_DIM],
            np.array([2.0, 1.0]),
            id="1d_x_one_rotation",
        ),
        pytest.param(
            np.array([1.0, 2.0]),
            2,
            [X_DIM],
            np.array([2.0, 1.0]),
            id="1d_x_two_rotations",
        ),
        pytest.param(
            np.array([1.0, 2.0]),
            1,
            [Y_DIM],
            np.array([1.0, 2.0]),
            id="1d_y_one_rotation",
        ),
        pytest.param(
            np.array([1.0, 2.0]),
            2,
            [Y_DIM],
            np.array([2.0, 1.0]),
            id="1d_y_two_rotations",
        ),
        pytest.param(
            np.zeros([2, 3]),
            0,
            [X_DIM, Y_DIM],
            np.zeros([2, 3]),
            id="2d_no_rotation",
        ),
        pytest.param(
            np.zeros([2, 3]),
            1,
            [X_DIM, Y_DIM],
            np.zeros([3, 2]),
            id="2d_1_rotation",
        ),
        pytest.param(
            np.zeros([2, 3]),
            2,
            [X_DIM, Y_DIM],
            np.zeros([2, 3]),
            id="2d_2_rotations",
        ),
        pytest.param(
            np.zeros([2, 3]),
            3,
            [X_DIM, Y_DIM],
            np.zeros([3, 2]),
            id="2d_3_rotations",
        ),
        pytest.param(
            np.arange(5)[:, None],
            1,
            [X_DIM, Y_DIM],
            np.arange(5)[None, ::-1],
            id="2d_x_increasing_values",
        ),
        pytest.param(
            np.arange(5)[:, None],
            2,
            [X_DIM, Y_DIM],
            np.arange(5)[::-1, None],
            id="2d_x_increasing_values_double_rotate",
        ),
        pytest.param(
            np.arange(5)[None, :],
            1,
            [X_DIM, Y_DIM],
            np.arange(5)[:, None],
            id="2d_y_increasing_values",
        ),
    ],
    indirect=["start_data"],
)
def test_rotate_scalar_data(
    start_data, n_clockwise_rotations, dims, numpy, target_data
):
    result = rotate_scalar_data(start_data, dims, numpy, n_clockwise_rotations)
    numpy.testing.assert_array_equal(result, target_data)


@pytest.mark.parametrize(
    "start_data, n_clockwise_rotations, dims, target_data",
    [
        pytest.param(
            (np.array([1.0]), np.array([1.0])),
            0,
            [Z_DIM],
            (np.array([1.0]), np.array([1.0])),
            id="scalar_no_rotation",
        ),
        pytest.param(
            (np.array([1.0]), np.array([1.0])),
            1,
            [Z_DIM],
            (np.array([1.0]), np.array([-1.0])),
            id="scalar_1_rotation",
        ),
        pytest.param(
            (np.array([1.0]), np.array([1.0])),
            2,
            [Z_DIM],
            (np.array([-1.0]), np.array([-1.0])),
            id="scalar_2_rotations",
        ),
        pytest.param(
            (np.array([1.0]), np.array([1.0])),
            3,
            [Z_DIM],
            (np.array([-1.0]), np.array([1.0])),
            id="scalar_3_rotations",
        ),
        pytest.param(
            (np.ones([3, 2]), np.ones([2, 3])),
            3,
            [Y_INTERFACE_DIM, X_DIM],
            (np.ones([3, 2]) * -1, np.ones([2, 3])),
            id="2d_array_flat_values",
        ),
        pytest.param(
            (np.arange(5)[:, None], np.arange(5)[None, :]),
            1,
            [X_DIM, Y_DIM],
            (np.arange(5)[:, None], np.arange(5)[None, ::-1] * -1),
            id="2d_array_increasing_values",
        ),
    ],
    indirect=["start_data"],
)
def test_rotate_vector_data(
    start_data, n_clockwise_rotations, dims, numpy, target_data
):
    x_data, y_data = start_data
    x_target, y_target = target_data
    x_result, y_result = rotate_vector_data(
        x_data, y_data, n_clockwise_rotations, dims, numpy
    )
    numpy.testing.assert_array_equal(x_result, x_target)
    numpy.testing.assert_array_equal(y_result, y_target)
