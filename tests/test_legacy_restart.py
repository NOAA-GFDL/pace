from datetime import datetime
import os
import xarray as xr
import numpy as np
import pytest
import fv3util
import fv3util._legacy_restart

TEST_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(TEST_DIRECTORY, "data")


@pytest.fixture
def coupler_res_file_and_time():
    return os.path.join(DATA_DIRECTORY, 'coupler.res'), datetime(2016, 8, 3)


def test_get_current_date_from_coupler_res(coupler_res_file_and_time):
    filename, current_time = coupler_res_file_and_time
    with open(filename, 'r') as f:
        result = fv3util.io.get_current_date_from_coupler_res(f)
    assert result == current_time


@pytest.fixture
def data_array():
    return xr.DataArray(
        np.random.randn(2, 3),
        dims=['x', 'y'],
        attrs={'units': 'm'}
    )


@pytest.fixture(params=['empty', '1_dim', '2_dims'])
def new_dims(request):
    if request.param == 'empty':
        return ()
    elif request.param == '1_dim':
        return ('dim1',)
    elif request.param == '2_dims':
        return ('dim_2', 'dim_1')
    else:
        raise NotImplementedError()


@pytest.fixture
def result_dims(data_array, new_dims):
    kept_dims = len(data_array.dims) - len(new_dims)
    return tuple(list(data_array.dims[:kept_dims]) + list(new_dims))


def test_apply_dims(data_array, new_dims, result_dims):
    result = fv3util._legacy_restart.apply_dims(data_array, new_dims)
    np.testing.assert_array_equal(result.values, data_array.values)
    assert result.dims == result_dims
    assert result.attrs == data_array.attrs


@pytest.mark.parametrize(
    "old_dict, key_mapping, new_dict",
    [
        pytest.param(
            {}, {}, {},
            id='empty_dict',
        ),
        pytest.param(
            {'key1': 1, 'key2': 2}, {}, {'key1': 1, 'key2': 2},
            id='empty_map',
        ),
        pytest.param(
            {'key1': 1, 'key2': 2}, {'key1': 'key_1'}, {'key_1': 1, 'key2': 2},
            id='one_item_map',
        ),
        pytest.param(
            {'key1': 1, 'key2': 2}, {'key3': 'key_3'}, {'key1': 1, 'key2': 2},
            id='map_not_in_dict',
        ),
        pytest.param(
            {'key1': 1, 'key2': 2}, {'key1': 'key_1', 'key2': 'key_2'}, {'key_1': 1, 'key_2': 2},
            id='two_item_map',
        ),
    ]
)
def test_map_keys(old_dict, key_mapping, new_dict):
    result = fv3util._legacy_restart.map_keys(old_dict, key_mapping)
    assert result == new_dict


@pytest.mark.parametrize(
    "rank, total_ranks, suffix",
    [
        pytest.param(
            0, 6, ".tile1.nc",
            id='first_tile',
        ),
        pytest.param(
            2, 6, ".tile3.nc",
            id='third_tile',
        ),
        pytest.param(
            2, 24, ".tile1.nc.0002",
            id='third_subtile',
        ),
        pytest.param(
            6, 24, ".tile2.nc.0002",
            id='third_subtile_second_tile',
        ),
    ]
)
def test_get_rank_suffix(rank, total_ranks, suffix):
    result = fv3util._legacy_restart.get_rank_suffix(rank, total_ranks)
    assert result == suffix


@pytest.mark.parametrize(
    'invalid_total_ranks', [5, 7, 9, 23]
)
def test_get_rank_suffix_invalid_total_ranks(invalid_total_ranks):
    with pytest.raises(ValueError):
        # total_ranks should be multiple of 6
        fv3util._legacy_restart.get_rank_suffix(0, invalid_total_ranks)
