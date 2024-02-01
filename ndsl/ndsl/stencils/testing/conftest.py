import os
import re
from typing import Optional, Tuple

import f90nml
import pytest
import xarray as xr
import yaml
from Comm.communicator import Communicator, CubedSphereCommunicator, TileCommunicator
from Comm.mpi import MPI
from Comm.partitioner import CubedSpherePartitioner, TilePartitioner

import ndsl.dsl
from ndsl.dsl.dace.dace_config import DaceConfig
from ndsl.namelist import Namelist
from ndsl.stencils.testing import ParallelTranslate, TranslateGrid
from ndsl.stencils.testing.savepoint import SavepointCase, dataset_to_dict


@pytest.fixture()
def data_path(pytestconfig):
    return data_path_and_namelist_filename_from_config(pytestconfig)


def data_path_and_namelist_filename_from_config(config) -> Tuple[str, str]:
    data_path = config.getoption("data_path")
    namelist_filename = os.path.join(data_path, "input.nml")
    return data_path, namelist_filename


@pytest.fixture
def threshold_overrides(pytestconfig):
    return thresholds_from_file(pytestconfig)


def thresholds_from_file(config):
    thresholds_file = config.getoption("threshold_overrides_file")
    if thresholds_file is None:
        return None
    return yaml.safe_load(open(thresholds_file, "r"))


def get_test_class(test_name):
    translate_class_name = f"Translate{test_name.replace('-', '_')}"
    try:
        return_class = getattr(translate, translate_class_name)  # noqa: F821
    except AttributeError as err:
        if translate_class_name in err.args[0]:
            return_class = None
        else:
            raise err
    return return_class


def is_parallel_test(test_name):
    test_class = get_test_class(test_name)
    if test_class is None:
        return False
    else:
        return issubclass(test_class, ParallelTranslate)


def get_test_class_instance(test_name, grid, namelist, stencil_factory):
    translate_class = get_test_class(test_name)
    if translate_class is None:
        return None
    else:
        return translate_class(grid, namelist, stencil_factory)


def get_all_savepoint_names(metafunc, data_path):
    only_names = metafunc.config.getoption("which_modules")
    if only_names is None:
        savepoint_names = [
            fname[:-3] for fname in os.listdir(data_path) if re.match(r".*\.nc", fname)
        ]
        savepoint_names = [s[:-3] for s in savepoint_names if s.endswith("-In")]
    else:
        savepoint_names = set(only_names.split(","))
        savepoint_names.discard("")
    skip_names = metafunc.config.getoption("skip_modules")
    if skip_names is not None:
        savepoint_names.difference_update(skip_names.split(","))
    return savepoint_names


def get_sequential_savepoint_names(metafunc, data_path):
    all_names = get_all_savepoint_names(metafunc, data_path)
    sequential_names = []
    for name in all_names:
        if not is_parallel_test(name):
            sequential_names.append(name)
    return sequential_names


def get_parallel_savepoint_names(metafunc, data_path):
    all_names = get_all_savepoint_names(metafunc, data_path)
    parallel_names = []
    for name in all_names:
        if is_parallel_test(name):
            parallel_names.append(name)
    return parallel_names


def get_ranks(metafunc, layout):
    only_rank = metafunc.config.getoption("which_rank")
    dperiodic = metafunc.config.getoption("dperiodic")
    if only_rank is None:
        if dperiodic:
            total_ranks = layout[0] * layout[1]
        else:
            total_ranks = 6 * layout[0] * layout[1]
        return range(total_ranks)
    else:
        return [int(only_rank)]


def get_namelist(namelist_filename):
    return Namelist.from_f90nml(f90nml.read(namelist_filename))


def get_config(backend: str, communicator: Optional[Communicator]):
    stencil_config = ndsl.dsl.stencil.StencilConfig(
        compilation_config=ndsl.dsl.stencil.CompilationConfig(
            backend=backend, rebuild=False, validate_args=True
        ),
        dace_config=DaceConfig(
            communicator=communicator,
            backend=backend,
        ),
    )
    return stencil_config


def sequential_savepoint_cases(metafunc, data_path, namelist_filename, *, backend: str):
    savepoint_names = get_sequential_savepoint_names(metafunc, data_path)
    namelist = get_namelist(namelist_filename)
    stencil_config = get_config(backend, None)
    ranks = get_ranks(metafunc, namelist.layout)
    compute_grid = metafunc.config.getoption("compute_grid")
    dperiodic = metafunc.config.getoption("dperiodic")
    return _savepoint_cases(
        savepoint_names,
        ranks,
        stencil_config,
        namelist,
        backend,
        data_path,
        compute_grid,
        dperiodic,
    )


def _savepoint_cases(
    savepoint_names,
    ranks,
    stencil_config,
    namelist,
    backend,
    data_path,
    compute_grid: bool,
    dperiodic: bool,
):
    return_list = []
    ds_grid: xr.Dataset = xr.open_dataset(os.path.join(data_path, "Grid-Info.nc")).isel(
        savepoint=0
    )
    for rank in ranks:
        grid = TranslateGrid(
            dataset_to_dict(ds_grid.isel(rank=rank)),
            rank=rank,
            layout=namelist.layout,
            backend=backend,
        ).python_grid()
        if compute_grid:
            compute_grid_data(grid, namelist, backend, namelist.layout, dperiodic)
        stencil_factory = ndsl.dsl.stencil.StencilFactory(
            config=stencil_config,
            grid_indexing=grid.grid_indexing,
        )
        for test_name in sorted(list(savepoint_names)):
            testobj = get_test_class_instance(
                test_name, grid, namelist, stencil_factory
            )
            n_calls = xr.open_dataset(
                os.path.join(data_path, f"{test_name}-In.nc")
            ).dims["savepoint"]
            for i_call in range(n_calls):
                return_list.append(
                    SavepointCase(
                        savepoint_name=test_name,
                        data_dir=data_path,
                        rank=rank,
                        i_call=i_call,
                        testobj=testobj,
                        grid=grid,
                    )
                )
    return return_list


def compute_grid_data(grid, namelist, backend, layout, dperiodic):
    grid.make_grid_data(
        npx=namelist.npx,
        npy=namelist.npy,
        npz=namelist.npz,
        communicator=get_communicator(MPI.COMM_WORLD, layout, dperiodic),
        backend=backend,
    )


def parallel_savepoint_cases(
    metafunc, data_path, namelist_filename, mpi_rank, *, backend: str, comm
):
    namelist = get_namelist(namelist_filename)
    dperiodic = metafunc.config.getoption("dperiodic")
    communicator = get_communicator(comm, namelist.layout, dperiodic)
    stencil_config = get_config(backend, communicator)
    savepoint_names = get_parallel_savepoint_names(metafunc, data_path)
    compute_grid = metafunc.config.getoption("compute_grid")
    return _savepoint_cases(
        savepoint_names,
        [mpi_rank],
        stencil_config,
        namelist,
        backend,
        data_path,
        compute_grid,
        dperiodic,
    )


def pytest_generate_tests(metafunc):
    backend = metafunc.config.getoption("backend")
    if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
        if metafunc.function.__name__ == "test_parallel_savepoint":
            generate_parallel_stencil_tests(metafunc, backend=backend)
    elif metafunc.function.__name__ == "test_sequential_savepoint":
        generate_sequential_stencil_tests(metafunc, backend=backend)


def generate_sequential_stencil_tests(metafunc, *, backend: str):
    data_path, namelist_filename = data_path_and_namelist_filename_from_config(
        metafunc.config
    )
    savepoint_cases = sequential_savepoint_cases(
        metafunc, data_path, namelist_filename, backend=backend
    )
    metafunc.parametrize(
        "case", savepoint_cases, ids=[str(item) for item in savepoint_cases]
    )


def generate_parallel_stencil_tests(metafunc, *, backend: str):
    data_path, namelist_filename = data_path_and_namelist_filename_from_config(
        metafunc.config
    )
    # get MPI environment
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    savepoint_cases = parallel_savepoint_cases(
        metafunc,
        data_path,
        namelist_filename,
        mpi_rank,
        backend=backend,
        comm=comm,
    )
    metafunc.parametrize(
        "case", savepoint_cases, ids=[str(item) for item in savepoint_cases]
    )


def get_communicator(comm, layout, dperiodic):
    if (MPI.COMM_WORLD.Get_size() > 1) and (not dperiodic):
        partitioner = CubedSpherePartitioner(TilePartitioner(layout))
        communicator = CubedSphereCommunicator(comm, partitioner)
    else:
        partitioner = TilePartitioner(layout)
        communicator = TileCommunicator(comm, partitioner)
    return communicator


@pytest.fixture()
def print_failures(pytestconfig):
    return pytestconfig.getoption("print_failures")


@pytest.fixture()
def failure_stride(pytestconfig):
    return int(pytestconfig.getoption("failure_stride"))


@pytest.fixture()
def compute_grid(pytestconfig):
    return pytestconfig.getoption("compute_grid")


@pytest.fixture()
def dperiodic(pytestconfig):
    return pytestconfig.getoption("dperiodic")
