import unittest.mock

from pace.dsl.dace.dace_config import DaceConfig, _determine_compiling_ranks
from pace.dsl.dace.orchestration import (
    DaCeOrchestration,
    orchestrate,
    orchestrate_function,
)
from pace.util.communicator import CubedSpherePartitioner, TilePartitioner


"""
Tests that the dace configuration pace.dsl.dace.dace_config
which determines whether we use dace to run wrapped functions.
"""


def test_orchestrate_function_calls_dace():
    def foo():
        pass

    dace_config = DaceConfig(
        communicator=None,
        backend="gtc:dace",
        orchestration=DaCeOrchestration.BuildAndRun,
    )
    wrapped = orchestrate_function(config=dace_config)(foo)
    with unittest.mock.patch(
        "pace.dsl.dace.orchestration._call_sdfg"
    ) as mock_call_sdfg:
        wrapped()
    assert mock_call_sdfg.called
    assert mock_call_sdfg.call_args.args[0].f == foo


def test_orchestrate_function_does_not_call_dace():
    def foo():
        pass

    dace_config = DaceConfig(
        communicator=None,
        backend="gtc:dace",
        orchestration=DaCeOrchestration.Python,
    )
    wrapped = orchestrate_function(config=dace_config)(foo)
    with unittest.mock.patch(
        "pace.dsl.dace.orchestration._call_sdfg"
    ) as mock_call_sdfg:
        wrapped()
    assert not mock_call_sdfg.called


def test_orchestrate_calls_dace():
    dace_config = DaceConfig(
        communicator=None,
        backend="gtc:dace",
        orchestration=DaCeOrchestration.BuildAndRun,
    )

    class A:
        def __init__(self):
            orchestrate(obj=self, config=dace_config, method_to_orchestrate="foo")

        def foo(self):
            pass

    with unittest.mock.patch(
        "pace.dsl.dace.orchestration._call_sdfg"
    ) as mock_call_sdfg:
        a = A()
        a.foo()
    assert mock_call_sdfg.called


def test_orchestrate_does_not_call_dace():
    dace_config = DaceConfig(
        communicator=None,
        backend="gtc:dace",
        orchestration=DaCeOrchestration.Python,
    )

    class A:
        def __init__(self):
            orchestrate(obj=self, config=dace_config, method_to_orchestrate="foo")

        def foo(self):
            pass

    with unittest.mock.patch(
        "pace.dsl.dace.orchestration._call_sdfg"
    ) as mock_call_sdfg:
        a = A()
        a.foo()
    assert not mock_call_sdfg.called


def test_orchestrate_distributed_build():
    dummy_dace_config = DaceConfig(
        communicator=None,
        backend="gtc:dace",
        orchestration=DaCeOrchestration.BuildAndRun,
    )

    def _does_compile(rank, partitioner) -> bool:
        dummy_dace_config.layout = partitioner.layout
        dummy_dace_config.rank_size = partitioner.layout[0] * partitioner.layout[1] * 6
        dummy_dace_config.my_rank = rank
        return _determine_compiling_ranks(dummy_dace_config, partitioner)

    # (1, 1) layout, one rank which compiles
    cube_partitioner_11 = CubedSpherePartitioner(TilePartitioner((1, 1)))
    assert _does_compile(0, cube_partitioner_11)
    assert not _does_compile(1, cube_partitioner_11)  # not compiling face

    # (2, 2) layout, 4 ranks, all compiling
    cube_partitioner_22 = CubedSpherePartitioner(TilePartitioner((2, 2)))
    assert _does_compile(0, cube_partitioner_22)
    assert _does_compile(1, cube_partitioner_22)
    assert _does_compile(2, cube_partitioner_22)
    assert _does_compile(3, cube_partitioner_22)
    assert not _does_compile(4, cube_partitioner_22)  # not compiling face

    # (3, 3) layout, 9 ranks, all compiling
    cube_partitioner_33 = CubedSpherePartitioner(TilePartitioner((3, 3)))
    assert _does_compile(0, cube_partitioner_33)
    assert _does_compile(1, cube_partitioner_33)
    assert _does_compile(2, cube_partitioner_33)
    assert _does_compile(3, cube_partitioner_33)
    assert _does_compile(4, cube_partitioner_33)
    assert _does_compile(5, cube_partitioner_33)
    assert _does_compile(6, cube_partitioner_33)
    assert _does_compile(7, cube_partitioner_33)
    assert _does_compile(8, cube_partitioner_33)
    assert not _does_compile(9, cube_partitioner_33)  # not compiling face

    # (4, 4) layout, 16 ranks,
    # expecting compiling:0, 1, 2, 3, 4, 5, 7, 12, 13, 15
    cube_partitioner_44 = CubedSpherePartitioner(TilePartitioner((4, 4)))
    assert _does_compile(0, cube_partitioner_44)
    assert _does_compile(1, cube_partitioner_44)
    assert _does_compile(4, cube_partitioner_44)
    assert _does_compile(5, cube_partitioner_44)
    assert _does_compile(7, cube_partitioner_44)
    assert _does_compile(12, cube_partitioner_44)
    assert _does_compile(13, cube_partitioner_44)
    assert _does_compile(15, cube_partitioner_44)
    assert not _does_compile(2, cube_partitioner_44)  # same code path as 3
    assert not _does_compile(6, cube_partitioner_44)  # same code path as 5
    assert not _does_compile(8, cube_partitioner_44)  # same code path as 4
    assert not _does_compile(11, cube_partitioner_44)  # same code path as 7
    assert not _does_compile(16, cube_partitioner_44)  # not compiling face

    # For a few other layouts, we check that we always have 9 compiling ranks
    for layout in [(5, 5), (10, 10), (20, 20)]:
        partition = CubedSpherePartitioner(TilePartitioner(layout))
        compiling = 0
        for i in range(layout[0] * layout[1] * 6):
            compiling += 1 if _does_compile(i, partition) else 0
        assert compiling == 9
